package main

import (
	"bufio"
	"bytes"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/fsnotify/fsnotify"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Printf("usage: %s ROOT TARGET", os.Args[0])
		os.Exit(1)
	}
	workspaceRoot := os.Args[1]
	target := os.Args[2]
	log.Printf("Watching target: %s", target)

	watch(workspaceRoot, target)
}

func watch(workspaceRoot, target string) {
	isTest := strings.HasSuffix(target, "_test")
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		log.Fatalf("error initializing watcher: %s", err.Error())
	}
	defer watcher.Close()

	for {
		deps := queryDeps(workspaceRoot, target)
		setFilesToWatch(watcher, deps)
		if isTest {
			testTarget(workspaceRoot, target)
		} else {
			buildTarget(workspaceRoot, target)
		}
		select {
		case event, ok := <-watcher.Events:
			if !ok {
				return
			}
			log.Println("event:", event)
		case err, ok := <-watcher.Errors:
			if !ok {
				return
			}
			log.Println("error:", err)
		}
	}
}

func setFilesToWatch(watcher *fsnotify.Watcher, files []string) {
	// Tracks what directories we need to watch,
	// and if they have already been added to the watcher.
	dirsAdded := make(map[string]bool, 0)
	for _, f := range files {
		dir := filepath.Dir(f)
		dirsAdded[dir] = false
	}

	// Mark directories already watched, and remove stale directories
	for _, d := range watcher.WatchList() {
		_, inNewList := dirsAdded[d]
		if !inNewList {
			// Stale
			watcher.Remove(d)
		} else {
			// Already added to watcher
			dirsAdded[d] = true
		}
	}

	// Add new directories to watch
	for dir, alreadyAdded := range dirsAdded {
		if alreadyAdded {
			continue
		}
		watcher.Add(dir)
	}
}

func buildTarget(workspaceRoot, target string) {
	buildCmd := exec.Command(
		"bazel",
		"build",
		target)
	buildCmd.Dir = workspaceRoot
	buildCmd.Stdout = os.Stdout
	buildCmd.Stderr = os.Stderr
	err := buildCmd.Run()
	if err != nil {
		log.Printf("Build failed with error: %s", err.Error())
	}
}

func testTarget(workspaceRoot, target string) {
	testCmd := exec.Command(
		"bazel",
		// Use run instead of test to show more output
		"run",
		target)
	testCmd.Dir = workspaceRoot
	testCmd.Stdout = os.Stdout
	testCmd.Stderr = os.Stderr
	err := testCmd.Run()
	if err != nil {
		log.Printf("test failed with error: %s", err.Error())
	}
}

// Returns the paths of all files under workspaceRoot that target depends on.
func queryDeps(workspaceRoot, target string) []string {
	queryCmd := exec.Command(
		"bazel",
		"query",
		fmt.Sprintf("deps(%s)",
			target),
		"--output",
		"location")
	queryCmd.Dir = workspaceRoot
	queryCmd.Stderr = os.Stderr
	queryOut, err := queryCmd.Output()
	if err != nil {
		log.Fatalf("error analysing target: %s", err.Error())
	}

	files := make([]string, 0)
	scanner := bufio.NewScanner(bytes.NewReader(queryOut))
	for scanner.Scan() {
		line := scanner.Text()
		lineCol := strings.Index(line, ":")
		if lineCol == -1 {
			log.Fatalf("error parsing query results: %s", line)
		}
		filePath := line[:lineCol]
		if strings.HasPrefix(filePath, workspaceRoot) {
			files = append(files, filePath)
		}
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("error reading from scanner: %s", err.Error())
	}

	if len(files) == 0 {
		log.Fatalf("no dependencies found")
	}
	return files
}
