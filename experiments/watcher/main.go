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

	deps := queryDeps(workspaceRoot, target)
	dirs := directories(deps)
	for _, d := range dirs {
		log.Printf("Directory to watch: %s", d)
	}

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		log.Fatalf("error initializing watcher: %s", err.Error())
	}
	defer watcher.Close()

	watch(watcher, workspaceRoot, target)
}

func watch(watcher *fsnotify.Watcher, workspaceRoot, target string) {
	buildTarget(workspaceRoot, target)
	// watch deps
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

func directories(files []string) []string {
	dirMap := make(map[string]int, 0)
	for _, f := range files {
		dir := filepath.Dir(f)
		dirMap[dir]++
	}

	dirs := make([]string, 0)
	for dir := range dirMap {
		dirs = append(dirs, dir)
	}
	return dirs
}
