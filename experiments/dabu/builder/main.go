package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path"
	"slices"
)

type Edge struct {
	Phony   bool
	Command string
	Inputs  []string
	Outputs []string
}

func build(dir string, edges []Edge, p string) error {
	// Find our target edge
	var target Edge
	targetFound := false
	for _, e := range edges {
		if slices.Contains(e.Outputs, p) {
			target = e
			targetFound = true
			break
		}
	}

	if !targetFound {
		abs := p
		if !path.IsAbs(p) {
			abs = path.Join(dir, p)
		}

		_, err := os.Stat(abs)
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("file %s does not exist, and there is no rule to build it", abs)
		}

		// File exists, so assume it is a source file rather than intermediate.
		log.Printf("source file %s", p)
		return nil
	}

	if !target.Phony {
		log.Printf("building file %s", p)
	}

	// First build all the inputs
	for _, input := range target.Inputs {
		if err := build(dir, edges, input); err != nil {
			return err
		}
	}

	if target.Phony {
		// No command to run
		return nil
	}

	log.Printf("run command for edge: %v", target.Command)
	cmd := exec.Command("sh", "-c", target.Command)
	cmd.Dir = dir
	err := cmd.Run()
	return err
}

func buildNinjaHelperBinary() (string, error) {
	wd, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}

	build := path.Join(wd, "experiments/dabu/ninja/build")

	cmake := exec.Command("cmake", "--build", build)
	if err := cmake.Run(); err != nil {
		return "", err
	}

	return path.Join(build, "dabu-ninja"), nil
}

func ninjaEdges(ninja string, build string) ([]Edge, error) {
	cmd := exec.Command(ninja)
	cmd.Dir = build
	// Capture JSON on stdout, which we will parse.
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}

	// Start the command.
	if err := cmd.Start(); err != nil {
		return nil, err
	}

	// Parse the output
	dec := json.NewDecoder(out)
	edges := make([]Edge, 0)
	for {
		var e Edge
		if err := dec.Decode(&e); err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		edges = append(edges, e)
	}

	if err := cmd.Wait(); err != nil {
		return nil, err
	}

	return edges, nil
}

type SourceId [32]byte

type RepoDirEntry struct {
	Name string
	Node *RepoNode
}

type BuildRule struct {
	Inputs  []RepoNode
	Outputs []RepoNode
	Command []string
}

type RepoNode struct {
	Rule     *BuildRule     // If built from other files
	Children []RepoDirEntry // If directory
	SId      SourceId       // If source file
}

type Repo struct {
	Sources map[SourceId][]byte
	Root    *RepoNode
}

func buildGitRepo(root string) (*Repo, error) {
	repo := &Repo{}

	// Get list of files from git
	git := exec.Command("git", "ls-files")
	git.Dir = root
	out, err := git.StdoutPipe()
	if err != nil {
		return nil, err
	}

	if err := git.Start(); err != nil {
		return nil, err
	}

	scanner := bufio.NewScanner(out)
	for scanner.Scan() {
		file := scanner.Text()
		// TODO: actually put it in the repo
		log.Printf("file to put into repo: %s", file)
	}

	if err := git.Wait(); err != nil {
		return nil, err
	}

	return repo, nil
}

func main() {
	/*
		ninja, err := buildNinjaHelperBinary()
		if err != nil {
			log.Fatal(err)
		}

		buildDir := "experiments/dabu/ninja/build"
		edges, err := ninjaEdges(ninja, buildDir)
		if err != nil {
			log.Fatal(err)
		}

		if err := build(buildDir, edges, "dabu-ninja"); err != nil {
			log.Fatal(err)
		}
	*/

	// TODO: get all source files in repo
	_, err := buildGitRepo(".")
	if err != nil {
		log.Fatal(err)
	}
}
