package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"slices"
)

type Edge struct {
	Phony   bool
	Command string
	Inputs  []string
	Outputs []string
}

func build(edges []Edge, path string) error {
	// Find our target edge
	var target Edge
	targetFound := false
	for _, e := range edges {
		if slices.Contains(e.Outputs, path) {
			target = e
			targetFound = true
			break
		}
	}

	if !targetFound {
		_, err := os.Stat(path)
		if errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("file %s does not exist, and there is not rule to build it", path)
		}

		// File exists, so assume it is a build input rather than intermediate.
		return nil
	}

	// First build all the inputs
	for _, input := range target.Inputs {
		if err := build(edges, input); err != nil {
			return err
		}
	}

	if target.Phony {
		// No command to run
		return nil
	}

	log.Printf("run command for edge: %v", target.Command)
	cmd := exec.Command("sh", "-c", target.Command)
	err := cmd.Run()
	return err
}

func main() {
	if err := os.Chdir("experiments/dabu/ninja/build/"); err != nil {
		log.Fatal(err)
	}

	f, err := os.Open("/tmp/ninja.json")
	if err != nil {
		log.Fatal(err)
	}

	edges := make([]Edge, 0)
	dec := json.NewDecoder(f)
	for {
		var e Edge
		if err := dec.Decode(&e); err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		edges = append(edges, e)
	}

	build(edges, "dabu-ninja")
}
