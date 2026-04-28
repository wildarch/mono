package main

import (
	"bufio"
	"cmp"
	"crypto/sha256"
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

func noSID() SourceId {
	return [32]byte{0}
}

func isNoSID(s SourceId) bool {
	for _, b := range s {
		if b != 0 {
			return false
		}
	}

	return true
}

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

func (n *RepoNode) addSource(p string, sid SourceId) {
	dir, file := path.Split(p)
	if dir == "" {
		// Put into this directory
		for _, child := range n.Children {
			if child.Name == file {
				// Update existing file
				child.Node.SId = sid
				return
			}
		}

		// Make a new file in the directory
		entry := RepoDirEntry{file, &RepoNode{SId: sid}}
		n.Children = append(n.Children, entry)
		// Ensure deterministic sort order
		slices.SortFunc(n.Children, func(a, b RepoDirEntry) int {
			return cmp.Compare(a.Name, b.Name)
		})
		return
	}

	// Drop trailing slash
	dir = dir[:len(dir)-1]

	// Find the directory
	for _, child := range n.Children {
		if child.Name == dir {
			child.Node.addSource(file, sid)
			return
		}
	}

	// Make a new directory and recurse into it
	dirNode := &RepoNode{}
	dirNode.addSource(file, sid)

	// Attach to the current node
	entry := RepoDirEntry{dir, dirNode}
	n.Children = append(n.Children, entry)
	// Ensure deterministic sort order
	slices.SortFunc(n.Children, func(a, b RepoDirEntry) int {
		return cmp.Compare(a.Name, b.Name)
	})
}

func (n *RepoNode) getSource(p string) SourceId {
	log.Printf("looking for source file %s", p)
	dir, file := path.Split(p)
	log.Printf("dir %s file %s", dir, file)
	if dir == "" {
		// File is in the current directory
		for _, child := range n.Children {
			if child.Name == file {
				return child.Node.SId
			}
		}

		// File not found
		return noSID()
	}

	// Drop trailing slash
	dir = dir[:len(dir)-1]

	// Find the directory
	for _, child := range n.Children {
		if child.Name == dir {
			return child.Node.getSource(file)
		}
	}

	// Directory not found
	return noSID()
}

func (r *Repo) AddSource(p string, d []byte) {
	h := sha256.New()
	h.Write(d)
	sid := [32]byte(h.Sum(nil))
	r.Sources[sid] = d
	r.Root.addSource(p, sid)
}

func (r *Repo) Instantiate(root string, p string) error {
	sid := r.Root.getSource(p)
	if isNoSID(sid) {
		return fmt.Errorf("no such source file '%s'", p)
	}

	d, found := r.Sources[sid]
	if !found {
		return fmt.Errorf("no source data for sid %x", sid)
	}

	dir, _ := path.Split(p)
	if err := os.MkdirAll(path.Join(root, dir), 0755); err != nil {
		return err
	}

	if err := os.WriteFile(path.Join(root, p), d, 0755); err != nil {
		return err
	}

	return nil
}

func buildGitRepo(root string) (*Repo, error) {
	repo := &Repo{
		Sources: map[SourceId][]byte{},
		Root:    &RepoNode{}}

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
		f, err := os.Open(file)
		if err != nil {
			return nil, err
		}

		defer f.Close()
		data, err := io.ReadAll(f)
		if err != nil {
			return nil, err
		}

		repo.AddSource(file, data)
	}

	if err := git.Wait(); err != nil {
		return nil, err
	}

	return repo, nil
}

func main() {
	// Get all source files in repo
	repo, err := buildGitRepo(".")
	if err != nil {
		log.Fatal(err)
	}

	// Create root
	root := "/tmp/dabu-root"
	if err := os.RemoveAll(root); err != nil {
		log.Fatal(err)
	}

	if err := os.Mkdir(root, 0755); err != nil {
		log.Fatal(err)
	}

	if err := repo.Instantiate(root, "experiments/dabu/builder/CMakeLists.txt"); err != nil {
		log.Fatal(err)
	}

	// TODO: run cmake configure

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
}
