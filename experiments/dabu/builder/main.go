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
	"path/filepath"
	"slices"
	"strings"
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
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	return err
}

func cmakeConfigure(source string, build string) error {
	cmake := exec.Command("cmake", "-S", source, "-B", build, "-GNinja")
	cmake.Stdout = os.Stdout
	cmake.Stderr = os.Stderr
	return cmake.Run()
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
	Inputs  []*RepoNode
	Outputs []*RepoNode
	Dir     string
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

func expandPhony(input string, edges []Edge) []string {
	for _, edge := range edges {
		if !edge.Phony {
			continue
		}

		found := false
		for _, out := range edge.Outputs {
			if out == input {
				// This edge generates the output we want
				found = true
				break
			}
		}

		if !found {
			continue
		}

		results := make([]string, 0)
		for _, input := range edge.Inputs {
			results = append(results, expandPhony(input, edges)...)
		}

		return results
	}

	return []string{input}
}

func (r *Repo) AddNinjaRules(root string, buildDir string, edges []Edge) error {
	for _, edge := range edges {
		if edge.Phony {
			// Skip phony
			continue
		}

		rule := &BuildRule{Dir: buildDir}
		rule.Command = []string{"sh", "-c", edge.Command}
		log.Printf("command: %s", edge.Command)

		// map inputs
		for _, p := range edge.Inputs {
			for _, p := range expandPhony(p, edges) {
				if path.IsAbs(p) {
					if strings.HasPrefix(p, root) {
						// Source file
						rel, err := filepath.Rel(root, p)
						if err != nil {
							return err
						}

						p = rel
					} else {
						// Absolute path to something in the external build environment
						log.Printf("ignoring absolute path %s", p)
						continue
					}
				} else {
					// Relative to build dir
					p = path.Join(buildDir, p)
				}

				if p == buildDir {
					continue
				}

				input := r.Root.getNode(p)
				if input == nil {
					return fmt.Errorf("no such input: %s", p)
				}

				rule.Inputs = append(rule.Inputs, input)
			}
		}

		// map outputs
		for _, p := range edge.Outputs {
			p = path.Join(buildDir, p)
			log.Printf("output: %s", p)
			node := &RepoNode{Rule: rule}
			r.Root.addNode(p, node)
			rule.Outputs = append(rule.Outputs, node)
		}
	}

	return nil
}

func splitFirst(p string) (dir, sub string) {
	dir = ""
	before, after, ok := strings.Cut(p, "/")
	if !ok {
		sub = p
	} else {
		dir = before
		sub = after
	}

	return
}

func (n *RepoNode) addNode(p string, node *RepoNode) {
	dir, p := splitFirst(p)
	if dir == "" {
		// Put into this directory
		for _, child := range n.Children {
			if child.Name == p {
				// Update existing file
				child.Node = node
				return
			}
		}

		// Make a new file in the directory
		entry := RepoDirEntry{p, node}
		n.Children = append(n.Children, entry)
		// Ensure deterministic sort order
		slices.SortFunc(n.Children, func(a, b RepoDirEntry) int {
			return cmp.Compare(a.Name, b.Name)
		})
		return
	}

	// Find the directory
	for _, child := range n.Children {
		if child.Name == dir {
			child.Node.addNode(p, node)
			return
		}
	}

	// Make a new directory and recurse into it
	dirNode := &RepoNode{}
	dirNode.addNode(p, node)

	// Attach to the current node
	entry := RepoDirEntry{dir, dirNode}
	n.Children = append(n.Children, entry)
	// Ensure deterministic sort order
	slices.SortFunc(n.Children, func(a, b RepoDirEntry) int {
		return cmp.Compare(a.Name, b.Name)
	})
}

func (n *RepoNode) getNode(p string) *RepoNode {
	dir, p := splitFirst(p)
	if dir == "" {
		// File is in the current directory
		for _, child := range n.Children {
			if child.Name == p {
				return child.Node
			}
		}

		// File not found
		return nil
	}

	// Find the directory
	for _, child := range n.Children {
		if child.Name == dir {
			return child.Node.getNode(p)
		}
	}

	// Directory not found
	return nil
}

func (r *Repo) AddSource(p string, d []byte) {
	h := sha256.New()
	h.Write(d)
	sid := [32]byte(h.Sum(nil))
	r.Sources[sid] = d
	node := &RepoNode{SId: sid}
	r.Root.addNode(p, node)
}

func (r *Repo) Instantiate(root string, p string) error {
	node := r.Root.getNode(p)
	if node == nil || node.SId == noSID() {
		return fmt.Errorf("no such source file '%s'", p)
	}

	d, found := r.Sources[node.SId]
	if !found {
		return fmt.Errorf("no source data for sid %x", node.SId)
	}

	dir, _ := path.Split(p)
	if err := os.MkdirAll(path.Join(root, dir), 0755); err != nil {
		return err
	}

	out := path.Join(root, p)
	if err := os.WriteFile(out, d, 0755); err != nil {
		return err
	}

	log.Printf("wrote %s", out)
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

	sourceDir := "experiments/dabu/cmake-hello"
	if err := repo.Instantiate(root, path.Join(sourceDir, "CMakeLists.txt")); err != nil {
		log.Fatal(err)
	}

	// Touch hello.cpp
	f, err := os.Create(path.Join(root, sourceDir, "hello.cpp"))
	if err != nil {
		log.Fatal(err)
	}
	f.Close()

	// Make build dir
	buildDir := path.Join(root, sourceDir, "build")
	if err := os.MkdirAll(buildDir, 0755); err != nil {
		log.Fatal(err)
	}

	// Run cmake configure
	if err := cmakeConfigure(path.Join(root, sourceDir), buildDir); err != nil {
		log.Fatal(err)
	}

	ninja, err := buildNinjaHelperBinary()
	if err != nil {
		log.Fatal(err)
	}

	edges, err := ninjaEdges(ninja, buildDir)
	if err != nil {
		log.Fatal(err)
	}

	if err := repo.AddNinjaRules(root, path.Join(sourceDir, "build"), edges); err != nil {
		log.Fatal(err)
	}

	/*
		// TODO: extract input files from edges
		if err := repo.Instantiate(root, path.Join(sourceDir, "hello.cpp")); err != nil {
			log.Fatal(err)
		}

		if err := build(buildDir, edges, "hello"); err != nil {
			log.Fatal(err)
		}
	*/
}
