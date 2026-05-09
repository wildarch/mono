package main

import (
	"bufio"
	"cmp"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
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
	Path     string
	Rule     *BuildRule     // If built from other files
	Children []RepoDirEntry // If directory
	SId      SourceId       // If source file
}

type Repo struct {
	Root      *RepoNode
	sourceCAS string
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

		rule := &BuildRule{
			Dir:     buildDir,
			Command: []string{"sh", "-c", edge.Command}}
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
			node := &RepoNode{Rule: rule, Path: p}
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

func (r *Repo) sourceCASPath(sid SourceId) string {
	return path.Join(r.sourceCAS, fmt.Sprintf("%x", sid))
}

func (r *Repo) AddSource(p string, d []byte) error {
	h := sha256.New()
	h.Write(d)
	sid := [32]byte(h.Sum(nil))
	if err := os.WriteFile(r.sourceCASPath(sid), d, 0755); err != nil {
		return err
	}

	node := &RepoNode{Path: p, SId: sid}
	r.Root.addNode(p, node)
	return nil
}

func createOrReplaceSymlink(oldname string, newname string) error {
	err := os.Symlink(oldname, newname)
	if err == nil {
		return nil
	} else if errors.Is(err, fs.ErrExist) {
		// EXIST: Need to remove existing file/link first
		if err := os.Remove(newname); err != nil {
			return err
		}

		return createOrReplaceSymlink(oldname, newname)
	} else {
		// Some other error
		return err
	}
}

func (r *Repo) instantiate(root string, node *RepoNode) error {
	// Ensure the target directory exists
	dir, _ := path.Split(path.Join(root, node.Path))
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	if node.SId != noSID() {
		// Is a source file
		out := path.Join(root, node.Path)
		if err := createOrReplaceSymlink(r.sourceCASPath(node.SId), out); err != nil {
			return err
		}

		log.Printf("linked source file %s", out)
	} else if node.Rule != nil {
		// Can be built
		rule := node.Rule
		for _, input := range rule.Inputs {
			if err := r.instantiate(root, input); err != nil {
				return err
			}
		}

		log.Printf("building %s: %s (working dir %s)", node.Path, strings.Join(node.Rule.Command, " "), rule.Dir)
		cmd := exec.Command(rule.Command[0], rule.Command[1:]...)
		cmd.Dir = path.Join(root, rule.Dir)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return err
		}
	} else {
		return fmt.Errorf("not a source file and no build rule to generate file '%s'", node.Path)
	}

	return nil
}

func (r *Repo) Build(root string, p string) error {
	node := r.Root.getNode(p)
	if node == nil {
		return fmt.Errorf("no such target: %s", p)
	}

	return r.instantiate(root, node)
}

func addGitRepo(root string, repo *Repo) error {
	// Get list of files from git
	git := exec.Command("git", "ls-files")
	git.Dir = root
	out, err := git.StdoutPipe()
	if err != nil {
		return err
	}

	if err := git.Start(); err != nil {
		return err
	}

	scanner := bufio.NewScanner(out)
	for scanner.Scan() {
		file := scanner.Text()
		f, err := os.Open(file)
		if err != nil {
			return err
		}

		defer f.Close()
		data, err := io.ReadAll(f)
		if err != nil {
			return err
		}

		if err := repo.AddSource(file, data); err != nil {
			return err
		}
	}

	if err := git.Wait(); err != nil {
		return err
	}

	return nil
}

func main() {
	// Store source files here
	sourceCAS := "/tmp/dabu-src"
	if err := os.MkdirAll(sourceCAS, 0755); err != nil {
		log.Fatal(err)
	}

	repo := &Repo{
		Root:      &RepoNode{},
		sourceCAS: sourceCAS,
	}

	// Get all source files in repo
	if err := addGitRepo(".", repo); err != nil {
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
	if err := repo.Build(root, path.Join(sourceDir, "CMakeLists.txt")); err != nil {
		log.Fatal(err)
	}

	// HACK: build hello.cpp before configuring
	if err := repo.Build(root, path.Join(sourceDir, "hello.cpp")); err != nil {
		log.Fatal(err)
	}

	log.Printf("Configuring Ninja rules")

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

	log.Printf("Start build phase")

	helloPath := path.Join(sourceDir, "build/hello")
	if err := repo.Build(root, helloPath); err != nil {
		log.Fatal(err)
	}

	log.Printf("instantiated at %s", path.Join(root, helloPath))
}
