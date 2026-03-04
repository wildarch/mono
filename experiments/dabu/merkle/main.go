package main

import (
	"crypto/sha256"
	"flag"
	"io"
	"log"
	"os"
	"path/filepath"
	"slices"
)

func hashDir(path string, f *os.File) ([]byte, error) {
	names, err := f.Readdirnames(0)
	if err != nil {
		return nil, err
	}

	// Deterministic order for the files inside the directory
	slices.Sort(names)

	h := sha256.New()
	for _, name := range names {
		h.Write([]byte(name))
		subh, err := hash(filepath.Join(path, name))
		if err != nil {
			return nil, err
		}

		h.Write(subh)
	}

	sum := h.Sum(nil)
	log.Printf("hash for dir %s: %x", path, sum)
	return sum, nil
}

func hashFile(path string, f *os.File) ([]byte, error) {
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return nil, err
	}

	sum := h.Sum(nil)
	log.Printf("hash for file %s: %x", path, sum)
	return sum, nil
}

func hash(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}

	if stat.IsDir() {
		return hashDir(path, f)
	} else {
		return hashFile(path, f)
	}
}

func main() {
	rootPath := flag.String("root", ".", "root directory for the merkle tree")
	flag.Parse()

	log.Printf("root dir: %s", *rootPath)
	h, err := hash(*rootPath)
	if err != nil {
		log.Fatalf("failed to hash directory: %s", err.Error())
	}

	log.Printf("root hash: %x", h)
}
