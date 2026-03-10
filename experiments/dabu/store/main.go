package main

import (
	"cmp"
	"crypto/sha256"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/rpc"
	"slices"
	"strings"

	"github.com/wildarch/mono/experiments/dabu/store/api"
)

type StoreEntry struct {
	data     []byte
	children []api.PutDirRequestChild
}

type Store struct {
	cas map[api.StoreId]StoreEntry
}

func (s *Store) PutFile(req *api.PutFileRequest, res *api.PutResponse) error {
	h := sha256.New()
	sum := [32]byte(h.Sum(req.Data))
	s.cas[sum] = StoreEntry{
		data: req.Data,
	}
	res.Id = sum
	log.Printf("put file with sum %x", sum)
	return nil
}

func (s *Store) PutDir(req *api.PutDirRequest, res *api.PutResponse) error {
	h := sha256.New()

	// Verify that all child ids are in the cas
	for _, child := range req.Children {
		_, exist := s.cas[child.Id]
		if !exist {
			return fmt.Errorf("refuse dir upload because of missing child %x", child.Id)
		}
	}

	// Ensure deterministic sort order
	slices.SortFunc(req.Children, func(a, b api.PutDirRequestChild) int {
		return cmp.Compare(a.Name, b.Name)
	})

	for _, child := range req.Children {
		h.Write([]byte(child.Name))
		h.Write(child.Id[:])
	}

	sum := [32]byte(h.Sum(nil))
	s.cas[sum] = StoreEntry{
		children: req.Children,
	}
	res.Id = sum

	log.Printf("put dir with sum %x", sum)
	for _, child := range req.Children {
		log.Printf("dir %x has child %s with sum %x", sum, child.Name, child.Id)
	}

	return nil
}

func (s *Store) GetFile(req *api.GetFileRequest, res *api.GetFileResponse) error {
	root, found := s.cas[req.Root]
	if !found {
		return fmt.Errorf("unknown root %x", req.Root)
	}

	path := req.Path
	log.Printf("searching for path %s at root %x", path, req.Root)
	for path != "" {
		idx := strings.Index(path, "/")
		var part string
		if idx == -1 {
			// Final component
			part = path
			path = ""
		} else {
			part = path[:idx]
			path = path[idx+1:]
		}

		childFound := false
		for _, child := range root.children {
			if child.Name == part {
				root = s.cas[child.Id]
				log.Printf("searching for path %s at root %x", path, child.Id)
				childFound = true
				break
			}
		}

		if !childFound {
			return fmt.Errorf("not found: %s of full path %s", part, req.Path)
		}
	}

	res.Data = root.data
	return nil
}

func main() {
	store := Store{
		cas: make(map[api.StoreId]StoreEntry),
	}
	rpc.Register(&store)
	rpc.HandleHTTP()
	l, e := net.Listen("tcp", "localhost:8000")
	if e != nil {
		log.Fatal("failed to start server: ", e)
	}

	http.Serve(l, nil)
}
