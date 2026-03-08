package main

import (
	"flag"
	"io"
	"log"
	"net/rpc"
	"os"
	"path/filepath"
	"slices"

	"github.com/wildarch/mono/experiments/dabu/store/api"
)

func uploadDir(client *rpc.Client, path string, f *os.File) (api.StoreId, error) {
	names, err := f.Readdirnames(0)
	if err != nil {
		return api.StoreId{}, err
	}

	// Deterministic order for the files inside the directory
	slices.Sort(names)

	req := &api.PutDirRequest{}
	for _, name := range names {
		subpath := filepath.Join(path, name)
		subid, err := upload(client, subpath)
		if err != nil {
			return api.StoreId{}, err
		}

		req.Children = append(req.Children, api.PutDirRequestChild{Name: name, Id: subid})
	}

	var res api.PutResponse
	if err = client.Call("Store.PutDir", req, &res); err != nil {
		return api.StoreId{}, err
	}

	return res.Id, nil
}

func uploadFile(client *rpc.Client, f *os.File) (api.StoreId, error) {
	data, err := io.ReadAll(f)
	if err != nil {
		return api.StoreId{}, nil
	}

	req := &api.PutFileRequest{Data: data}
	var res api.PutResponse
	if err = client.Call("Store.PutFile", req, &res); err != nil {
		return api.StoreId{}, err
	}

	return res.Id, nil
}

func upload(client *rpc.Client, path string) (api.StoreId, error) {
	f, err := os.Open(path)
	if err != nil {
		return api.StoreId{}, err
	}

	stat, err := f.Stat()
	if err != nil {
		return api.StoreId{}, err
	}

	if stat.IsDir() {
		return uploadDir(client, path, f)
	} else {
		return uploadFile(client, f)
	}
}

func main() {
	rootPath := flag.String("root", ".", "root directory for the merkle tree")
	flag.Parse()

	client, err := rpc.DialHTTP("tcp", "localhost:8000")
	if err != nil {
		log.Fatal("failed to connect to store:", err)
	}

	log.Printf("root dir: %s", *rootPath)
	h, err := upload(client, *rootPath)
	if err != nil {
		log.Fatalf("failed to upload directory: %s", err.Error())
	}

	log.Printf("root hash: %x", h)
}
