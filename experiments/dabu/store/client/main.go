package main

import (
	"encoding/hex"
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

		req.Children = append(req.Children, api.DirListing{Name: name, Id: subid})
	}

	var res api.PutResponse
	if err = client.Call("Store.PutDir", req, &res); err != nil {
		return api.StoreId{}, err
	}

	return res.Id, nil
}

func uploadFile(client *rpc.Client, path string, f *os.File) (api.StoreId, error) {
	data, err := io.ReadAll(f)
	if err != nil {
		return api.StoreId{}, nil
	}

	req := &api.PutFileRequest{Data: data, PathForDebug: path}
	var res api.PutResponse
	if err = client.Call("Store.PutFile", req, &res); err != nil {
		return api.StoreId{}, err
	}

	if res.IsNew {
		log.Printf("new file %s: %x", path, res.Id)
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
		return uploadFile(client, path, f)
	}
}

func download(client *rpc.Client, root api.StoreId, path string) ([]byte, error) {
	req := &api.GetFileRequest{Root: root, Path: path}
	var res api.GetFileResponse
	if err := client.Call("Store.GetFile", req, &res); err != nil {
		return nil, err
	}

	return res.Data, nil
}

func main() {
	uploadRoot := flag.String("upload-root", "", "root directory to be uploaded")
	downloadRootHash := flag.String("download-root-hash", "", "root directory hash")
	downloadPath := flag.String("download-file-path", "", "path to the file to be downloaded")
	flag.Parse()

	client, err := rpc.DialHTTP("tcp", "localhost:8000")
	if err != nil {
		log.Fatal("failed to connect to store:", err)
	}

	if *uploadRoot != "" {
		log.Printf("root dir: %s", *uploadRoot)
		h, err := upload(client, *uploadRoot)
		if err != nil {
			log.Fatalf("failed to upload directory: %s", err.Error())
		}

		log.Printf("root hash: %x", h)
	}

	if *downloadRootHash != "" {
		rootHash, err := hex.DecodeString(*downloadRootHash)
		if err != nil {
			log.Fatalf("invalid root hash: %s", err.Error())
		}

		log.Printf("download %s from root %x", *downloadPath, rootHash)
		data, err := download(client, api.StoreId(rootHash), *downloadPath)
		if err != nil {
			log.Fatalf("failed to download file: %s", err.Error())
		}
		os.Stdout.Write(data)
	}
}
