package main

import (
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"net/rpc"
	"os"
	"path"

	"github.com/wildarch/mono/experiments/dabu/store/api"
)

type Forester struct {
	client   *rpc.Client
	cacheDir string
}

func (f *Forester) cachePath(id api.StoreId) string {
	idStr := fmt.Sprintf("%x", id)
	return path.Join(f.cacheDir, idStr)
}

func (f *Forester) instantiate(id api.StoreId) error {
	req := &api.GetRequest{Id: id}
	var res api.GetResponse
	if err := f.client.Call("Store.Get", req, &res); err != nil {
		return err
	}

	// Check if already exists
	targetPath := f.cachePath(id)
	_, err := os.Stat(targetPath)
	if err == nil {
		// Already exists, no need to instantiate.
		return nil
	} else if errors.Is(err, fs.ErrNotExist) {
		// Not instantiate before, code below will create it.
	} else {
		// Unexpected error
		return err
	}

	if len(res.Children) > 0 {
		// Directory
		if err := os.Mkdir(targetPath, 0775); err != nil {
			return err
		}

		for _, child := range res.Children {
			subpath := path.Join(targetPath, child.Name)
			if err := f.instantiate(child.Id); err != nil {
				return err
			}

			if err := os.Symlink(f.cachePath(child.Id), subpath); err != nil {
				return err
			}
		}
	} else {
		// File
		if err := os.WriteFile(targetPath, res.Data, 0664); err != nil {
			return err
		}
	}

	return nil
}

func main() {
	rootHash := flag.String("root-hash", "", "root directory hash")
	cacheDir := flag.String("cache", "/tmp/forester-cache", "cache directory for object storage")
	flag.Parse()

	client, err := rpc.DialHTTP("tcp", "localhost:8000")
	if err != nil {
		log.Fatal("failed to connect to store:", err)
	}

	if *rootHash == "" {
		log.Fatalf("-root-hash flag is required")
	}

	decodedId, err := hex.DecodeString(*rootHash)
	if err != nil {
		log.Fatalf("invalid root hash: %s", err.Error())
	}

	rootId := api.StoreId(decodedId)

	if err := os.MkdirAll(*cacheDir, 0775); err != nil {
		log.Fatalf("cannot create cache directory at %s: %s", *cacheDir, err.Error())
	}

	f := Forester{client: client, cacheDir: *cacheDir}
	err = f.instantiate(rootId)
	if err != nil {
		log.Fatalf("failed to instantiate: %s", err.Error())
	}

	log.Println(f.cachePath(rootId))
}
