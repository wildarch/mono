package main

import (
	"encoding/hex"
	"flag"
	"log"
	"net/rpc"
	"os"
	"path"

	"github.com/wildarch/mono/experiments/dabu/store/api"
)

func instantiate(client *rpc.Client, id api.StoreId, targetPath string) error {
	req := &api.GetRequest{Id: id}
	var res api.GetResponse
	if err := client.Call("Store.Get", req, &res); err != nil {
		return err
	}

	if len(res.Children) > 0 {
		// Directory
		if err := os.Mkdir(targetPath, 0775); err != nil {
			return err
		}

		for _, child := range res.Children {
			subpath := path.Join(targetPath, child.Name)
			if err := instantiate(client, child.Id, subpath); err != nil {
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
	targetRoot := flag.String("target", "", "target directory at which to instantiate the root")
	flag.Parse()

	client, err := rpc.DialHTTP("tcp", "localhost:8000")
	if err != nil {
		log.Fatal("failed to connect to store:", err)
	}

	if *rootHash == "" {
		log.Fatalf("-root-hash flag is required")
	}
	if *targetRoot == "" {
		log.Fatalf("-target flag is required")
	}
	rootId, err := hex.DecodeString(*rootHash)
	if err != nil {
		log.Fatalf("invalid root hash: %s", err.Error())
	}

	err = instantiate(client, api.StoreId(rootId), *targetRoot)
	if err != nil {
		log.Fatalf("failed to instantiate: %s", err.Error())
	}
}
