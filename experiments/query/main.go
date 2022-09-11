package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"unsafe"
)

type Sqlite3DatabaseHeader struct {
	HeaderString [16]byte
	PageSize     uint16
}

func (h *Sqlite3DatabaseHeader) Read(f io.Reader) error {
	size := int(unsafe.Sizeof(*h))
	data := make([]byte, size)
	n, err := f.Read(data)
	if err != nil {
		return err
	}
	if n != size {
		// TODO: technically this is allowed, we should attempt to read more bytes
		return errors.New(fmt.Sprintf("expected to read %d bytes, got %d", size, n))
	}

	buffer := bytes.NewBuffer(data)
	err = binary.Read(buffer, binary.BigEndian, h)
	if err != nil {
		return errors.New(fmt.Sprintf("conversion to header failed (%s)", err.Error()))
	}

	const expectedHeaderString = "SQLite format 3\000"
	if !bytes.Equal([]byte(expectedHeaderString), h.HeaderString[:]) {
		return errors.New("invalid header string")
	}
	return nil
}

func main() {
	dbFilePath := flag.String("db", "", "path to sqlite database file")

	flag.Parse()

	if *dbFilePath == "" {
		log.Fatal("missing argument -db")
	}

	log.Printf("reading from %s", *dbFilePath)

	dbFile, err := os.Open(*dbFilePath)
	if err != nil {
		log.Fatalf("cannot open %s: %s", *dbFilePath, err.Error())
	}
	defer dbFile.Close()

	header := Sqlite3DatabaseHeader{}
	err = header.Read(dbFile)
	if err != nil {
		log.Fatalf("failed to decode header: %s", err.Error())
	}
	log.Printf("Header: %+v", header)
}
