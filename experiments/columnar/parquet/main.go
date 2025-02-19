package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"

	"github.com/apache/arrow/go/parquet"
	"github.com/apache/arrow/go/parquet/file"
	"github.com/apache/arrow/go/parquet/schema"
)

func writeColumnI32(reader *file.Int32ColumnChunkReader, out io.Writer) error {
	buf := make([]int32, 1024)
	for reader.HasNext() {
		_, read, err := reader.ReadBatch(1024, buf, nil, nil)
		if err != nil {
			return fmt.Errorf("failed to read batch: %w", err)
		}

		binary.Write(out, binary.LittleEndian, buf[:read])
	}
	return nil
}

func writeColumnI64(reader *file.Int64ColumnChunkReader, out io.Writer) error {
	buf := make([]int64, 1024)
	for reader.HasNext() {
		_, read, err := reader.ReadBatch(1024, buf, nil, nil)
		if err != nil {
			return fmt.Errorf("failed to read batch: %w", err)
		}

		binary.Write(out, binary.LittleEndian, buf[:read])
	}
	return nil
}

func writeColumnByteArray(reader *file.ByteArrayColumnChunkReader, out io.Writer) error {
	// TODO: Write string index
	buf := make([]parquet.ByteArray, 1024)
	for reader.HasNext() {
		_, read, err := reader.ReadBatch(1024, buf, nil, nil)
		if err != nil {
			return fmt.Errorf("failed to read batch: %w", err)
		}

		for _, str := range buf[:read] {
			binary.Write(out, binary.LittleEndian, str)
		}
	}
	return nil
}

func writeColumn(reader *file.Reader, c int, out io.Writer) error {
	var err error
	for g := 0; g < reader.NumRowGroups(); g++ {
		group := reader.RowGroup(g)
		col := group.Column(c)
		switch col.Type() {
		case parquet.Types.Int32:
			err = writeColumnI32(col.(*file.Int32ColumnChunkReader), out)
		case parquet.Types.Int64:
			err = writeColumnI64(col.(*file.Int64ColumnChunkReader), out)
		case parquet.Types.ByteArray:
			err = writeColumnByteArray(col.(*file.ByteArrayColumnChunkReader), out)
		default:
			return fmt.Errorf("unsupported type: %s", col.Type().String())
		}

		if err != nil {
			break
		}
	}

	return err
}

func readFile(path string) error {
	reader, err := file.OpenParquetFile(path, true)
	if err != nil {
		return fmt.Errorf("failed to open parquet file: %w", err)
	}

	sc := reader.MetaData().Schema
	schema.PrintSchema(sc.Root(), os.Stdout, 2)
	log.Printf("Found %d rows", reader.NumRows())

	for i := 0; i < sc.NumColumns(); i++ {
		col := sc.Column(i)
		colPath := fmt.Sprintf("%s/%s.col", filepath.Dir(path), col.Path())
		colFile, err := os.Create(colPath)
		if err != nil {
			return fmt.Errorf("failed to create column file at %s: %w", colPath, err)
		}

		defer colFile.Close()

		err = writeColumn(reader, i, colFile)
		if err != nil {
			return fmt.Errorf("failed to convert column %s: %w", col.Path(), err)
		}
	}

	return nil
}

func main() {
	for _, path := range os.Args[1:] {
		err := readFile(path)
		if err != nil {
			log.Fatalf("failed to convert file at %s: %v", path, err)
		}
	}
}
