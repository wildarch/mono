package sqlite

import (
	"fmt"
	"io"
	"os"
)

type Conn struct {
	file   io.ReadSeekCloser
	header databaseHeader
}

func (c *Conn) Close() error {
	return c.file.Close()
}

func Open(path string) (*Conn, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open database file at '%s': %w", path, err)
	}

	header := databaseHeader{}
	err = header.Read(file)
	if err != nil {
		return nil, fmt.Errorf("invalid header: %w", err)
	}

	return &Conn{
		file:   file,
		header: header,
	}, nil
}

func (c *Conn) readPage(b []byte, pageNum uint32) error {
	offset := int(pageNum-1) * len(b)
	_, err := c.file.Seek(int64(offset), io.SeekStart)
	if err != nil {
		return fmt.Errorf("failed to seek to page %d: %w", pageNum, err)
	}
	_, err = io.ReadFull(c.file, b)
	if err != nil {
		return fmt.Errorf("failed to read page %d: %w", pageNum, err)
	}
	return nil
}
