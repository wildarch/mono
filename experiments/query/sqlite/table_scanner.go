package sqlite

import (
	"bytes"
	"encoding/binary"
	"io"
	"log"
)

func (c *Conn) ScanTable(name string) (*TableScanner, error) {
	if name == "sqlite_schema" {
		return &TableScanner{
			conn:     c,
			rootpage: 1,
		}, nil
	}
	panic("not implemented")
}

type TableScanner struct {
	conn     *Conn
	rootpage uint32
	// TODO: schema
	pageBuf      []byte
	pageNum      uint32
	pageHeader   pageHeader
	cellPointers []uint16 // cell pointers for the current page. Empty if all cells have been visited
	pageStack    []cachedInteriorPage
}

func (s *TableScanner) ReadRow() (*Row, error) {
	if s.pageBuf == nil {
		err := s.readPage(s.rootpage)
		if err != nil {
			return nil, err
		}
	}

	for s.pageHeader.IsInterior() {
		if len(s.cellPointers) == 0 {
			newPage, err := s.nextPage()
			if err != nil {
				return nil, err
			}
			if newPage {
				continue
			} else {
				// Nothing left to scan
				return nil, nil
			}
		}
		// get the page numbers
		childPages := make([]uint32, len(s.cellPointers))
		for i, ptr := range s.cellPointers {
			pageNum := binary.BigEndian.Uint32(s.pageBuf[ptr : ptr+4])
			// Followed by a varint integer key, but we don't need it

			childPages[i] = pageNum
		}
		// Add this page to the stack, omitting the first child page
		s.pageStack = append(s.pageStack, cachedInteriorPage{
			pageNum:    s.pageNum,
			childPages: childPages[1:],
		})
		// Decend into the first child page
		err := s.readPage(childPages[0])
		if err != nil {
			return nil, err
		}
	}
	// TODO: read leaf
	panic("not implemented")
}

func (s *TableScanner) readPage(page uint32) error {
	log.Printf("read page %d", page)
	if s.pageBuf == nil {
		s.pageBuf = make([]byte, s.conn.header.PageSize)
	}

	// Read page
	err := s.conn.readPage(s.pageBuf, page)
	if err != nil {
		return err
	}
	s.pageNum = page
	pageReader := bytes.NewReader(s.pageBuf)
	if page == 1 {
		// Special case for the first page: skip over the database header
		pageReader.Seek(databaseHeaderSize, io.SeekCurrent)
	}

	// header
	err = s.pageHeader.Read(pageReader)
	if err != nil {
		return err
	}

	// cell pointers
	s.cellPointers = make([]uint16, s.pageHeader.Cells)
	err = binary.Read(pageReader, binary.BigEndian, s.cellPointers)
	if err != nil {
		return err
	}
	return nil
}

func (s *TableScanner) nextPage() (bool, error) {
	for len(s.pageStack) > 0 {
		topInteriorPage := s.pageStack[len(s.pageStack)-1]

		if len(topInteriorPage.childPages) == 0 {
			// Remove from the stack
			s.pageStack = s.pageStack[:len(s.pageStack)-1]
			continue
		}

		nextPage := topInteriorPage.childPages[0]
		topInteriorPage.childPages = topInteriorPage.childPages[1:]
		// TODO: do we need this?
		s.pageStack[len(s.pageStack)-1] = topInteriorPage

		return true, s.readPage(nextPage)
	}
	// Nothing left to move to
	return false, nil
}

type Row struct {
	Rowid  int64
	Values []interface{}
}

type cachedInteriorPage struct {
	pageNum    uint32
	childPages []uint32
}
