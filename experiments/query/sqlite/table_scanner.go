package sqlite

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
)

func (c *Conn) ScanTable(name string) (*TableScanner, error) {
	rootpage, err := c.findRootPage(name)
	if err != nil {
		return nil, fmt.Errorf("could not find root page: %w", err)
	}
	return &TableScanner{
		conn:     c,
		rootpage: rootpage,
	}, nil
}

func (c *Conn) findRootPage(name string) (uint32, error) {
	if name == "sqlite_schema" {
		// schema table always lives at the first page
		return 1, nil
	}

	// Scan the schema table to find the table we want
	schemaScan, err := c.ScanTable("sqlite_schema")
	if err != nil {
		return 0, fmt.Errorf("failed to open scanner for schema: %w", err)
	}

	for {
		row, err := schemaScan.ReadRow()
		if err != nil {
			return 0, fmt.Errorf("failed to read row from schema: %w", err)
		}
		if row == nil {
			return 0, fmt.Errorf("no such table '%s'", name)
		}
		ty := row.Values[0].(string)
		if ty != "table" {
			continue
		}
		tableName := row.Values[1].(string)
		if tableName != name {
			continue
		}

		rootPage := uint32(row.Values[3].(int64))
		return rootPage, nil
	}
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

type cachedInteriorPage struct {
	pageNum    uint32
	childPages []uint32
}

func (s *TableScanner) ReadRow() (*Row, error) {
	if s.pageBuf == nil {
		err := s.readPage(s.rootpage)
		if err != nil {
			return nil, err
		}
	}

	for {
		// If we are currently at an interior page, first go to the left-most leaf page
		hasLeaf, err := s.traverseToLeaf()
		if err != nil {
			return nil, err
		}
		if !hasLeaf {
			// Nothing left to scan
			return nil, nil
		}

		// We are now at a leaf node, check if it contains any data
		if (len(s.cellPointers)) == 0 {
			// Leaf is empty, get the next page
			hasNextPage, err := s.nextPage()
			if err != nil {
				return nil, err
			}
			if !hasNextPage {
				return nil, nil
			}
			// Restart from new page
			continue
		}

		// Remove the first of the cell pointers, we will return that row
		ptr := s.cellPointers[0]
		s.cellPointers = s.cellPointers[1:]

		return s.readLeafCell(ptr)
	}
}

func (s *TableScanner) readPage(page uint32) error {
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
	s.cellPointers = make([]uint16, s.pageHeader.Common.Cells)
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
		s.pageStack[len(s.pageStack)-1] = topInteriorPage

		return true, s.readPage(nextPage)
	}
	// Nothing left to move to
	return false, nil
}

func (s *TableScanner) traverseToLeaf() (bool, error) {
	for s.pageHeader.IsInterior() {
		if len(s.cellPointers) == 0 {
			newPage, err := s.nextPage()
			if err != nil {
				return false, err
			}
			if newPage {
				continue
			} else {
				// Nothing left to scan
				return false, nil
			}
		}
		// get the page numbers
		childPages := make([]uint32, len(s.cellPointers)+1)
		for i, ptr := range s.cellPointers {
			pageNum := binary.BigEndian.Uint32(s.pageBuf[ptr : ptr+4])
			// Followed by a varint integer key, but we don't need it

			childPages[i] = pageNum
		}
		// Add the right-most pointer
		childPages[len(childPages)-1] = s.pageHeader.RightMostPointer
		// Add this page to the stack, omitting the first child page
		s.pageStack = append(s.pageStack, cachedInteriorPage{
			pageNum:    s.pageNum,
			childPages: childPages[1:],
		})
		// Decend into the first child page
		err := s.readPage(childPages[0])
		if err != nil {
			return false, err
		}
	}

	return true, nil
}

func (s *TableScanner) readLeafCell(ptr uint16) (*Row, error) {
	// Leaf cell format:
	// - varint number of bytes payload, including any overflow
	// - varint integer key (rowid)
	// - initial portion of payload
	// - uint32 page number for first page of overflow page list. Omitted if all payload fits on this page

	// Payload size
	payloadSize, bytesRead := decodeVarint(s.pageBuf[ptr:])
	ptr += uint16(bytesRead)

	// Rowid
	rowid, bytesRead := decodeVarint(s.pageBuf[ptr:])
	ptr += uint16(bytesRead)

	// Payload
	payload := make([]byte, payloadSize)
	err := s.readPayload(ptr, payload)
	if err != nil {
		return nil, err
	}
	record, err := readRecord(int64(rowid), payload)
	if err != nil {
		return nil, err
	}
	return record, nil
}

func (s *TableScanner) readPayload(payloadStart uint16, payloadBuffer []byte) error {
	// Calculate how much of the payload spills onto overflow pages
	usableSize := int(s.conn.header.PageSize) - int(s.conn.header.ReservedBytes)
	maxStorablePayload := usableSize - 35
	bytesOnStartPage := 0
	if len(payloadBuffer) <= maxStorablePayload {
		bytesOnStartPage = len(payloadBuffer)
	} else {
		minimumStoredPayload := ((usableSize - 12) * 32 / 255) - 23
		// K = M + ( (P-M) % (U-4) )
		// 675 = 103 + ( (8835 - 103) % (1024 - 4) ) = 103 + (8732 % 1020) = 103 + 572
		k := minimumStoredPayload + ((len(payloadBuffer) - minimumStoredPayload) % (usableSize - 4))
		if k <= maxStorablePayload {
			bytesOnStartPage = k
		} else {
			bytesOnStartPage = minimumStoredPayload
		}
	}

	// Copy the bytes from the start page
	copy(payloadBuffer, s.pageBuf[payloadStart:payloadStart+uint16(bytesOnStartPage)])
	// Shrink payload buffer to the part that remains to be filled
	payloadBuffer = payloadBuffer[bytesOnStartPage:]

	if len(payloadBuffer) == 0 {
		return nil
	}

	// Now copy from the overflow pages
	nextPageNum := binary.BigEndian.Uint32(s.pageBuf[payloadStart+uint16(bytesOnStartPage):])
	page := make([]byte, s.conn.header.PageSize)
	for nextPageNum != 0 {
		err := s.conn.readPage(page, nextPageNum)
		if err != nil {
			return err
		}

		nextPageNum = binary.BigEndian.Uint32(page[:4])
		bytesCopied := copy(payloadBuffer, page[4:])
		payloadBuffer = payloadBuffer[bytesCopied:]
	}

	return nil
}

const (
	SerialTypeNull = iota
	SerialTypeInt8
	SerialTypeInt16
	SerialTypeInt24
	SerialTypeInt32
	SerialTypeInt48
	SerialTypeInt64
	SerialTypeDouble
	SerialTypeFalse
	SerialTypeTrue
	SerialTypeReserved10
	SerialTypeReserved11
)

func readRecord(rowid int64, payload []byte) (*Row, error) {
	values := make([]interface{}, 0)
	headerSize, bytesRead := decodeVarint(payload[:9])
	headerReader := bytes.NewReader(payload[bytesRead:headerSize])
	valuesReader := bytes.NewReader(payload[headerSize:])

	for {
		serialType, err := readVarint(headerReader)
		if err == io.EOF {
			// Header has been fully parsed
			break
		} else if err != nil {
			return nil, err
		}

		switch serialType {
		case SerialTypeNull:
			values = append(values, nil)
		case SerialTypeInt8:
			v := int8(0)
			err := binary.Read(valuesReader, binary.BigEndian, &v)
			if err != nil {
				return nil, err
			}
			values = append(values, int64(v))
		case SerialTypeInt16:
			v := int16(0)
			err := binary.Read(valuesReader, binary.BigEndian, &v)
			if err != nil {
				return nil, err
			}
			values = append(values, int64(v))
		case SerialTypeInt24:
			vs := make([]byte, 3)
			err := binary.Read(valuesReader, binary.BigEndian, &vs)
			if err != nil {
				return nil, err
			}
			v := int32(vs[0])<<16 | int32(vs[1])<<8 | int32(vs[2])
			values = append(values, v)
		case SerialTypeInt32:
			v := int32(0)
			err := binary.Read(valuesReader, binary.BigEndian, &v)
			if err != nil {
				return nil, err
			}
			values = append(values, int64(v))
		case SerialTypeInt48:
			vs := make([]byte, 5)
			err := binary.Read(valuesReader, binary.BigEndian, &vs)
			if err != nil {
				return nil, err
			}
			v := int64(vs[0])<<32 | int64(vs[1])<<24 | int64(vs[2])<<16 | int64(vs[3])<<8 | int64(vs[4])
			values = append(values, int64(v))
		case SerialTypeInt64:
			v := int64(0)
			err := binary.Read(valuesReader, binary.BigEndian, &v)
			if err != nil {
				return nil, err
			}
			values = append(values, v)
		case SerialTypeDouble:
			v := float64(0.0)
			err := binary.Read(valuesReader, binary.BigEndian, &v)
			if err != nil {
				return nil, err
			}
			values = append(values, v)
		case SerialTypeFalse:
			values = append(values, int64(0))
		case SerialTypeTrue:
			values = append(values, int64(1))
		case SerialTypeReserved10:
			fallthrough
		case SerialTypeReserved11:
			return nil, fmt.Errorf("found reserved serial type %d", serialType)
		default:
			// Even for blob, odd for string
			isBlob := serialType%2 == 0
			if isBlob {
				size := (serialType - 12) / 2
				blob := make([]byte, size)
				err := binary.Read(valuesReader, binary.BigEndian, &blob)
				if err != nil {
					return nil, err
				}
				values = append(values, blob)
			} else {
				// string
				size := (serialType - 13) / 2
				data := make([]byte, size)
				err := binary.Read(valuesReader, binary.BigEndian, &data)
				if err != nil {
					return nil, err
				}
				str := string(data)
				values = append(values, str)
			}
		}
	}

	return &Row{Rowid: rowid, Values: values}, nil
}

type Row struct {
	Rowid  int64
	Values []interface{}
}
