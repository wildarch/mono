package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"flag"
	"io"
	"log"
	"os"
)

const DatabaseHeaderSize = 100

type DatabaseHeader struct {
	HeaderString               [16]byte
	PageSize                   uint16
	WriteVersion               uint8
	ReadVersion                uint8
	ReservedBytes              uint8
	MaxEmbeddedPayloadFraction uint8
	MinEmbeddedPayloadFraction uint8
	LeafPayloadFraction        uint8
	FileChangeCounter          uint32
	DatabasePages              uint32
	FirstFreelistTrunkPage     uint32
	FreelistPages              uint32
	SchemaCookie               uint32
	SchemaFormat               uint32
	DefaultPageCacheSize       uint32
	LargestRootBtreePage       uint32
	TextEncoding               uint32
	UserVersion                uint32
	IncrementalVacuumMode      uint32
	ApplicationId              uint32
	_                          [20]byte
	VersionValidFor            uint32
	VersionNumber              uint32
}

func (h *DatabaseHeader) Read(f io.Reader) error {
	err := binary.Read(f, binary.BigEndian, h)
	if err != nil {
		return err
	}

	const expectedHeaderString = "SQLite format 3\000"
	if !bytes.Equal([]byte(expectedHeaderString), h.HeaderString[:]) {
		return errors.New("invalid header string")
	}
	// TODO: Validate other fields
	return nil
}

const (
	PageTypeInteriorIndex = 2
	PageTypeInteriorTable = 5
	PagetypeLeafIndex     = 10
	PageTypeLeafTable     = 13
)

type PageHeader struct {
	PageType             uint8
	FirstFreeblock       uint16
	Cells                uint16
	CellContentAreaStart uint16 // Note: 0 interpreted as 65536
	FragmentedFreeBytes  uint8
	// Interior b-tree pages have an extra uint32 with a page number for the right-most pointer
}

func (h *PageHeader) Read(f io.Reader) error {
	// TODO: Validate fields
	err := binary.Read(f, binary.BigEndian, h)
	if err != nil {
		return err
	}
	rightMostPointer := uint32(0)
	if h.IsInterior() {
		err = binary.Read(f, binary.BigEndian, &rightMostPointer)
	}
	return err
}

func (h *PageHeader) IsInterior() bool {
	return h.PageType == PageTypeInteriorIndex || h.PageType == PageTypeInteriorTable
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

	dbHeader := DatabaseHeader{}
	err = dbHeader.Read(dbFile)
	if err != nil {
		log.Fatalf("failed to decode header: %s", err.Error())
	}
	log.Printf("Database Header: %+v", dbHeader)

	err = scanTable(&dbHeader, dbFile, 1)
	if err != nil {
		log.Fatalf("failed to scan table: %s", err.Error())
	}
}

func readPage(b []byte, f io.ReadSeeker, pageNum uint32) error {
	offset := int(pageNum-1) * len(b)
	_, err := f.Seek(int64(offset), io.SeekStart)
	if err != nil {
		return err
	}
	_, err = io.ReadFull(f, b)
	if err != nil {
		return err
	}
	return nil
}

func scanTable(dbHeader *DatabaseHeader, f io.ReadSeeker, rootPageNum uint32) error {
	pageBuf := make([]byte, dbHeader.PageSize)
	err := readPage(pageBuf, f, rootPageNum)
	if err != nil {
		return err
	}
	pageReader := bytes.NewReader(pageBuf)

	if rootPageNum == 1 {
		// Page 1 also contains the database header, skip it
		_, err = pageReader.Seek(DatabaseHeaderSize, io.SeekCurrent)
		if err != nil {
			return err
		}
	}

	// Header
	pageHeader := PageHeader{}
	err = pageHeader.Read(pageReader)
	if err != nil {
		return err
	}
	log.Printf("page header: %+v", pageHeader)

	// Cell pointers
	cellPointers := make([]uint16, pageHeader.Cells)
	err = binary.Read(pageReader, binary.BigEndian, cellPointers)
	if err != nil {
		return err
	}

	if pageHeader.PageType == PageTypeInteriorTable {
		for _, ptr := range cellPointers {
			// Format for interior table cells:
			// - 4-byte page number
			// - varint integer key
			leftPageNum := binary.BigEndian.Uint32(pageBuf[ptr : ptr+4])
			err = scanTableChild(dbHeader, f, leftPageNum)
			if err != nil {
				return err
			}
		}
	} else if pageHeader.PageType == PageTypeLeafTable {
		return scanTableChildLeaf(pageBuf, f, cellPointers)
	}

	return nil
}

func scanTableChild(dbHeader *DatabaseHeader, f io.ReadSeeker, pageNum uint32) error {
	pageBuf := make([]byte, dbHeader.PageSize)
	err := readPage(pageBuf, f, pageNum)
	if err != nil {
		return err
	}
	pageReader := bytes.NewReader(pageBuf)

	// Header
	pageHeader := PageHeader{}
	err = pageHeader.Read(pageReader)
	if err != nil {
		return err
	}

	// Cell pointers
	cellPointers := make([]uint16, pageHeader.Cells)
	err = binary.Read(pageReader, binary.BigEndian, cellPointers)
	if err != nil {
		return err
	}

	// Check leaf or interior
	switch pageHeader.PageType {
	case PageTypeInteriorTable:
		return scanTableChildInterior(pageBuf, f, cellPointers)
	case PageTypeLeafTable:
		return scanTableChildLeaf(pageBuf, f, cellPointers)
	default:
		panic("expected table page type")
	}
}

func scanTableChildInterior(pageBuf []byte, f io.ReadSeeker, cellPointers []uint16) error {
	for _, ptr := range cellPointers {
		panic(ptr)
	}
	return nil
}

func scanTableChildLeaf(pageBuf []byte, f io.ReadSeeker, cellPointers []uint16) error {
	for _, ptr := range cellPointers {
		// Leaf cell format:
		// - varint number of bytes payload, including any overflow
		// - varint integer key (rowid)
		// - initial portion of payload
		// - uint32 page number for first page of overflow page list. Omitted if all payload fits on this page

		// Payload size
		payloadSize, bytesRead := decodeVarint(pageBuf[ptr:])
		ptr += uint16(bytesRead)

		// Rowid
		rowid, bytesRead := decodeVarint(pageBuf[ptr:])
		log.Printf("rowid: %d", rowid)
		ptr += uint16(bytesRead)

		// Payload
		payload := make([]byte, payloadSize)
		err := readPayload(pageBuf, ptr, f, payload)
		if err != nil {
			return err
		}
		err = scanRecord(payload)
		if err != nil {
			return err
		}
	}
	return nil
}

func readPayload(startPage []byte, payloadStart uint16, f io.ReadSeeker, payloadBuffer []byte) error {
	// Calculate how much of the payload spills onto overflow pages
	usableSize := len(startPage) // TODO: account for reserved space
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
	copy(payloadBuffer, startPage[payloadStart:payloadStart+uint16(bytesOnStartPage)])
	// Shrink payload buffer to the part that remains to be filled
	payloadBuffer = payloadBuffer[bytesOnStartPage:]

	if len(payloadBuffer) == 0 {
		return nil
	}

	// Now copy from the overflow pages
	nextPageNum := binary.BigEndian.Uint32(startPage[payloadStart+uint16(bytesOnStartPage):])
	page := make([]byte, len(startPage))
	for nextPageNum != 0 {
		err := readPage(page, f, nextPageNum)
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

func scanRecord(payload []byte) error {
	headerSize, bytesRead := decodeVarint(payload[:9])
	headerReader := bytes.NewReader(payload[bytesRead:headerSize])
	valuesReader := bytes.NewReader(payload[headerSize:])

	for {
		serialType, err := readVarint(headerReader)
		if err == io.EOF {
			// Header has been fully parsed
			break
		} else if err != nil {
			return err
		}

		switch serialType {
		case SerialTypeNull:
			log.Printf("NULL")
		case SerialTypeInt8:
			v := int8(0)
			err := binary.Read(valuesReader, binary.BigEndian, &v)
			if err != nil {
				return err
			}
			log.Printf("int8: %d", v)
		case SerialTypeInt16:
			v := int16(0)
			err := binary.Read(valuesReader, binary.BigEndian, &v)
			if err != nil {
				return err
			}
			log.Printf("int16: %d", v)
		case SerialTypeInt24:
			log.Printf("int24 not supported")
		case SerialTypeInt32:
			v := int32(0)
			err := binary.Read(valuesReader, binary.BigEndian, &v)
			if err != nil {
				return err
			}
			log.Printf("int32: %d", v)
		case SerialTypeInt48:
			log.Printf("int48 not supported")
		case SerialTypeInt64:
			v := int64(0)
			err := binary.Read(valuesReader, binary.BigEndian, &v)
			if err != nil {
				return err
			}
			log.Printf("int64: %d", v)
		case SerialTypeDouble:
			v := float64(0.0)
			err := binary.Read(valuesReader, binary.BigEndian, &v)
			if err != nil {
				return err
			}
			log.Printf("double: %f", v)
		case SerialTypeFalse:
			log.Printf("int: 0")
		case SerialTypeTrue:
			log.Printf("int: 1")
		case SerialTypeReserved10:
			fallthrough
		case SerialTypeReserved11:
			return errors.New("found reserved serial type")
		default:
			// Even for blob, odd for string
			isBlob := serialType%2 == 0
			if isBlob {
				size := (serialType - 12) / 2
				blob := make([]byte, size)
				err := binary.Read(valuesReader, binary.BigEndian, &blob)
				if err != nil {
					return err
				}
				log.Printf("blob: %v", blob)
			} else {
				// string
				size := (serialType - 13) / 2
				log.Printf("Size: %d", size)
				data := make([]byte, size)
				err := binary.Read(valuesReader, binary.BigEndian, &data)
				if err != nil {
					return err
				}
				str := string(data)
				log.Printf("string: '%s'", str)
			}
		}

	}

	return nil
}

/*
** The variable-length integer encoding is as follows:
**
** KEY:
**         A = 0xxxxxxx    7 bits of data and one flag bit
**         B = 1xxxxxxx    7 bits of data and one flag bit
**         C = xxxxxxxx    8 bits of data
**
**  7 bits - A
** 14 bits - BA
** 21 bits - BBA
** 28 bits - BBBA
** 35 bits - BBBBA
** 42 bits - BBBBBA
** 49 bits - BBBBBBA
** 56 bits - BBBBBBBA
** 64 bits - BBBBBBBBC
 */
func decodeVarint(buf []byte) (uint64, int) {
	const highBit = byte(1 << 7)

	v := uint64(0)
	sz := 0
	for _, b := range buf {
		sz += 1
		if sz == 9 {
			// There are at most 9 bytes per varint.
			// For this last byte, we are adding all the bits (including high bit).
			v = (v << 8) | uint64(b)
			break
		}

		// Take the 7 highest bits, the final bit is not part of the number
		bv := b & ^highBit
		// Shift previous bits up to make room for the added 7 bits
		v = (v << 7) | uint64(bv)

		// If the high bit was not set, this was the last byte to read
		if (b & highBit) == 0 {
			break
		}
	}

	return v, sz
}

func readVarint(r io.ByteReader) (uint64, error) {
	buf := make([]byte, 9)
	for i, _ := range buf {
		b, err := r.ReadByte()
		if err != nil {
			return 0, err
		}
		buf[i] = b

		if (b & (1 << 7)) == 0 {
			// No high bit, this was the last bit
			break
		}
	}

	v, _ := decodeVarint(buf)
	return v, nil
}
