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
	log.Printf("read page %d", pageNum)
	offset := int(pageNum-1) * len(b)
	_, err := f.Seek(int64(offset), 0)
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
	if pageHeader.PageType != PageTypeInteriorTable {
		panic("expected interior table page")
	}

	// Cell pointers
	cellPointers := make([]uint16, pageHeader.Cells)
	err = binary.Read(pageReader, binary.BigEndian, cellPointers)
	if err != nil {
		return err
	}

	for _, ptr := range cellPointers {
		// Format for interior table cells:
		// - 4-byte page number
		// - varint integer key
		leftPageNum := binary.BigEndian.Uint32(pageBuf[ptr : ptr+4])
		log.Printf("Page num: %d", leftPageNum)
		err = scanTableChild(dbHeader, f, leftPageNum)
		if err != nil {
			return err
		}
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
	log.Printf("page header: %+v", pageHeader)

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
		payloadSize, bytesRead := binary.Uvarint(pageBuf[ptr : ptr+binary.MaxVarintLen64])
		log.Printf("payload size: %d", payloadSize)
		ptr += uint16(bytesRead)

		// Rowid
		rowid, bytesRead := binary.Uvarint(pageBuf[ptr : ptr+binary.MaxVarintLen64])
		log.Printf("rowid: %d", rowid)
		ptr += uint16(bytesRead)

		// Payload
		payload := make([]byte, payloadSize)
		err := readPayload(pageBuf, ptr, f, payload)
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
	log.Printf("X: %d", maxStorablePayload)
	bytesOnStartPage := 0
	if len(payloadBuffer) <= maxStorablePayload {
		bytesOnStartPage = len(payloadBuffer)
	} else {
		minimumStoredPayload := ((usableSize - 12) * 32 / 255) - 23
		log.Printf("M: %d", minimumStoredPayload)
		// K = M + ( (P-M) % (U-4) )
		// 675 = 103 + ( (8835 - 103) % (1024 - 4) ) = 103 + (8732 % 1020) = 103 + 572
		k := minimumStoredPayload + ((len(payloadBuffer) - minimumStoredPayload) % (usableSize - 4))
		if k <= maxStorablePayload {
			log.Printf("K")
			bytesOnStartPage = k
		} else {
			log.Printf("M")
			bytesOnStartPage = minimumStoredPayload
		}
	}

	log.Printf("payloadStart[%d] bytesOnStartPage[%d]", payloadStart, bytesOnStartPage)

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
