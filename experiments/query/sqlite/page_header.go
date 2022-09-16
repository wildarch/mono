package sqlite

import (
	"encoding/binary"
	"io"
)

const (
	PageTypeInteriorIndex = 2
	PageTypeInteriorTable = 5
	PagetypeLeafIndex     = 10
	PageTypeLeafTable     = 13
)

type pageHeader struct {
	PageType             uint8
	FirstFreeblock       uint16
	Cells                uint16
	CellContentAreaStart uint16 // Note: 0 interpreted as 65536
	FragmentedFreeBytes  uint8
	// Interior b-tree pages have an extra uint32 with a page number for the right-most pointer
}

func (h *pageHeader) Read(f io.Reader) error {
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

func (h *pageHeader) IsInterior() bool {
	return h.PageType == PageTypeInteriorIndex || h.PageType == PageTypeInteriorTable
}

func (h *pageHeader) Size() int {
	if h.IsInterior() {
		return 12
	} else {
		return 8
	}
}
