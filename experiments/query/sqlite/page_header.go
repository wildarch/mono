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

// pageHeader but without the RightMostPointer, which exists only on interior pages
type commonPageHeader struct {
	PageType             uint8
	FirstFreeblock       uint16
	Cells                uint16
	CellContentAreaStart uint16 // Note: 0 interpreted as 65536
	FragmentedFreeBytes  uint8
}

type pageHeader struct {
	Common           commonPageHeader
	RightMostPointer uint32
}

func (h *pageHeader) Read(f io.Reader) error {
	err := binary.Read(f, binary.BigEndian, &h.Common)
	if err != nil {
		return err
	}
	if h.IsInterior() {
		err = binary.Read(f, binary.BigEndian, &h.RightMostPointer)
	}
	return err
}

func (h *pageHeader) IsInterior() bool {
	return h.Common.PageType == PageTypeInteriorIndex || h.Common.PageType == PageTypeInteriorTable
}

func (h *pageHeader) Size() int {
	if h.IsInterior() {
		return 12
	} else {
		return 8
	}
}
