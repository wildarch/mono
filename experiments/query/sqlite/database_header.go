package sqlite

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
)

const databaseHeaderSize = 100

type databaseHeader struct {
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

func (h *databaseHeader) Read(f io.Reader) error {
	err := binary.Read(f, binary.BigEndian, h)
	if err != nil {
		return err
	}

	const expectedHeaderString = "SQLite format 3\000"
	if !bytes.Equal([]byte(expectedHeaderString), h.HeaderString[:]) {
		return fmt.Errorf("invalid header string '%s'", string(h.HeaderString[:]))
	}
	// TODO: Validate other fields
	return nil
}
