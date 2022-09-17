package sqlite

import "io"

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
	for i := range buf {
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
