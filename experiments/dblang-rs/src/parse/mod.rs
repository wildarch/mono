pub mod chunker;
#[cfg(test)]
mod golden;
pub mod lexer;

/// Whether `c` is a UTF-8 continuation byte, i.e. not the first byte of a
/// character. Used to count columns in characters rather than bytes.
fn is_utf8_continuation(c: u8) -> bool {
    c & 0xC0 == 0x80
}
