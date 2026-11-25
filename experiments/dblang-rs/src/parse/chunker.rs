use crate::ir::{InFilePos, Loc, SourceFileRef};
use crate::parse::is_utf8_continuation;

#[derive(Clone, Copy, Debug)]
pub struct Chunk<'a> {
    pub loc: Loc,
    pub text: &'a str,
}

/// Splits a file into individual top-level definitions, appending them to
/// `chunks`.
///
/// The first chunk (always present) contains the file header, that is
/// everything before the first definition.
///
/// Every definition starts with the keyword "def", which must be at the start
/// of a line.
pub fn chunk<'a>(source: &'a str, source_file: SourceFileRef, chunks: &mut Vec<Chunk<'a>>) {
    let bytes = source.as_bytes();
    let mut add_chunk =
        |start_offset: usize, end_offset: usize, start: InFilePos, end: InFilePos| {
            chunks.push(Chunk {
                loc: Loc {
                    source_file,
                    start,
                    end,
                },
                text: &source[start_offset..end_offset],
            });
        };

    // Offset and start of the current chunk.
    let mut chunk_start_offset = 0;
    let mut chunk_start_pos = InFilePos::start_of_file();

    // Where we are currently scanning in the file.
    let mut offset = 0;
    let mut pos = InFilePos::start_of_file();

    if source.starts_with("def ") {
        // No header present, but we should still produce a header chunk.
        add_chunk(0, 0, pos, pos);
    }

    while offset < bytes.len() {
        let cur = bytes[offset];
        if cur != b'\n' {
            offset += 1;
            // Columns count characters: skip UTF-8 continuation bytes.
            if !is_utf8_continuation(cur) {
                pos.column += 1;
            }
            continue;
        }

        // Newline
        offset += 1;
        pos.line += 1;
        pos.column = 1;

        // Does this line begin a new definition?
        if bytes[offset..].starts_with(b"def ") {
            // End the previous chunk (if any).
            if chunk_start_offset < offset {
                add_chunk(chunk_start_offset, offset, chunk_start_pos, pos);
            }

            // Record the start of the next definition.
            chunk_start_offset = offset;
            chunk_start_pos = pos;
        }
    }

    if chunk_start_offset < offset {
        // End the final chunk.
        add_chunk(chunk_start_offset, offset, chunk_start_pos, pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Program;
    use crate::parse::golden::{fmt_loc, run_golden};

    const TESTS_PATH: &str = "experiments/dblang/parse/Chunker.test";

    fn run(input: &str) -> String {
        let mut program = Program::new();
        let file = program.add_source_file(TESTS_PATH);
        let mut chunks = Vec::new();
        chunk(input, file, &mut chunks);

        let mut out = String::new();
        for chunk in &chunks {
            out += &format!("{}:\n{}\n", fmt_loc(TESTS_PATH, &chunk.loc), chunk.text);
        }
        out
    }

    /// Runs the golden tests in `chunker.test`.
    #[test]
    fn golden() {
        run_golden("chunker", include_str!("chunker.test"), run);
    }

    /// Columns count characters, not bytes. Golden test inputs always end in
    /// a newline, so the final column is only observable here.
    #[test]
    fn columns_count_chars() {
        let mut program = Program::new();
        let file = program.add_source_file(TESTS_PATH);
        let mut chunks = Vec::new();
        chunk("def A = \"é€😀\"", file, &mut chunks);

        let end = chunks.last().unwrap().loc.end;
        assert_eq!((end.line, end.column), (1, 14));
    }
}
