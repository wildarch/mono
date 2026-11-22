use crate::ir::SourceFileRef;

#[derive(Clone, Copy, Debug)]
pub struct InFilePos {
    // nonzero/option tricks
    pub line: usize,
    pub column: usize,
}

impl InFilePos {
    pub fn start_of_file() -> InFilePos {
        InFilePos { line: 1, column: 1 }
    }

    pub fn unknown() -> InFilePos {
        InFilePos { line: 0, column: 0 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Loc {
    pub source_file: SourceFileRef,
    pub start: InFilePos,
    pub end: InFilePos,
}
