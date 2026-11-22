mod def;
mod generated;
mod loc;
mod r#type;

pub use def::*;
pub use generated::*;
pub use loc::*;
pub use r#type::*;

#[derive(Clone, Copy, Debug)]
pub struct Value {
    offset: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct BlockRef {
    offset: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct DefRef {
    offset: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct LocRef {
    offset: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum Comparator {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

pub struct Operation {
    // TODO: loc
    pub loc: LocRef,
    pub result_type: Type,
    pub kind: OpKind,
}

pub struct Block {
    ops: Vec<Value>,
}

#[derive(Clone, Copy, Debug)]
pub enum Const {
    Bool(bool),
    Char(char),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    F32(f32),
    F64(f64),
}

pub struct SourceFile {
    filename: String,
}

/// Context for a full program
pub struct Program {
    source_files: Vec<SourceFile>,
    defs: Vec<Def>,
}

#[derive(Clone, Copy, Debug)]
pub struct SourceFileRef {
    offset: u32,
}

impl Program {
    pub fn new() -> Program {
        Program {
            source_files: Vec::new(),
            defs: Vec::new(),
        }
    }

    pub fn add_source_file(&mut self, filename: &str) -> SourceFileRef {
        let offset = self.source_files.len() as u32;
        self.source_files.push(SourceFile {
            filename: filename.to_string(),
        });
        SourceFileRef { offset }
    }

    pub fn add_def(&mut self, def: Def) -> DefRef {
        let offset = self.defs.len() as u32;
        self.defs.push(def);
        DefRef { offset }
    }
}

/// Context for an executable code region with one or more blocks of operations
pub struct RegionContext {
    locs: Vec<Loc>,
    ops: Vec<Operation>,
    blocks: Vec<Block>,
}

impl RegionContext {
    pub fn new() -> RegionContext {
        RegionContext {
            locs: Vec::new(),
            ops: Vec::new(),
            blocks: Vec::new(),
        }
    }

    pub fn new_builder(&mut self) -> OpBuilder<'_> {
        OpBuilder {
            region: self,
            insertion_block: None,
        }
    }

    pub fn add_block(&mut self) -> BlockRef {
        let offset = self.blocks.len() as u32;
        self.blocks.push(Block { ops: Vec::new() });
        BlockRef { offset }
    }
}

pub struct OpBuilder<'a> {
    region: &'a mut RegionContext,
    insertion_block: Option<BlockRef>,
}

impl<'a> OpBuilder<'a> {
    pub fn set_insertion_block(&mut self, block: BlockRef) {
        self.insertion_block = Some(block);
    }

    fn add_loc(&mut self, loc: Loc) -> LocRef {
        let offset = self.region.locs.len() as u32;
        self.region.locs.push(loc);
        LocRef { offset }
    }

    fn insert_op(&mut self, loc: Loc, result_type: Type, kind: OpKind) -> Value {
        let insert_block = self.insertion_block.expect("no insertion block");
        // insert the op
        let loc = self.add_loc(loc);
        let offset = self.region.ops.len() as u32;
        self.region.ops.push(Operation {
            loc,
            result_type,
            kind,
        });
        // place into the block
        let block = &mut self.region.blocks[insert_block.offset as usize];
        block.ops.push(Value { offset });
        Value { offset }
    }
}
