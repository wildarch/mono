mod generated;
pub use generated::*;

mod r#type;
pub use r#type::*;

mod def;
pub use def::*;

pub struct Value {
    offset: u32,
}

pub struct Block {
    offset: u32,
}

pub struct Function {
    offset: u32,
}

pub struct Def {
    offset: u32,
}

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
    pub result_type: Type,
    pub kind: OpKind,
}

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

pub struct Context {
    // TODO: ops (values)
    // TODO: blocks
    // TODO: defs
}
