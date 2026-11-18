#[repr(u32)]
pub enum BasicType {
    Unresolved,
    Void,
    Bool,
    Char,
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    F32,
    F64,
    // Compound/nested types
    Struct,   // TODO: type info
    Enum,     // TODO: type info
    Array,    // Fixed-size array, TODO: type info
    Function, // Function reference, TODO: type info
}

pub struct Type {
    pub basic_type: BasicType,
    pub pointer_depth: u32,
}

pub struct Value {
    offset: u32,
}

pub struct Block {
    offset: u32,
}

pub struct Function {
    offset: u32,
}

pub struct Operation {
    // TODO: loc
    pub result_type: Type,
    pub kind: OpKind,
}

pub enum OpKind {
    If(IfOp),
    Const(ConstOp),
    Call(CallOp),
}

pub struct IfOp {
    pub cond: Value,
    pub if_branch: Block,
    pub else_branch: Block,
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

pub struct ConstOp {
    pub value: Const,
}

pub struct CallOp {
    pub func: Function,
    pub args: Vec<Value>,
}
