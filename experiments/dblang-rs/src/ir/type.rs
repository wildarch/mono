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
