#[repr(u32)]
#[derive(Clone, Copy, Debug)]
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

#[derive(Clone, Copy, Debug)]
pub struct Type {
    pub basic_type: BasicType,
    pub pointer_depth: u32,
}

impl Type {
    pub fn basic(basic_type: BasicType) -> Type {
        Type {
            basic_type,
            pointer_depth: 0,
        }
    }

    pub fn void() -> Type {
        Type::basic(BasicType::Void)
    }
}
