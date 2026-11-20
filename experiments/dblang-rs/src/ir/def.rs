pub use super::Type;

pub struct Def {
    // TODO: loc
    pub name: String, // TODO: interned string
    pub kind: DefKind,
}
pub enum DefKind {
    Struct(StructDef),
    Enum(EnumDef),
    Alias(AliasDef),
    Func(FuncDef),
    Const(ConstDef),
    Global(GlobalDef),
}

pub struct StructField {
    pub name: String, // intern
    pub field_type: Type,
}

pub struct StructDef {
    pub fields: Vec<Type>,
}

pub struct EnumDef {
    pub alts: Vec<String>,
}

pub struct AliasDef {
    pub aliased: Type,
}

pub struct FuncDef {
    pub param_types: Vec<Type>,
    pub return_type: Type,
}

pub struct ConstDef {
    // TODO: expression body
}

pub struct GlobalDef {
    pub typ: Type,
}
