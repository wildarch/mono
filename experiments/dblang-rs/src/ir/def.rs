use crate::ir::{BlockRef, RegionContext, Value};

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
    pub region: RegionContext,
    pub entry_block: BlockRef,
}

pub struct ConstDef {
    pub region: RegionContext,
    pub value: Value,
}

pub struct GlobalDef {
    pub typ: Type,
}

#[cfg(test)]
mod tests {
    use std::assert_matches;

    use crate::ir::{BasicType, Const, ConstOp, InFilePos, Loc, OpKind, Program};

    use super::*;

    #[test]
    fn const_def() {
        let mut program = Program::new();
        let source_file = program.add_source_file("test.db");
        let loc = Loc {
            source_file,
            start: InFilePos::unknown(),
            end: InFilePos::unknown(),
        };

        let mut region = RegionContext::new();
        let block = region.add_block();
        let value = {
            let mut builder = region.new_builder();
            builder.set_insertion_block(block);
            builder.build_const(loc, Type::basic(BasicType::I32), Const::I32(42))
        };

        let def_ref = program.add_def(Def {
            name: "ANSWER".to_string(),
            kind: DefKind::Const(ConstDef { region, value }),
        });

        assert_eq!(def_ref.offset, 0);
        assert_eq!(program.defs.len(), 1);

        let def = &program.defs[0];
        assert_eq!(def.name, "ANSWER");
        let DefKind::Const(const_def) = &def.kind else {
            panic!("expected a const def");
        };

        assert_eq!(const_def.region.blocks.len(), 1);
        assert_eq!(const_def.region.blocks[0].ops.len(), 1);
        assert_eq!(const_def.region.ops.len(), 1);
        assert_eq!(const_def.value.offset, 0);

        let op = &const_def.region.ops[const_def.value.offset as usize];
        assert_matches!(op.result_type.basic_type, BasicType::I32);
        assert_matches!(
            op.kind,
            OpKind::Const(ConstOp {
                value: Const::I32(42)
            })
        );
    }
}
