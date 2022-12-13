#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(i32)]
pub enum Comb {
    S,
    K,
    I,
    Y,
    U,
    P,
    B,
    C,
    Plus,
    Minus,
    Times,
    Divide,
    Cond,
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    And,
    Or,
    Not,
    Abort,
}

impl Comb {
    pub fn all() -> &'static [Comb] {
        return ALL_COMBS;
    }
}

const ALL_COMBS: &'static [Comb] = &[
    Comb::S,
    Comb::K,
    Comb::I,
    Comb::Y,
    Comb::U,
    Comb::P,
    Comb::B,
    Comb::C,
    Comb::Plus,
    Comb::Minus,
    Comb::Times,
    Comb::Divide,
    Comb::Cond,
    Comb::Eq,
    Comb::Neq,
    Comb::Gt,
    Comb::Gte,
    Comb::Lt,
    Comb::Lte,
    Comb::And,
    Comb::Or,
    Comb::Not,
    Comb::Abort,
];

#[derive(Clone, PartialEq)]
pub enum CompiledExpr {
    Comb(Comb),
    Ap(Box<CompiledExpr>, Box<CompiledExpr>),
    Var(String),
    Int(i32),
}

impl std::fmt::Debug for CompiledExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompiledExpr::Comb(c) => write!(f, "{:?}", c),
            CompiledExpr::Ap(a, b) => write!(f, "({:?} {:?})", a, b),
            CompiledExpr::Var(v) => write!(f, "{}", v),
            CompiledExpr::Int(i) => write!(f, "{}", i),
        }
    }
}

impl Into<CompiledExpr> for Comb {
    fn into(self) -> CompiledExpr {
        CompiledExpr::Comb(self)
    }
}

pub fn cap<A: Into<CompiledExpr>, B: Into<CompiledExpr>>(a: A, b: B) -> CompiledExpr {
    CompiledExpr::Ap(Box::new(a.into()), Box::new(b.into()))
}
