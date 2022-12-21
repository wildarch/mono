#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    // Kiselyov's bulk combinators
    Sn(usize),
    Bn(usize),
    Cn(usize),
}

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
            CompiledExpr::Comb(c) => match c {
                Comb::Sn(i) => write!(f, "S{}", i),
                Comb::Bn(i) => write!(f, "B{}", i),
                Comb::Cn(i) => write!(f, "C{}", i),
                _ => write!(f, "{:?}", c),
            },
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
