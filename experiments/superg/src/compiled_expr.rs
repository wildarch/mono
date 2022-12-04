use crate::ast;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(i32)]
pub enum Comb {
    S,
    K,
    I,
    Y,
    U,
    P,
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

#[derive(Debug, Clone)]
pub enum CompiledExpr {
    Comb(Comb),
    Ap(Box<CompiledExpr>, Box<CompiledExpr>),
    Var(String),
    Int(i32),
}

impl Into<CompiledExpr> for Comb {
    fn into(self) -> CompiledExpr {
        CompiledExpr::Comb(self)
    }
}

fn cap<A: Into<CompiledExpr>, B: Into<CompiledExpr>>(a: A, b: B) -> CompiledExpr {
    CompiledExpr::Ap(Box::new(a.into()), Box::new(b.into()))
}

impl CompiledExpr {
    pub fn compile(e: &ast::Expr) -> CompiledExpr {
        match e {
            ast::Expr::Int(i) => CompiledExpr::Int(*i),
            ast::Expr::Var(s) => match s.as_str() {
                "if" => CompiledExpr::Comb(Comb::Cond),
                s => CompiledExpr::Var(s.to_owned()),
            },
            ast::Expr::BinOp(l, o, r) => {
                let op_comb = match o {
                    ast::BinOp::Cons => Comb::P,
                    ast::BinOp::Plus => Comb::Plus,
                    ast::BinOp::Minus => Comb::Minus,
                    ast::BinOp::Times => Comb::Times,
                    ast::BinOp::Divide => Comb::Divide,
                    ast::BinOp::Eq => Comb::Eq,
                    ast::BinOp::Neq => Comb::Neq,
                    ast::BinOp::Gt => Comb::Gt,
                    ast::BinOp::Gte => Comb::Gte,
                    ast::BinOp::Lt => Comb::Lt,
                    ast::BinOp::Lte => Comb::Lte,
                    ast::BinOp::And => Comb::And,
                    ast::BinOp::Or => Comb::Or,
                };
                let l = CompiledExpr::compile(l);
                let r = CompiledExpr::compile(r);
                cap(cap(op_comb, l), r)
            }
            ast::Expr::Not(e) => cap(Comb::Not, CompiledExpr::compile(e)),
            ast::Expr::Ap(l, r) => {
                let l = CompiledExpr::compile(l);
                let r = CompiledExpr::compile(r);
                cap(l, r)
            }
            ast::Expr::Lam(_, _) => todo!(),
        }
    }

    pub fn abstract_var(self, n: &str) -> CompiledExpr {
        match self {
            CompiledExpr::Comb(c) => cap(Comb::K, c),
            CompiledExpr::Ap(l, r) => cap(cap(Comb::S, l.abstract_var(n)), r.abstract_var(n)),
            CompiledExpr::Var(s) => {
                if s == n {
                    CompiledExpr::Comb(Comb::I)
                } else {
                    cap(CompiledExpr::Comb(Comb::K), CompiledExpr::Var(s))
                }
            }
            i @ CompiledExpr::Int(_) => cap(CompiledExpr::Comb(Comb::K), i),
        }
    }
}
