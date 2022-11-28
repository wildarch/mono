#[derive(Debug, PartialEq)]
pub struct Program {
    pub defs: Vec<Def>,
}

#[derive(Debug, PartialEq)]
pub struct Def {
    pub name: String,
    pub params: Vec<String>,
    pub expr: Expr,
}

#[derive(Debug, PartialEq)]
pub enum Expr {
    Int(i32),
    Var(String),
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    Not(Box<Expr>),
    Ap(Box<Expr>, Box<Expr>),
}

#[derive(Debug, PartialEq)]
pub enum BinOp {
    Cons,
    // Arithmetic
    Plus,
    Minus,
    Times,
    Divide,
    // Comparison
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    And,
    Or,
}
