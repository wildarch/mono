/// The strict compilation algorithm, as given in fig.10
use super::BExpr;

use compiled_expr::*;

fn infer_n(e: &BExpr) -> usize {
    match e {
        BExpr::Var(i) => i + 1,
        BExpr::Lam(e) => {
            let n = infer_n(e);
            if n == 0 {
                0
            } else {
                n - 1
            }
        }
        BExpr::Ap(f, a) => {
            let n_f = infer_n(f);
            let n_a = infer_n(a);
            std::cmp::max(n_f, n_a)
        }
        BExpr::Int(_) => 0,
        BExpr::SVar(_) => 0,
        BExpr::BinOp(l, _, r) => std::cmp::max(infer_n(l), infer_n(r)),
        BExpr::Not(e) => infer_n(e),
    }
}

pub fn compile_linear(e: &BExpr) -> CompiledExpr {
    match e {
        BExpr::Var(i) => {
            let n = infer_n(e);
            if n == 1 {
                CompiledExpr::Comb(Comb::I)
            } else {
                let comp_inner = compile_linear(&BExpr::Var(i - 1));
                semantic(0, CompiledExpr::Comb(Comb::K), n - 1, comp_inner)
            }
        }
        BExpr::Lam(e) => {
            let n = infer_n(e);
            if n == 0 {
                CompiledExpr::Ap(
                    Box::new(CompiledExpr::Comb(Comb::K)),
                    Box::new(compile_linear(e)),
                )
            } else {
                compile_linear(e)
            }
        }
        BExpr::Ap(e1, e2) => semantic(
            infer_n(e1),
            compile_linear(e1),
            infer_n(e2),
            compile_linear(e2),
        ),
        BExpr::Int(i) => CompiledExpr::Int(*i),
        BExpr::SVar(s) => CompiledExpr::Var(s.clone()),
        BExpr::BinOp(_, _, _) => todo!(),
        BExpr::Not(_) => todo!(),
    }
}

fn semantic(n1: usize, e1: CompiledExpr, n2: usize, e2: CompiledExpr) -> CompiledExpr {
    use Comb::*;
    match (n1, n2) {
        (0, 0) => cap(e1, e2),
        (0, n) => cap(cap(B(n), e1), e2),
        (n, 0) => cap(cap(C(n), e1), e2),
        (n, m) => {
            if n == m {
                cap(cap(S(n), e1), e2)
            } else if n < m {
                cap(cap(B(m - n), cap(S(n), e1)), e2)
            } else {
                cap(cap(C(n - m), cap(cap(B(n - m), S(m)), e1)), e2)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::to_debruijn;
    use super::*;
    use crate::{lex, parser::parse_expr};

    #[test]
    fn example() {
        let expr = "lam (x y) (y x)";
        let mut tokens = lex(expr);
        let parsed_expr = parse_expr(&mut tokens);
        let bexpr = to_debruijn(&parsed_expr, &mut vec![]);
        let compiled = compile_linear(&bexpr);
        use Comb::{B, I, K, S};
        assert_eq!(compiled, cap(cap(B(1), cap(S(1), I)), cap(cap(B(1), K), I)))
    }
}

// TODO: dedupe with crate::compiled_expr
mod compiled_expr {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum Comb {
        S(usize),
        B(usize),
        C(usize),
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

    #[derive(Debug, Clone, PartialEq)]
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

    pub fn cap<A: Into<CompiledExpr>, B: Into<CompiledExpr>>(a: A, b: B) -> CompiledExpr {
        CompiledExpr::Ap(Box::new(a.into()), Box::new(b.into()))
    }
}
