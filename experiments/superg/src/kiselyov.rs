use crate::{
    ast::Expr,
    compiled_expr::{Comb, CompiledExpr},
};

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Fun(Box<Type>, Box<Type>),
}

type Context = Vec<Type>;

// Same as ast::Expr but using de Bruijn indices
#[derive(Debug, PartialEq, Clone)]
pub enum BExpr {
    Int(i32),
    Var(usize),
    SVar(String),
    BinOp(Box<BExpr>, crate::ast::BinOp, Box<BExpr>),
    Not(Box<BExpr>),
    Ap(Box<BExpr>, Box<BExpr>),
    Lam(Box<BExpr>),
}

#[derive(Debug, PartialEq, Clone)]
pub struct TypedExpr {
    expr: BExpr,
    n: usize,
}

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

fn compile(e: &BExpr) -> CompiledExpr {
    let n = infer_n(e);
    match e {
        BExpr::Var(i) => {
            if n == 1 {
                CompiledExpr::Comb(Comb::I)
            } else {
                // TODO: Check if decrementing var works
                let comp_inner = compile(&BExpr::Var(i - 1));
                semantic(0, CompiledExpr::Comb(Comb::K), n - 1, comp_inner)
            }
        }
        BExpr::Lam(e) => {
            if n == 0 {
                CompiledExpr::Ap(Box::new(CompiledExpr::Comb(Comb::K)), Box::new(compile(e)))
            } else {
                // TODO: Check
                compile(e)
            }
        }
        BExpr::Ap(e1, e2) => semantic(infer_n(e1), compile(e1), infer_n(e2), compile(e2)),
        BExpr::Int(i) => CompiledExpr::Int(*i),
        BExpr::SVar(s) => CompiledExpr::Var(s.clone()),
        BExpr::BinOp(_, _, _) => todo!(),
        BExpr::Not(_) => todo!(),
    }
}

fn semantic(n1: usize, e1: CompiledExpr, n2: usize, e2: CompiledExpr) -> CompiledExpr {
    match (n1, n2) {
        (0, 0) => cap(e1, e2),
        (0, n2) => semantic(0, cap(Comb::B, e1), n2 - 1, e2),
        (n1, 0) => semantic(0, cap(cap(Comb::C, Comb::C), e2), n1 - 1, e1),
        (n1, n2) => semantic(
            n1,
            semantic(0, CompiledExpr::Comb(Comb::S), n1 - 1, e1),
            n2 - 1,
            e2,
        ),
        _ => todo!(),
    }
}

fn cap<F: Into<CompiledExpr>, A: Into<CompiledExpr>>(f: F, a: A) -> CompiledExpr {
    CompiledExpr::Ap(Box::new(f.into()), Box::new(a.into()))
}

fn to_debruijn(e: &Expr, vars: &mut Vec<String>) -> BExpr {
    match e {
        Expr::Int(i) => BExpr::Int(*i),
        Expr::Var(v) => {
            if let Some(i) = vars.iter().position(|x| x == v) {
                BExpr::Var(vars.len() - 1 - i)
            } else {
                println!("Did not find {} in {:?}", v, vars);
                BExpr::SVar(v.clone())
            }
        }
        Expr::BinOp(l, o, r) => BExpr::BinOp(
            Box::new(to_debruijn(l, vars)),
            *o,
            Box::new(to_debruijn(r, vars)),
        ),
        Expr::Not(e) => BExpr::Not(Box::new(to_debruijn(e, vars))),
        Expr::Ap(f, a) => BExpr::Ap(
            Box::new(to_debruijn(f, vars)),
            Box::new(to_debruijn(a, vars)),
        ),
        Expr::Lam(v, e) => {
            vars.push(v.clone());
            let e = to_debruijn(e, vars);
            vars.pop().unwrap();
            BExpr::Lam(Box::new(e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lex, parser::parse_expr};

    use super::BExpr;
    #[test]
    fn test_to_debruijn() {
        assert_de_bruijn_equals("lam (x y) (y x)", lam(lam(ap(z(), s(z())))));
    }

    #[test]
    fn test_to_debruijn_k() {
        assert_de_bruijn_equals("lam (x y) x", lam(lam(s(z()))));
    }

    #[test]
    fn test_to_debruijn_s() {
        assert_de_bruijn_equals(
            "lam (x y z) ((x z) (y z))",
            lam(lam(lam(ap(ap(s(s(z())), z()), ap(s(z()), z()))))),
        );
    }

    #[test]
    fn kiselyov() {
        let expr = "lam (x y) (y x)";
        let mut tokens = lex(expr);
        let parsed_expr = parse_expr(&mut tokens);
        let bexpr = to_debruijn(&parsed_expr, &mut vec![]);
        let compiled = compile(&bexpr);
        use Comb::{B, C, I};
        assert_eq!(compiled, cap(cap(B, cap(C, I)), I))
    }

    fn assert_de_bruijn_equals(expr: &str, expected_bexpr: BExpr) {
        let mut tokens = lex(expr);
        let parsed_expr = parse_expr(&mut tokens);
        let bexpr = to_debruijn(&parsed_expr, &mut vec![]);
        assert_eq!(bexpr, expected_bexpr);
    }

    fn z() -> BExpr {
        BExpr::Var(0)
    }
    fn s(e: BExpr) -> BExpr {
        if let BExpr::Var(i) = e {
            BExpr::Var(i + 1)
        } else {
            panic!()
        }
    }

    fn lam(e: BExpr) -> BExpr {
        BExpr::Lam(Box::new(e))
    }

    fn ap(f: BExpr, a: BExpr) -> BExpr {
        BExpr::Ap(Box::new(f), Box::new(a))
    }
}
