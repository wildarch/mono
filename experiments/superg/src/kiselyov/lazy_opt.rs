/// The lazy compilation algorithm, as given in fig.8
use crate::compiled_expr::{cap, Comb, CompiledExpr};

use super::BExpr;

pub fn compile_lazy_opt(e: &BExpr) -> CompiledExpr {
    match compile(e) {
        CExpr::Closed(e) => e,
        _ => unreachable!("expect to reach a closed expression"),
    }
}

#[derive(Debug, Clone)]
enum CExpr {
    Closed(CompiledExpr),
    Value,
    Next(Box<CExpr>),
    Weak(Box<CExpr>),
}

fn compile(e: &BExpr) -> CExpr {
    match e {
        BExpr::Var(i) => {
            let mut e = CExpr::Value;
            for _ in 0..*i {
                e = CExpr::Weak(Box::new(e));
            }
            e
        }
        BExpr::Lam(e) => match compile(e) {
            CExpr::Closed(d) => CExpr::Closed(cap(Comb::K, d)),
            CExpr::Value => CExpr::Closed(CompiledExpr::Comb(Comb::I)),
            CExpr::Next(e) => *e,
            CExpr::Weak(e) => semantic(CExpr::Closed(CompiledExpr::Comb(Comb::K)), *e),
        },
        BExpr::Ap(a, b) => semantic(compile(a), compile(b)),
        _ => todo!(),
    }
}

fn semantic(e1: CExpr, e2: CExpr) -> CExpr {
    use CExpr::*;
    fn weak(e: CExpr) -> CExpr {
        Weak(Box::new(e))
    }
    fn next(e: CExpr) -> CExpr {
        Next(Box::new(e))
    }
    match (e1, e2) {
        (Weak(e1), Weak(e2)) => weak(semantic(*e1, *e2)),
        (Weak(e), Closed(d)) => weak(semantic(*e, Closed(d))),
        (Closed(d), Weak(e)) => weak(semantic(Closed(d), *e)),
        (Weak(e), Value) => Next(e),
        (Value, Weak(e)) => next(semantic(Closed(cap(Comb::C, Comb::I)), *e)),
        (Weak(e1), Next(e2)) => next(semantic(
            semantic(Closed(CompiledExpr::Comb(Comb::B)), *e1),
            *e2,
        )),
        (Next(e1), Weak(e2)) => next(semantic(
            semantic(Closed(CompiledExpr::Comb(Comb::C)), *e1),
            *e2,
        )),
        (Next(e1), Next(e2)) => next(semantic(
            semantic(Closed(CompiledExpr::Comb(Comb::S)), *e1),
            *e2,
        )),
        (Next(e), Value) => next(semantic(
            semantic(Closed(CompiledExpr::Comb(Comb::S)), *e),
            Closed(CompiledExpr::Comb(Comb::I)),
        )),
        (Value, Next(e)) => next(semantic(Closed(cap(Comb::S, Comb::I)), *e)),

        (Closed(d), Next(e)) => next(semantic(Closed(cap(Comb::B, d)), *e)),
        (Closed(d), Value) => next(Closed(d)),
        (Value, Closed(d)) => next(Closed(cap(cap(Comb::C, Comb::I), d))),
        (Value, Value) => unreachable!("can't happen for simple types"),
        (Next(e), Closed(d)) => next(semantic(Closed(cap(cap(Comb::C, Comb::C), d)), *e)),
        (Closed(d1), Closed(d2)) => Closed(cap(d1, d2)),
    }
}

#[cfg(test)]
mod tests {
    use super::super::to_debruijn;
    use super::*;
    use crate::{
        lex,
        parser::{parse_compiled_expr, parse_expr},
    };

    // Testcases taken from Kiselyov's paper, Table 1.
    #[test]
    fn kiselyov_examples() {
        assert_compiles_to("lam (x) (lam (y) y)", "KI");
        assert_compiles_to("lam (x) (lam (y) x)", "K");
        assert_compiles_to("lam (x) (lam (y) (x y))", "I");
        assert_compiles_to("lam (x) (lam (y) (y x))", "CI");
        assert_compiles_to("lam (x) (lam (y) (lam (z) (z x)))", "BK(CI)");
        // Note: BK(BKI) in Kiselyov
        assert_compiles_to("lam (x) (lam (y) (lam (z) ((lam (w) w) x)))", "BK(BKI)");
        assert_compiles_to("lam (x) (lam (y) (lam (z) ((x z) (y z))))", "S");
    }

    #[test]
    fn kiselyov_worsecase() {
        assert_compiles_to("lam (x) (lam (y) (y x))", "CI");
        assert_compiles_to("lam (x) (lam (y) (lam (z) (z y x)))", "C(BC(CI))");
        assert_compiles_to(
            "lam (x) (lam (y) (lam (z) (lam (a) (a z y x))))",
            "C(BC(B(BC) (C(BC(CI)))))",
        );
    }

    fn assert_compiles_to(expr: &str, compiled_expr: &str) {
        // Compile expr
        let mut tokens = lex(expr);
        let parsed_expr = parse_expr(&mut tokens);
        let bexpr = to_debruijn(&parsed_expr, &mut vec![]);
        let actual_compiled_expr = compile_lazy_opt(&bexpr);
        // Parse expected expr
        let expected_compiled_expr = parse_compiled_expr(lex(compiled_expr));

        assert_eq!(actual_compiled_expr, expected_compiled_expr);
    }
}
