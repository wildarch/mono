/// The lazy compilation algorithm, as given in fig.8
use crate::compiled_expr::{cap, Comb, CompiledExpr};

use super::BExpr;

#[derive(Debug, Clone, Copy, PartialEq)]
enum ContextElem {
    Ignored,
    Used,
}
type Context = Vec<ContextElem>;

fn infer_context(e: &BExpr) -> Context {
    use ContextElem::*;
    match e {
        BExpr::Var(i) => {
            let mut v = vec![Used];
            for _ in 0..*i {
                v.push(Ignored);
            }
            v
        }
        BExpr::Lam(e) => {
            let mut ctx = infer_context(e);
            ctx.pop();
            ctx
        }
        BExpr::Ap(f, a) => unify_contexts(infer_context(f), infer_context(a)),
        _ => todo!(),
    }
}

fn unify_contexts(mut a: Context, mut b: Context) -> Context {
    use ContextElem::*;
    let mut ctx = Context::new();
    loop {
        let e = match (a.pop(), b.pop()) {
            (Some(Used), _) | (_, Some(Used)) => Used,
            (None, Some(Ignored)) | (Some(Ignored), None) | (Some(Ignored), Some(Ignored)) => {
                Ignored
            }
            (None, None) => {
                // We are building the context in reverse order
                ctx.reverse();
                return ctx;
            }
        };
        ctx.push(e);
    }
}

pub fn compile_lazy_opt(e: &BExpr) -> CompiledExpr {
    use ContextElem::*;
    let compiled = match e {
        BExpr::Var(_) => {
            // Lazy weakening allows us to just reduce to I in all cases.
            // K is inserted in Abs1 case for Lam below.
            CompiledExpr::Comb(Comb::I)
        }
        BExpr::Lam(e) => {
            // Context for the inner expression
            let mut ctx = infer_context(e);
            match ctx.last() {
                // Abs0
                None => cap(Comb::K, compile_lazy_opt(e)),
                // Abs1
                Some(Ignored) => {
                    ctx.pop().unwrap();
                    semantic(Context::new(), Comb::K, ctx, compile_lazy_opt(e))
                }
                // Abs
                Some(Used) => compile_lazy_opt(e),
            }
        }
        BExpr::Ap(e1, e2) => semantic(
            infer_context(e1),
            compile_lazy_opt(e1),
            infer_context(e2),
            compile_lazy_opt(e2),
        ),
        _ => todo!(),
    };
    optimize_exhaustive(compiled)
}

fn optimize_exhaustive(mut e: CompiledExpr) -> CompiledExpr {
    loop {
        let optimized = optimize(e.clone());
        if optimized == e {
            return e;
        }
        println!("Optimize {:?} to {:?}", e, optimized);
        e = optimized;
    }
}

fn optimize(e: CompiledExpr) -> CompiledExpr {
    match e {
        CompiledExpr::Ap(a0, a1) => match *a0 {
            CompiledExpr::Ap(b0, b1) => match *b0 {
                CompiledExpr::Ap(c0, c1) => opt_ap4(*c0, *c1, *b1, *a1),
                _ => opt_ap3(*b0, *b1, *a1),
            },
            _ => opt_ap2(*a0, *a1),
        },
        _ => e,
    }
}

fn opt_ap2(a0: CompiledExpr, a1: CompiledExpr) -> CompiledExpr {
    match (a0, a1) {
        (a0, a1) => cap(optimize(a0), optimize(a1)),
    }
}

fn opt_ap3(a0: CompiledExpr, a1: CompiledExpr, a2: CompiledExpr) -> CompiledExpr {
    match (a0, a1, a2) {
        // B d I = d
        (CompiledExpr::Comb(Comb::B), d, CompiledExpr::Comb(Comb::I)) => d,
        // CBI = I
        (CompiledExpr::Comb(Comb::C), CompiledExpr::Comb(Comb::B), CompiledExpr::Comb(Comb::I)) => {
            CompiledExpr::Comb(Comb::I)
        }
        // C (BBd) I = d
        (CompiledExpr::Comb(Comb::C), p, CompiledExpr::Comb(Comb::I)) => match p {
            CompiledExpr::Ap(p0, p1) => {
                let p0 = *p0;
                let p1 = *p1;
                match p0 {
                    CompiledExpr::Ap(q0, q1) => {
                        let q0 = *q0;
                        let q1 = *q1;
                        // C (q0 q1 p1) I
                        if q0 == CompiledExpr::Comb(Comb::B) && q1 == CompiledExpr::Comb(Comb::B) {
                            // C (BBd) I = d
                            optimize(p1)
                        } else {
                            cap(cap(Comb::C, opt_ap3(q0, q1, p1)), Comb::I)
                        }
                    }
                    _ => cap(cap(Comb::C, opt_ap2(p0, p1)), Comb::I),
                }
            }
            _ => cap(cap(Comb::C, optimize(p)), Comb::I),
        },
        (a0, a1, a2) => cap(opt_ap2(a0, a1), optimize(a2)),
    }
}

fn opt_ap4(a0: CompiledExpr, a1: CompiledExpr, a2: CompiledExpr, a3: CompiledExpr) -> CompiledExpr {
    match (a0, a1, a2, a3) {
        // C f x g = f g x
        (CompiledExpr::Comb(Comb::C), f, g, x) => opt_ap3(f, x, g),
        // B f x g = f (x g)
        (CompiledExpr::Comb(Comb::B), f, g, x) => opt_ap2(f, opt_ap2(x, g)),
        (a0, a1, a2, a3) => cap(opt_ap3(a0, a1, a2), optimize(a3)),
    }
}

fn semantic<E1: Into<CompiledExpr>, E2: Into<CompiledExpr>>(
    mut c1: Context,
    e1: E1,
    mut c2: Context,
    e2: E2,
) -> CompiledExpr {
    use ContextElem::*;
    let e1 = e1.into();
    let e2 = e2.into();

    match (c1.pop(), c2.pop()) {
        (None, None) => cap(e1, e2),
        (None, Some(Used)) => semantic(Context::new(), cap(Comb::B, e1), c2, e2),
        (Some(Used), None) => semantic(Context::new(), cap(cap(Comb::C, Comb::C), e2), c1, e1),
        (Some(Used), Some(Used)) => semantic(
            c1.clone(),
            semantic(Context::new(), Comb::S, c1, e1),
            c2,
            e2,
        ),
        // From fig.8.
        (Some(Ignored), Some(Ignored)) => semantic(c1, e1, c2, e2),
        (Some(Ignored), Some(Used)) => semantic(
            // Need to check, maybe used Context::new() instead
            c1.clone(),
            semantic(Context::new(), Comb::B, c1, e1),
            c2,
            e2,
        ),
        (Some(Used), Some(Ignored)) => semantic(
            // Need to check, maybe used Context::new() instead
            c1.clone(),
            semantic(Context::new(), Comb::C, c1, e1),
            c2,
            e2,
        ),
        // Not clearly described in the paper, but necessary for matching results.
        // This is correct, because we can model a shorter context as one with implicit 'unused' elements.
        (None, Some(Ignored)) => semantic(c1, e1, c2, e2),
        (Some(Ignored), None) => semantic(c1, e1, c2, e2),
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
        assert_compiles_to("lam (x) (lam (y) (lam (z) ((lam (w) w) x)))", "BKK");
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
