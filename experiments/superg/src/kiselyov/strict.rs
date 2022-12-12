/// The strict compilation algorithm, as given in fig.6
use crate::compiled_expr::{cap, Comb, CompiledExpr};

use super::BExpr;

fn infer_n(e: &BExpr) -> usize {
    let n = match e {
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
    };
    println!("infer_n({:?}) = {}", e, n);
    n
}

pub fn compile_strict(e: &BExpr) -> CompiledExpr {
    let c = match e {
        BExpr::Var(i) => {
            let n = infer_n(e);
            if n == 1 {
                CompiledExpr::Comb(Comb::I)
            } else {
                let comp_inner = compile_strict(&BExpr::Var(i - 1));
                semantic(0, CompiledExpr::Comb(Comb::K), n - 1, comp_inner)
            }
        }
        BExpr::Lam(e) => {
            let n = infer_n(e);
            if n == 0 {
                CompiledExpr::Ap(
                    Box::new(CompiledExpr::Comb(Comb::K)),
                    Box::new(compile_strict(e)),
                )
            } else {
                compile_strict(e)
            }
        }
        BExpr::Ap(e1, e2) => semantic(
            infer_n(e1),
            compile_strict(e1),
            infer_n(e2),
            compile_strict(e2),
        ),
        BExpr::Int(i) => CompiledExpr::Int(*i),
        BExpr::SVar(s) => CompiledExpr::Var(s.clone()),
        BExpr::BinOp(_, _, _) => todo!(),
        BExpr::Not(_) => todo!(),
    };
    println!("compile({:?}) = {:#?}", e, c);
    c
}

fn semantic(n1: usize, e1: CompiledExpr, n2: usize, e2: CompiledExpr) -> CompiledExpr {
    let dbg = format!("semantic({}, {:?}, {}, {:?}) = ", n1, e1, n2, e2);
    let c = match (n1, n2) {
        (0, 0) => cap(e1, e2),
        (0, n2) => semantic(0, cap(Comb::B, e1), n2 - 1, e2),
        (n1, 0) => semantic(0, cap(cap(Comb::C, Comb::C), e2), n1 - 1, e1),
        (n1, n2) => semantic(
            n1 - 1,
            semantic(0, CompiledExpr::Comb(Comb::S), n1 - 1, e1),
            n2 - 1,
            e2,
        ),
    };
    println!("{}{:?}", dbg, c);
    c
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
        let compiled = compile_strict(&bexpr);
        use Comb::{B, I, K, S};
        assert_eq!(compiled, cap(cap(B, cap(S, I)), cap(cap(B, K), I)))
    }
}
