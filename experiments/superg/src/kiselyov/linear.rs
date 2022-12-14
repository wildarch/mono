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

    // Testcases taken from Kiselyov's paper, Table 1.
    #[test]
    fn kiselyov_examples() {
        assert_compiles_to("lam (x) (lam (y) y)", "KI");
        assert_compiles_to("lam (x) (lam (y) x)", "BKI");
        // Kiselyov' paper contains an error, it states the output is: C(BS(BKI))I
        assert_compiles_to("lam (x) (lam (y) (x y))", "C(BS(BKI))I");
        assert_compiles_to("lam (x) (lam (y) (y x))", "B(SI)(BKI)");
        assert_compiles_to("lam (x) (lam (y) (lam (z) (z x)))", "B2(SI) (B2 K(BKI))");
        assert_compiles_to(
            "lam (x) (lam (y) (lam (z) ((lam (w) w) x)))",
            "B3 I(B2 K(BKI))",
        );
        assert_compiles_to(
            "lam (x) (lam (y) (lam (z) ((x z) (y z))))",
            "C(B S2(C2 (B2 S(B2 K (BKI)))I)) (C(BS(BKI))I)",
        );
    }

    #[test]
    fn kiselyov_worsecase() {
        assert_compiles_to("lam (x) (lam (y) (y x))", "B(SI)(BKI)");
        assert_compiles_to(
            "lam (x) (lam (y) (lam (z) (z y x)))",
            "B(S2 (B(SI)(BKI))) (B2 K(BKI))",
        );
        assert_compiles_to(
            "lam (x) (lam (y) (lam (z) (lam (a) (a z y x))))",
            "B(S3 (B(S2 (B(SI)(BKI))) (B2 K(BKI)))) (B3 K(B2 K (BKI)))",
        );
    }

    fn assert_compiles_to(expr: &str, compiled_expr: &str) {
        // Compile expr
        let mut tokens = lex(expr);
        let parsed_expr = parse_expr(&mut tokens);
        let bexpr = to_debruijn(&parsed_expr, &mut vec![]);
        let actual_compiled_expr = compile_linear(&bexpr);
        // Parse expected expr
        let expected_compiled_expr = parse_compiled_expr(lex(compiled_expr));

        assert_eq!(actual_compiled_expr, expected_compiled_expr);
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
                    Comb::S(i) => write!(f, "S{}", i),
                    Comb::B(i) => write!(f, "B{}", i),
                    Comb::C(i) => write!(f, "C{}", i),
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

    use crate::lexer::Token;
    use std::collections::VecDeque;

    pub fn parse_compiled_expr(mut tokens: VecDeque<Token>) -> CompiledExpr {
        let mut expr = parse_compiled_expr_inner(&mut tokens);
        while !tokens.is_empty() {
            expr = cap(expr, parse_compiled_expr_inner(&mut tokens));
        }
        expr
    }

    fn parse_compiled_expr_inner(tokens: &mut VecDeque<Token>) -> CompiledExpr {
        use Comb::*;
        match tokens.pop_front() {
            None => panic!("Expected an expression, but found nothing"),
            Some(Token::Integer(i)) => CompiledExpr::Int(i),
            Some(Token::LParen) => {
                let mut expr = parse_compiled_expr_inner(tokens);
                while tokens.front() != Some(&Token::RParen) {
                    expr = cap(expr, parse_compiled_expr_inner(tokens));
                }
                eat(tokens, Token::RParen);
                expr
            }
            Some(Token::Symbol(s)) => CompiledExpr::Comb(match s.as_str() {
                "S" => S(1),
                "S2" => S(2),
                "S3" => S(3),
                "K" => K,
                "I" => I,
                "Y" => Y,
                "U" => U,
                "P" => P,
                "B" => B(1),
                "B2" => B(2),
                "B3" => B(3),
                "C" => C(1),
                "C2" => C(2),
                "C3" => C(3),
                "Plus" => Plus,
                "Minus" => Minus,
                "Times" => Times,
                "Divide" => Divide,
                "Cond" => Cond,
                "Eq" => Eq,
                "Neq" => Neq,
                "Gt" => Gt,
                "Gte" => Gte,
                "Lt" => Lt,
                "Lte" => Lte,
                "And" => And,
                "Or" => Or,
                "Not" => Not,
                "Abort" => Abort,
                _ => {
                    // Assume a string of combinators
                    return s
                        .chars()
                        .map(|c| CompiledExpr::Comb(parse_comb(c)))
                        .reduce(cap)
                        .unwrap();
                }
            }),
            Some(Token::RParen) => panic!("Unexpected right paren"),
        }
    }

    fn parse_comb(c: char) -> Comb {
        use Comb::*;
        match c {
            'S' => S(1),
            'K' => K,
            'I' => I,
            'Y' => Y,
            'U' => U,
            'P' => P,
            'B' => B(1),
            'C' => C(1),
            _ => panic!("Illegal combinator: '{}'", c),
        }
    }

    fn eat(tokens: &mut VecDeque<Token>, expected_token: Token) {
        match tokens.pop_front() {
            None => panic!("Expected {:?}, but found nothing to eat", expected_token),
            Some(t) => {
                if t != expected_token {
                    panic!("Expected {:?}, but found {:?}", expected_token, t)
                }
            }
        }
    }
}
