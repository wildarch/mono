use crate::{ast::*, lexer::Token};

pub fn parse(mut tokens: Vec<Token>) -> Program {
    // We will pop from left to right
    tokens.reverse();
    let mut defs = Vec::new();

    while !tokens.is_empty() {
        defs.push(parse_def(&mut tokens));
    }
    Program { defs }
}

fn parse_def(tokens: &mut Vec<Token>) -> Def {
    eat(tokens, Token::LParen);
    eat(tokens, Token::Symbol("defun".to_owned()));

    let name = parse_symbol(tokens);
    let params = parse_params(tokens);
    let expr = parse_expr(tokens);

    eat(tokens, Token::RParen);

    Def { name, params, expr }
}

fn parse_symbol(tokens: &mut Vec<Token>) -> String {
    match tokens.pop() {
        None => panic!("Expected a symbol, but found nothing"),
        Some(Token::Symbol(s)) => s,
        Some(t) => panic!("Expected a symbol, but found {:?}", t),
    }
}

fn parse_params(tokens: &mut Vec<Token>) -> Vec<String> {
    eat(tokens, Token::LParen);
    let mut params = Vec::new();
    while tokens.last() != Some(&Token::RParen) {
        params.push(parse_symbol(tokens));
    }
    eat(tokens, Token::RParen);
    params
}

fn parse_expr(tokens: &mut Vec<Token>) -> Expr {
    match tokens.pop() {
        None => panic!("Expected an expression, but found nothing"),
        Some(Token::Integer(i)) => Expr::Int(i),
        Some(Token::LParen) => {
            let mut expr = parse_expr(tokens);
            while tokens.last() != Some(&Token::RParen) {
                expr = ap(expr, parse_expr(tokens));
            }
            eat(tokens, Token::RParen);
            expr
        }
        Some(Token::Symbol(s)) => match s.as_str() {
            // Pair
            ":" => parse_binop(tokens, BinOp::Cons),
            // Arithmetic
            "+" => parse_binop(tokens, BinOp::Plus),
            "-" => parse_binop(tokens, BinOp::Minus),
            "*" => parse_binop(tokens, BinOp::Times),
            "/" => parse_binop(tokens, BinOp::Divide),
            // Comparison
            "=" => parse_binop(tokens, BinOp::Eq),
            "/=" => parse_binop(tokens, BinOp::Neq),
            ">" => parse_binop(tokens, BinOp::Gt),
            "<" => parse_binop(tokens, BinOp::Lt),
            ">=" => parse_binop(tokens, BinOp::Gte),
            "<=" => parse_binop(tokens, BinOp::Lte),
            // Other identifiers
            _ => Expr::Var(s),
        },
        Some(Token::RParen) => panic!("Unexpected right paren"),
    }
}

fn parse_binop(tokens: &mut Vec<Token>, op: BinOp) -> Expr {
    binop(parse_expr(tokens), op, parse_expr(tokens))
}

fn eat(tokens: &mut Vec<Token>, expected_token: Token) {
    match tokens.pop() {
        None => panic!("Expected {:?}, but found nothing to eat", expected_token),
        Some(t) => {
            if t != expected_token {
                panic!("Expected {:?}, but found {:?}", expected_token, t)
            }
        }
    }
}

fn ap(a: Expr, b: Expr) -> Expr {
    Expr::Ap(Box::new(a), Box::new(b))
}

fn binop(l: Expr, op: BinOp, r: Expr) -> Expr {
    Expr::BinOp(Box::new(l), op, Box::new(r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::BinOp::*;
    use crate::lexer::lex;

    #[test]
    fn parse_factorial() {
        let program = r#"
(defun fac (n) 
  (if (= n 1)
      1
      (* n (fac (- n 1)))))
        "#;

        let parsed = parse(lex(program));
        assert_eq!(
            parsed,
            Program {
                defs: vec![Def {
                    name: "fac".to_owned(),
                    params: vec!["n".to_owned()],
                    expr: ap(
                        ap(ap(var("if"), binop(var("n"), Eq, int(1))), int(1)),
                        binop(
                            var("n"),
                            Times,
                            ap(var("fac"), binop(var("n"), Minus, int(1)))
                        )
                    ),
                }]
            }
        );
    }

    fn var(s: &str) -> Expr {
        Expr::Var(s.to_owned())
    }

    fn int(i: i32) -> Expr {
        Expr::Int(i)
    }
}
