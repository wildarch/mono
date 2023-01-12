use superg::bracket::BracketCompiler;
use superg::compiled_expr::ExprCompiler;
use superg::kiselyov::{LazyCompiler, LazyOptCompiler, LinearCompiler, StrictCompiler};
use superg::lexer::lex;
use superg::parser::parse;
use superg::tigre::TigreEngine;
use superg::turner::TurnerEngine;
use superg::Engine;

#[test]
fn test_id() {
    assert_runs_to_int(
        r#"
(defun id (x) x)
(defun main () (id 42))
        "#,
        42,
    );
}

#[test]
fn test_k() {
    assert_runs_to_int(
        r#"
(defun k (x y) x)
(defun main () (k 42 84))
        "#,
        42,
    );
}

#[test]
fn test_s() {
    assert_runs_to_int(
        r#"
(defun s (f g x) (f x (g x)))
(defun k (x y) x)
(defun main () (s k k 42))
        "#,
        42,
    );
}

#[test]
fn test_b() {
    assert_runs_to_int(
        r#"
(defun b (f g x) (f (g x)))
(defun k (x y) x)
(defun i (x) x)
(defun main () (b i i 42))
    "#,
        42,
    );
}

#[test]
fn test_add() {
    assert_runs_to_int(
        r#"
(defun main () (+ 2 40))
        "#,
        42,
    );
}

#[test]
fn test_add_indirect() {
    assert_runs_to_int(
        r#"
(defun id (x) x)
(defun main () (+ (id 2) (id 40)))
        "#,
        42,
    );
}

#[test]
fn test_cond() {
    assert_runs_to_int(
        r#"
(defun main () (if 0 1000 (if 1 42 2000)))
        "#,
        42,
    )
}

#[test]
fn test_cond_add() {
    assert_runs_to_int(
        r#"
(defun main () (+ 2 (if 0 30 40)))
        "#,
        42,
    );

    assert_runs_to_int(
        r#"
(defun main () (if 0 30 (+ 40 2)))
        "#,
        42,
    );
}

#[test]
fn test_eq() {
    assert_runs_to_int(
        r#"
    (defun id (x) x)
    (defun k (x y) x)
    (defun main () (= (k 1 1000) 0))
            "#,
        0,
    );
}

#[test]
fn test_cond_eq() {
    assert_runs_to_int(
        r#"
    (defun main () (if (= 2 2) 42 43))
            "#,
        42,
    );
}

#[test]
fn test_factorial() {
    assert_runs_to_int(
        r#"
    (defun fac (n)
      (if (= n 1)
          1
          (* n (fac (- n 1)))))
    (defun main () (fac 5))
            "#,
        120,
    );
}

#[test]
fn test_fibonacci() {
    assert_runs_to_int(
        r#"
    (defun fib (n)
      (if (< n 2) 
          n
          (+ (fib (- n 1)) (fib (- n 2)))))
    (defun main () (fib 5))
            "#,
        5,
    );
}

#[test]
fn test_ackermann() {
    let program = r#"
(defun ack (x z) (if (= x 0)
                     (+ z 1)
                     (if (= z 0)
                         (ack (- x 1) 1)
                         (ack (- x 1) (ack x (- z 1))))))
(defun main () (ack 3 4))
    "#;
    assert_runs_to_int(program, 125);
}

#[derive(Copy, Clone)]
struct StepLimit(usize);

fn assert_runs_to_int(program: &str, v: i32) {
    println!("=== Turner ===");
    assert_runs_to_int_all_compilers::<TurnerEngine>(program, v);
    println!("=== Tigre ===");
    assert_runs_to_int_all_compilers::<TigreEngine>(program, v);
}

fn assert_runs_to_int_all_compilers<E: Engine>(program: &str, v: i32) {
    println!("Bracket");
    assert_runs_to_int_gen::<_, E>(BracketCompiler, program, v);
    println!("Strict");
    assert_runs_to_int_gen::<_, E>(StrictCompiler, program, v);
    println!("Lazy");
    assert_runs_to_int_gen::<_, E>(LazyCompiler, program, v);
    println!("Lazy opt");
    assert_runs_to_int_gen::<_, E>(LazyOptCompiler, program, v);
    println!("Linear");
    assert_runs_to_int_gen::<_, E>(LinearCompiler, program, v);
}

fn assert_runs_to_int_gen<C: ExprCompiler, E: Engine>(mut compiler: C, program: &str, v: i32) {
    let parsed = parse(lex(program));
    let mut engine = E::compile(&mut compiler, &parsed);
    let res = engine.run();
    assert_eq!(res, v);
}
