use superg::bracket::BracketCompiler;
use superg::compiled_expr::ExprCompiler;
use superg::kiselyov::{LazyCompiler, LazyOptCompiler, LinearCompiler, StrictCompiler};
use superg::lexer::lex;
use superg::parser::parse;
use superg::turner::TurnerEngine;

#[test]
fn test_id() {
    assert_runs_to_int(
        "test_id",
        r#"
(defun id (x) x)
(defun main () (id 42))
        "#,
        42,
        StepLimit(10),
    );
}

#[test]
fn test_k() {
    assert_runs_to_int(
        "test_k",
        r#"
(defun k (x y) x)
(defun main () (k 42 84))
        "#,
        42,
        StepLimit(20),
    );
}

#[test]
fn test_s() {
    assert_runs_to_int(
        "test_s",
        r#"
(defun s (f g x) (f x (g x)))
(defun k (x y) x)
(defun main () (s k k 42))
        "#,
        42,
        StepLimit(200),
    );
}

#[test]
fn test_b() {
    assert_runs_to_int(
        "test_b",
        r#"
(defun b (f g x) (f (g x)))
(defun k (x y) x)
(defun i (x) x)
(defun main () (b i i 42))
    "#,
        42,
        StepLimit(200),
    );
}

#[test]
fn test_add() {
    assert_runs_to_int(
        "test_add",
        r#"
(defun main () (+ 2 40))
        "#,
        42,
        StepLimit(10),
    );
}

#[test]
fn test_add_indirect() {
    assert_runs_to_int(
        "test_add_indirect",
        r#"
(defun id (x) x)
(defun main () (+ (id 2) (id 40)))
        "#,
        42,
        StepLimit(20),
    );
}

#[test]
fn test_cond() {
    assert_runs_to_int(
        "test_cond",
        r#"
(defun main () (if 0 1000 (if 1 42 2000)))
        "#,
        42,
        StepLimit(10),
    )
}

#[test]
fn test_cond_add() {
    assert_runs_to_int(
        "test_cond_add1",
        r#"
(defun main () (+ 2 (if 0 30 40)))
        "#,
        42,
        StepLimit(10),
    );

    assert_runs_to_int(
        "test_cond_add2",
        r#"
(defun main () (if 0 30 (+ 40 2)))
        "#,
        42,
        StepLimit(10),
    );
}

#[test]
fn test_eq() {
    assert_runs_to_int(
        "test_eq",
        r#"
    (defun id (x) x)
    (defun k (x y) x)
    (defun main () (= (k 1 1000) 0))
            "#,
        0,
        StepLimit(20),
    );
}

#[test]
fn test_cond_eq() {
    assert_runs_to_int(
        "test_cond_eq",
        r#"
    (defun main () (if (= 2 2) 42 43))
            "#,
        42,
        StepLimit(10),
    );
}

#[test]
fn test_factorial() {
    // TODO: was 5! = 120
    assert_runs_to_int(
        "test_factorial",
        r#"
    (defun fac (n)
      (if (= n 1)
          1
          (* n (fac (- n 1)))))
    (defun main () (fac 2))
            "#,
        2,
        StepLimit(1000),
    );
}

#[test]
fn test_fibonacci() {
    assert_runs_to_int(
        "test_fibonacci",
        r#"
    (defun fib (n)
      (if (< n 2) 
          n
          (+ (fib (- n 1)) (fib (- n 2)))))
    (defun main () (fib 5))
            "#,
        5,
        StepLimit(2000),
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
    assert_runs_to_int("test_ackermann", program, 125, StepLimit(10_000_000));
}

#[derive(Copy, Clone)]
struct StepLimit(usize);

fn assert_runs_to_int(_test_name: &str, program: &str, v: i32, l: StepLimit) {
    println!("Bracket");
    assert_runs_to_int_gen(BracketCompiler, _test_name, program, v, l);
    println!("Strict");
    assert_runs_to_int_gen(StrictCompiler, _test_name, program, v, l);
}

fn assert_runs_to_int_gen<C: ExprCompiler>(
    mut compiler: C,
    _test_name: &str,
    program: &str,
    v: i32,
    l: StepLimit,
) {
    let parsed = parse(lex(program));

    let mut engine = TurnerEngine::compile(&mut compiler, &parsed).with_debug();
    // disabled by default because it slows things down a lot, enable for debugging
    //engine.set_dump_path(format!("/tmp/{}", _test_name));
    engine.set_step_limit(l.0);

    let ptr = engine.run();
    assert_eq!(engine.get_int(ptr), Some(v));
}
