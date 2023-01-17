use criterion::{criterion_group, criterion_main, Criterion};
use superg::compiled_expr::ExprCompiler;

use std::time::{Duration, Instant};
use superg::bracket::BracketCompiler;
use superg::turner::TurnerEngine;
use superg::{lex, parse, Engine};

mod miranda;

criterion_group!(turner, bench_fib_20);
criterion_main!(turner, miranda::benches);

const FIB_20: &str = r#"
    (defun fib (n)
      (if (< n 2) 
          n
          (+ (fib (- n 1)) (fib (- n 2)))))
    (defun main () (fib 20))
"#;

pub fn bench_fib_20(c: &mut Criterion) {
    assert_runs_to_int::<_, TurnerEngine>(BracketCompiler, c, "turner_fib_20", FIB_20, 6765);
}

fn assert_runs_to_int<C: ExprCompiler, E: Engine>(
    mut compiler: C,
    c: &mut Criterion,
    test_name: &str,
    program: &str,
    v: i32,
) {
    let parsed = parse(lex(program));

    c.bench_function(test_name, |b| {
        b.iter_custom(|iters| {
            let mut elapsed = Duration::ZERO;
            for _ in 0..iters {
                let mut engine = E::compile(&mut compiler, &parsed);
                // Measure only the time spent reducing the graph
                let start = Instant::now();
                let res = engine.run();
                elapsed += start.elapsed();
                assert_eq!(res, v);
            }
            elapsed
        })
    });
}
