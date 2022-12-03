use criterion::{criterion_group, criterion_main, Criterion};
use superg::turner::TurnerEngine;
use superg::{lex, parse};

pub fn bench_ackermann(c: &mut Criterion) {
    let program = r#"
(defun ack (x z) (if (= x 0)
                     (+ z 1)
                     (if (= z 0)
                         (ack (- x 1) 1)
                         (ack (- x 1) (ack x (- z 1))))))
(defun main () (ack 3 4))
    "#;
    assert_runs_to_int(c, "bench_ackermann", program, 125);
}

fn assert_runs_to_int(c: &mut Criterion, test_name: &str, program: &str, v: i32) {
    let parsed = parse(lex(program));
    let mut engine = TurnerEngine::compile(&parsed);

    c.bench_function(test_name, |b| {
        b.iter(|| {
            let ptr = engine.run();
            assert_eq!(engine.get_int(ptr), Some(v));
        })
    });
}

criterion_group!(benches, bench_ackermann);
criterion_main!(benches);
