use criterion::{criterion_group, Criterion};
use regex::Regex;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

criterion_group!(benches, bench_fib_20);

pub fn bench_fib_20(c: &mut Criterion) {
    bench_fib(20, c)
}

fn bench_fib(n: u16, c: &mut Criterion) {
    bench_miranda(
        c,
        &format!("miranda_fib_{n}"),
        r#"
fib n = n,                              if n <= 2
      = fib(n - 1) + fib(n - 2),        otherwise
        "#,
        &format!("fib {n}"),
    );
}

fn bench_miranda(c: &mut Criterion, test_name: &str, module_file_content: &str, command: &str) {
    // Create the temp file with the module contents
    let mut temp_file = tempfile::Builder::new().suffix(".m").tempfile().unwrap();
    write!(temp_file, "{}", module_file_content).unwrap();
    temp_file.flush().unwrap();
    c.bench_function(test_name, |b| {
        b.iter_custom(|iters| {
            let mut elapsed = Duration::ZERO;

            for _ in 0..iters {
                elapsed += run_miranda(temp_file.path(), command).runtime;
            }
            elapsed
        })
    });
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Stats {
    runtime: Duration,
    cycles: usize,
}

fn run_miranda(file_path: &Path, command: &str) -> Stats {
    let miranda_path = std::env::var("MIRANDA_PATH").expect("'MIRANDA_PATH' must be set");
    let mut mira = Command::new("./mira")
        .arg(file_path)
        .current_dir(miranda_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("error starting miranda");
    {
        let mut stdin = mira.stdin.take().unwrap();
        writeln!(stdin, "{}", command).unwrap();
        // stdin dropped and closed here
    }

    let output = mira.wait_with_output().expect("error running miranda");

    let stderr = String::from_utf8(output.stderr).unwrap();

    let stats_regex = Regex::new("Nanos: (\\d+)\nCycles: (\\d+)\n").unwrap();
    let caps = match stats_regex.captures(&stderr) {
        Some(caps) => caps,
        None => panic!("stats not found, stderr: {}", stderr),
    };
    let nanos: u64 = caps.get(1).unwrap().as_str().parse().unwrap();
    let cycles = caps.get(2).unwrap().as_str().parse().unwrap();
    Stats {
        runtime: Duration::from_nanos(nanos),
        cycles,
    }
}
