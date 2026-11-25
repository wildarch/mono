//! Helpers for golden tests: files of test cases, each an input followed by
//! its expected output.
//!
//! Format: input, a line of 80 '-', expected output, a line of 80 '='.

use crate::ir::Loc;

/// Formats `loc` the same way as the C++ implementation's `operator<<`.
pub fn fmt_loc(filename: &str, loc: &Loc) -> String {
    let mut s = filename.to_string();
    if loc.start.line != 0 {
        if loc.start.line == loc.end.line {
            s += &format!(
                ":{}:{}-{}",
                loc.start.line, loc.start.column, loc.end.column
            );
        } else {
            s += &format!(
                " lines {}-{} characters {}-{}",
                loc.start.line, loc.end.line, loc.start.column, loc.end.column
            );
        }
    }
    s
}

/// Runs every test case in `content` through `run`, panicking if any output
/// differs from the expected output.
pub fn run_golden(name: &str, content: &str, run: impl Fn(&str) -> String) {
    let mid_sep = format!("{}\n", "-".repeat(80));
    let test_sep = format!("{}\n", "=".repeat(80));

    let mut failures = Vec::new();
    let mut count = 0;
    let mut rest = content;
    while let Some(end) = rest.find(&test_sep) {
        // Separators must be at the start of a line.
        assert!(end == 0 || rest.as_bytes()[end - 1] == b'\n');
        let case = &rest[..end];
        rest = &rest[end + test_sep.len()..];

        let mid = case
            .match_indices(&mid_sep)
            .find(|&(i, _)| i == 0 || case.as_bytes()[i - 1] == b'\n')
            .expect("missing input/output separator")
            .0;
        let input = &case[..mid];
        let expected = &case[mid + mid_sep.len()..];
        let actual = run(input);
        count += 1;
        if actual != expected {
            failures.push(format!(
                "input:\n{input}\nexpected:\n{expected}\nactual:\n{actual}"
            ));
        }
    }

    assert!(rest.trim().is_empty(), "trailing content after last test");
    assert!(count > 0);
    assert!(
        failures.is_empty(),
        "{}/{count} {name} tests failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
