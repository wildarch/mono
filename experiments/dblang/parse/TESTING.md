# Testing the dblang parser
This document describes how to build and run the unit tests for the parser
components in `experiments/dblang/parse`, and how to author `.test` files.

There are currently two test executables:

| Executable | Source | Test data file |
| ---------- | ------ | -------------- |
| `LexerTest`   | `parse/LexerTest.cpp`   | `parse/Lexer.test`   |
| `ChunkerTest` | `parse/ChunkerTest.cpp` | `parse/Chunker.test` |

Both follow the same pattern: they read a `.test` file containing a sequence of
input/output pairs, run the component under test on each input, and compare the
actual output against the expected output.

## Building
To build the test targets:

```bash
cmake --build experiments/dblang/build --target LexerTest
cmake --build experiments/dblang/build --target ChunkerTest
```

The test executables are written to `experiments/dblang/build/`.

## Running the tests
Run each executable from the workspace root:

```bash
experiments/dblang/build/LexerTest
experiments/dblang/build/ChunkerTest
```

Each prints a summary line, e.g.:

```
running 8 tests
8/8 pass
```

If any test fails, the harness prints the failure count and writes a *candidate*
test file containing the actual (current) output for every test.
It then prints instructions for comparing and accepting the new output:

```
wrote candidate test file to /tmp/Lexer.test
    to compare, run:
vimdiff experiments/dblang/parse/Lexer.test /tmp/Lexer.test
    to accept the current diff:
cp /tmp/Lexer.test experiments/dblang/parse/Lexer.test
```

This workflow is the intended way to update expected output after a deliberate
change to the lexer or chunker: inspect the diff with `vimdiff` (or `diff`), and if the new
output is correct, copy the candidate file over the `.test` file.

## `.test` file format

A `.test` file is a plain text file containing a sequence of test cases. Each
test case has two blocks separated by separator lines:

```
<input block>
--------------------------------------------------------------------------------
<expected output block>
================================================================================
```

- The **input block** is the raw source text passed to the component under test
  (the lexer or chunker). It is used verbatim, including whitespace and
  newlines.
- The **expected output block** is the exact text the component is expected to
  produce for that input.
- The separator lines are **exactly 80 characters** of `-` (input/output
  separator) and `=` (test separator). The harness matches these lines by
  comparing the whole line against an 80-char string, so the separators must
  not be shortened or lengthened.

The harness in each `*Test.cpp` scans the file line by line and
tracks a two-state machine:

1. **Input state** — accumulates lines until it sees an 80-char `---` line.
2. **Expected state** — accumulates lines until it sees an 80-char `===` line,
   at which point the input/output pair is recorded and the state resets to
   Input.

The first test case starts at the beginning of the file; there is no leading
separator. The file must end in the Input state (i.e. the last test must be
terminated by a `===` line).

### Lexer output format

For each token, `LexerTest` prints one line:

```
<loc>: <TOKEN_KIND> '<body>'
```

where `<body>` is the token's source text. For example:

```
experiments/dblang/parse/Lexer.test:1:1-4: INT_KW 'int'
```

Comments and whitespace produce no tokens, so an input consisting only of a
comment yields an empty expected block.

### Chunker output format

For each chunk, `ChunkerTest` prints the chunk location followed by the chunk
text:

```
<loc>:
<chunk text>
```

For example:

```
experiments/dblang/parse/Chunker.test:9:1-36:
def MyAlias = alias { <type expr> }
```

### Location format

Both formats embed a `Loc`, rendered by `operator<<` in
`parse/Location.cpp`. There are two forms:

- **Single-line** (start and end on the same line):

  ```
  <filename>:<line>:<start-col>-<end-col>
  ```

  e.g. `experiments/dblang/parse/Lexer.test:2:1-14`

- **Multi-line** (start and end on different lines):

  ```
  <filename> lines <start-line>-<end-line> characters <start-col>-<end-col>
  ```

  e.g. `experiments/dblang/parse/Chunker.test lines 1-4 characters 1-2`

Columns are 1-based; the end column is exclusive (one past the last character).

## Adding a test
1. Open the relevant `.test` file (`Lexer.test` or `Chunker.test`).
2. Append a new test case: the input block, an 80-char `---` line, dummy text (TODO) for the expected output block, and an 80-char `===` line.
3. Rebuild and run the test executable.
4. Check the proposed output in the candidate files. If it corrrect, copy it over to the `.test` file.

## Notes for agents

- The `.test` files are the source of truth for expected behavior. When you
  change the lexer or chunker, run the tests and update the expected output via
  the propose/diff/copy workflow rather than hand-editing expected blocks.
- The input block is passed verbatim to the component. Watch for trailing
  newlines: the input block includes everything up to the `---` separator.
- The expected output must match **exactly**, including trailing newlines and
  the blank line the chunker emits after each chunk.
- The separator lines must remain exactly 80 characters; the harness compares
  them as whole lines.