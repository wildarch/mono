#!/bin/bash
set -e

python parse_to_ir.py bug.c -o bug.c.json
python ir_to_c.py bug.c.json -o bug_roundtrip.c
clang-format-21 -i bug.c
clang-format-21 -i bug_roundtrip.c
vimdiff bug.c bug_roundtrip.c