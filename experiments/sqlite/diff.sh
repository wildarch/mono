#!/bin/bash
set -e

python parse_to_ir.py -o sqlite3.c.json
python ir_to_c.py sqlite3.c.json -o sqlite3_roundtrip.c
clang-format-21 -i sqlite3_preprocessed.c
clang-format-21 -i sqlite3_roundtrip.c
vimdiff sqlite3_preprocessed.c sqlite3_roundtrip.c