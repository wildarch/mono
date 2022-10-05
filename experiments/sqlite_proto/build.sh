#!/bin/bash
ROOT_DIR=experiments/sqlite_proto

g++ -g -fPIC -shared $ROOT_DIR/compress.cpp -o $ROOT_DIR/compress.so
