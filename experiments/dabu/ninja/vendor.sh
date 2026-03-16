#!/bin/bash

THIRDPARTY_DIR="experiments/dabu/ninja/thirdparty"
rm -rf $THIRDPARTY_DIR
mkdir -p $THIRDPARTY_DIR
curl -L -o $THIRDPARTY_DIR/ninja-src.tar.gz https://github.com/ninja-build/ninja/archive/refs/tags/v1.13.2.tar.gz
tar -xvf $THIRDPARTY_DIR/ninja-src.tar.gz -C $THIRDPARTY_DIR
