#!/bin/bash
set -e

PACKAGE_ROOT=cast/cast-backend_0.0.1-1_arm64

GOARCH=arm64 GOOS=linux go build cast/main.go
mkdir -p $PACKAGE_ROOT/usr/bin
mv main $PACKAGE_ROOT/usr/bin/cast-backend
dpkg --build $PACKAGE_ROOT
