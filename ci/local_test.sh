#!/bin/bash
set -e

docker build -f ci/Dockerfile -t mono_ci .
docker run --rm mono_ci