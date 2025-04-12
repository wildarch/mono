#!/bin/bash
set -e

docker build -t ghcr.io/wildarch/mono:ci -f docker/ci.Dockerfile docker/
docker build -t ghcr.io/wildarch/mono:devcontainer -f docker/devcontainer.Dockerfile docker/
