#!/bin/bash
set -e

if [ ! -f "WORKSPACE" ]; then
    echo "Must be run inside the WORKSPACE"
    exit 1
fi

ansible-galaxy install -r ansible/requirements.yml
ansible-playbook ansible/build.yml
bazel build //...
bazel test //...