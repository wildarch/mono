#!/bin/bash
set -e

if [ ! -f "ansible/build.yml" ]; then
    echo "Must be run from the repository root"
    exit 1
fi

ansible-galaxy install -r ansible/requirements.yml
ansible-playbook ansible/build.yml
bazel build //...
bazel test //...