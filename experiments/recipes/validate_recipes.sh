#!/bin/bash
set -e

# shellcheck disable=SC2153
if [ -z "$RECIPES" ]; then
  echo "no recipes!"
  exit 1
fi

for RECIPE in $RECIPES; do
  echo "==== checking recipe: $RECIPE ===="
  $VALIDATOR "$RECIPE"
done