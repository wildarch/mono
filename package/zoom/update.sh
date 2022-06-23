#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: update.sh WORKSPACE_DIR"
    exit 1
fi
WORKSPACE_DIR="$1"

if [ ! -f "$WORKSPACE_DIR/WORKSPACE" ]; then
    echo "No workspace file at $WORKSPACE_DIR/WORKSPACE"
    exit 1
fi

ZOOM_LATEST_URL="https://zoom.us/client/latest/zoom_amd64.deb"
ZOOM_VERSION_URL=$(curl -ILs -o /dev/null -w "%{url_effective}" $ZOOM_LATEST_URL)

ZOOM_SHA256=$(curl -s "$ZOOM_VERSION_URL" | sha256sum | awk '{ print $1 }')

cat <<EOF | tee "$WORKSPACE_DIR/package/zoom/zoom.bzl"
ZOOM_URL="$ZOOM_VERSION_URL"
ZOOM_SHA256="$ZOOM_SHA256"
EOF