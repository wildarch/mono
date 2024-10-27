#!/bin/bash
# Run this script within a directory of subtitle files (.en.vtt extension).

set -x

for file in *.en.vtt; do
	echo "Removing align and position from '$file'"
	sed -i 's/align:start position:0%//' "$file"

	echo "Removing word highlight from '$file'"
	sed -i 's/<[^>]*>//g' "$file"
done
