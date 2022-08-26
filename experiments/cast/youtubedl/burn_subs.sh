#!/bin/bash

set -x

for file in *.{mp4,mkv}; do
	output_file="../${file%.*}.mp4"
	subs_file="${file%.*}.en.vtt"
	if [[ -f "$output_file" ]]; then
		echo "Found $output_file, skipping"
	else
		cp "$subs_file" /tmp/subs.vtt
		ffmpeg -i "$file" -pattern_type none -vf subtitles=/tmp/subs.vtt "$output_file" 
	fi
done
