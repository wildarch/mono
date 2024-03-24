#!/usr/bin/env python3
from collections import defaultdict
import csv
import gzip
import json
import sys

def parse_characters(c):
    return json.loads(c)

characters_per_name = defaultdict(set)

with gzip.open(sys.argv[1], 'rt') as f:
    # Parse lines to a dictionary
    tsv_reader = csv.DictReader(f, delimiter='\t', quotechar='"')
    for i, line in enumerate(tsv_reader):
        # Progress update
        if i % 10_000 == 0:
            print(f"Processed {i} lines")

        # Grab the columns we care about
        name = line['nconst']
        characters = line['characters']
        if characters == "\\N":
            # Null
            continue

        # Add characters to the set.
        add_to = characters_per_name[name]
        for character in json.loads(characters):
            add_to.add(character)

# Find the name with the most associated characters
max_char = 0
max_name = None

for name in characters_per_name:
    characters = characters_per_name[name]
    if len(characters) > max_char:
        max_char = len(characters)
        max_name = name

print(f"Final answer: {max_name} ({max_char} characters)")
"""
characters = characters_per_name[max_name]
for c in characters:
    print(c)
"""