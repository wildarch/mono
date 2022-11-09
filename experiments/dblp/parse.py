#!/usr/bin/env python3
import sys
from lxml import etree
import gzip
import sqlite3
import re

DBLP_ELEMENTS = [
    "article",
    "inproceedings",
    "proceedings",
    "book",
    "incollection",
    "phdthesis",
    "mastersthesis",
    "www",
    "person",
    "data",
]

# Cached from a previous run of get_keys
KEYS = [
    'element',
    'isbn', 
    'title', 
    'mdate', 
    'publisher', 
    'publtype', 
    'url', 
    'number', 
    'cite', 
    'publnr', 
    'key', 
    'note', 
    'series', 
    'booktitle', 
    'address', 
    'school', 
    'crossref', 
    'editor', 
    'pages', 
    'cdrom', 
    'ee', 
    'year', 
    'month', 
    'volume', 
    'journal', 
    'author', 
    'cdate', 
    'chapter',
]

def get_keys():
    keys = set()
    count = 0
    with gzip.open(sys.argv[1], 'rb') as gzf:
        # load_dtd looks for the DTD by itself, on the local file system.
        for event, elem in etree.iterparse(gzf, events=("end",), load_dtd=True):
            if event == "end":
                if elem.tag in DBLP_ELEMENTS:
                    for key, value in elem.items():
                        keys.add(key)
                    for child in elem:
                        keys.add(child.tag)
                        for key, value in child.items():
                            keys.add(f"{child.tag}_{key}")
                    count += 1
                elem.clear()
    print(keys)

def to_tuple(row):
    return tuple([row.get(k) for k in KEYS])

def flatten_text(node):
    if len(node) == 0:
        return node.text
    else:
        text = ""
        if node.text:
            text += node.text
        for child in node:
            ct = flatten_text(child)
            if ct is not None:
                text += ct
        if node.tail:
            text += node.tail
        
        if text == "":
            return None
        else:
            return text


count = 0
with sqlite3.connect(':memory:') as conn:
    cur = conn.cursor()
    cur.execute(f"CREATE TABLE dblp({', '.join(KEYS)})")
    rows = []
    with gzip.open(sys.argv[1], 'rb') as gzf:
        # load_dtd looks for the DTD by itself, on the local file system.
        for event, elem in etree.iterparse(gzf, events=("end",), load_dtd=True):
            if event == "end":
                if elem.tag in DBLP_ELEMENTS:
                    row = {'element': elem.tag}
                    for key, value in elem.items():
                        row[key] = value
                    for child in elem:
                        row[child.tag] = flatten_text(child)

                    rows.append(to_tuple(row))
                    if len(rows) >= 10_000:
                        # Insert to sqlite
                        cur.executemany(f"INSERT INTO dblp VALUES({','.join(['?' for k in KEYS])})", rows)
                        conn.commit()
                        rows.clear()

                    count += 1
                    # Limit
                    # if count >= 100_000:
                    #     break 

                    # Only clear the element and its children once we have parsed a full entry
                    elem.clear()
    with sqlite3.connect('/tmp/dblp.sqlite3') as fconn:
        conn.backup(fconn)
print(f"Processed {count} rows")
