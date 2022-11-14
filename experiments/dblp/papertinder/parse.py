#!/usr/bin/env python3
import sys
from lxml import etree
import gzip
import sqlite3
import json

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

PUBLICATION_ELEMENTS = [
    "article",
    "inproceedings",
]

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
    cur.execute("""
        CREATE TABLE dblp(
            key TEXT PRIMARY KEY,
            -- Publication type. 
            -- * 'article' for a journal article
            -- * 'inproceedings' for a conference paper
            pub_type TEXT NOT NULL,
            -- Title of the publication.
            -- In DBLP titles can have style elements like <it>,
            -- but in parsing we remove those and use flat text only.
            title TEXT NOT NULL,
            -- Stored as a JSON array. 
            authors TEXT NOT NULL,
            -- Optional but usually present.
            year INT,
            -- Link to the authoritative source.
            -- Often a DOI.
            ee TEXT
        );
    """)
    rows = []
    def commit_rows():
        if len(rows) > 0:
            cur.executemany("INSERT INTO dblp VALUES(?,?,?,?,?,?)", rows)
            conn.commit()
            rows.clear()

    with gzip.open(sys.argv[1], 'rb') as gzf:
        # load_dtd looks for the DTD by itself, on the local file system.
        for event, elem in etree.iterparse(gzf, events=("end",), load_dtd=True):
            if event == "end":
                if elem.tag in DBLP_ELEMENTS:
                    if elem.tag in PUBLICATION_ELEMENTS:
                        key = elem.attrib["key"]
                        pub_type = elem.tag
                        # Child elements
                        title = None
                        authors = []
                        year = None
                        ee = None
                        for child in elem:
                            if child.tag == "title":
                                title = flatten_text(child)
                            elif child.tag == "author":
                                authors.append(flatten_text(child))
                            elif child.tag == "year":
                                year = flatten_text(child)
                            elif child.tag == "ee":
                                ee = flatten_text(child)
                        
                        if title is None:
                            print(f"No title: {etree.tostring(elem)}")
                        assert(title is not None)
                        rows.append((key, pub_type, title, json.dumps(authors), year, ee))
                        if len(rows) > 10_000:
                            commit_rows()

                        count += 1
                        if count % 100_000 == 0:
                            print(f"Processed {count} elements")

                    # Only clear the element and its children once we have parsed a full entry
                    elem.clear()
        commit_rows()
    with sqlite3.connect('/tmp/dblp.sqlite3') as fconn:
        conn.backup(fconn)
print(f"Processed {count} rows")
