#!/usr/bin/env python3
import sys
import gzip
from lxml import etree

with gzip.open(sys.argv[1], 'rb') as gzf:
    tag_stack = []

    record_types = set()
    field_types = set()
    attr_types = set()
    unhandled = set()

    multi_fields = {}

    # Fields observed in the current record
    record_fields = set()
    def start(elem):
        tag_stack.append(elem.tag)

        if len(tag_stack) == 2:
            # level 2 is a record type
            # (level 1 is the DBLP elem)
            record_types.add(elem.tag)
        if len(tag_stack) == 3:
            field_types.add(elem.tag)

            if elem.tag in record_fields:
                multi_fields[elem.tag] = 1 + multi_fields.get(elem.tag, 0)
            else:
                record_fields.add(elem.tag)
        if len(tag_stack) > 3:
            if tag_stack[2] == "title" and ("sub", "sup", "i"):
                # Allow sub, sup, i (nested) inside title
                pass
            else:
                path = f"{tag_stack}"
                if path not in unhandled:
                    unhandled.add(path)
                    print(path)

    def end(elem):
        if len(tag_stack) == 2:
            # record type
            for key,val in elem.items():
                attr_types.add(key)
            record_fields.clear()
        else:
            for key,val in elem.items():
                path = f"{tag_stack} {key}"
                if path not in unhandled:
                    unhandled.add(path)
                    print(path)

        elem.clear()
        tag_stack.pop()
    for event, elem in etree.iterparse(gzf, events=("start", "end",), load_dtd=True):
        if event == "start":
            start(elem)
        elif event == "end":
            end(elem)
        else:
            raise RuntimeError(f"illegal event: {event}")
    
    print("Record types: ", record_types)
    print("Field types: ", field_types)
    print("Attr types: ", attr_types)
    print("Field that may repeat within records: ", multi_fields)
