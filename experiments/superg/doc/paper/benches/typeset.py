#!/usr/bin/env python3
"""
Parses Criterion benchmark results for superg, and outputs a Latex/pgfplots-friendy format.
"""

import sys
import json

ENGINES = {
    "Miranda": "Miranda (original)",
    "TurnerEngine": "Miranda-style Engine",
    "TigreEngine": "TIGRE-style Engine",
}

BENCHMARKS = {
    "Fibonacci (Turner vs. Miranda)",
    "Fibonacci (Turner vs. TIGRE)",
    "Fibonacci n (All compilers)",
    "Ackermann 3 n (All compilers)",
}

COMPILERS = {
    "Bracket",
    "Kiselyov - Strict",
    "Kiselyov - Lazy",
    "Kiselyov - LazyOpt",
    "Kiselyov - Linear",
}

miranda_coords = []
turner_coords = []

def handle_message(msg):
    if msg["reason"] != "benchmark-complete":
        return
    benchmark, config, val = tuple(msg["id"].split("/"))
    val = float(val)
    score = msg["typical"]["estimate"]

    # Turner v. Miranda
    if benchmark != "Fibonacci (Turner vs. Miranda)":
        return
    if config == "Miranda":
        miranda_coords.append((val, score))
    elif config == "TurnerEngine (Bracket)":
        turner_coords.append((val, score))
    

with open(sys.argv[1]) as f:
    for line in f:
        msg = json.loads(line)
        handle_message(msg)

print("=== Original Miranda vs. Turner ===")
print("""
\\addplot[
    color = blue,
    mark=square,
    ]
    coordinates {""")
for (val, ns) in miranda_coords:
    ms = ns / 1000.0 / 1000.0
    print(f"        ({val}, {ms})")
print("""    };
""")

print("""
\\addplot[
    color = red,
    mark=o,
    ]
    coordinates {""")
for (val, ns) in turner_coords:
    ms = ns / 1000.0 / 1000.0
    print(f"        ({val}, {ms})")
print("""    };
""")

print("\\legend{Miranda(original), Miranda-style Engine}")