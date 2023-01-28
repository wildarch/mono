#!/usr/bin/env python3
"""
Parses Criterion benchmark results for superg, and outputs a Latex/pgfplots-friendy format.
"""

import sys
import json

BASE_DIR="experiments/superg/doc/paper/benches"

COMPILERS = {
    "Bracket": "Bracket Abstraction",
    "Kiselyov - Strict": "Kiselyov - $\\strict$",
    "Kiselyov - Lazy": "Kiselyov - $\\lazy$",
    "Kiselyov - LazyOpt": "Kiselyov - $\\lazyeta$",
    "Kiselyov - Linear": "Kiselyov - $\\linear$",
}

miranda_coords = []
turner_coords = []
tigre_coords = []

compiler_coords = {c: [] for c in COMPILERS.values()}

def handle_message(msg):
    if msg["reason"] != "benchmark-complete":
        return
    benchmark, config, val = tuple(msg["id"].split("/"))
    val = float(val)
    score = msg["typical"]["estimate"]

    # Turner v. Miranda
    if benchmark == "Fibonacci (Turner vs. Miranda)":
        if config == "Miranda":
            miranda_coords.append((val, score))
        elif config == "TurnerEngine (Bracket)":
            turner_coords.append((val, score))
    # Turner v. TIGRE
    if benchmark == "Fibonacci (Turner vs. TIGRE)":
        if config == "TigreEngine (Bracket)":
            tigre_coords.append((val, score))
        # Turner results are the same as Turner v. Miranda
    
    if benchmark == "Fibonacci n (All compilers)":
        compiler = COMPILERS[config]
        assert(compiler is not None)

        compiler_coords[compiler].append((val, score))
    

with open(BASE_DIR + "/output.json") as f:
    for line in f:
        msg = json.loads(line)
        handle_message(msg)

with open(BASE_DIR + "/miranda_vs_turner.tex", "w") as f:
    f.write("""
    \\addplot[
        color=blue,
        mark=*,
    ]
        coordinates {""")
    for (val, ns) in turner_coords:
        ms = ns / 1000.0 / 1000.0
        f.write(f"        ({val}, {ms})")
    f.write("""    };
    """)

    f.write("""
    \\addplot
        coordinates {""")
    for (val, ns) in miranda_coords:
        ms = ns / 1000.0 / 1000.0
        f.write(f"        ({val}, {ms})")
    f.write("""    };
    """)


    f.write("\\legend{Miranda(original), Miranda-style Engine}")

with open(BASE_DIR + "/turner_vs_tigre.tex", "w") as f:
    f.write("""
    \\addplot[
        color=blue,
        mark=*,
    ]
        coordinates {""")
    for (val, ns) in turner_coords:
        ms = ns / 1000.0 / 1000.0
        f.write(f"        ({val}, {ms})")
    f.write("""    };
    """)

    f.write("""
    \\addplot
        coordinates {""")
    for (val, ns) in tigre_coords:
        ms = ns / 1000.0 / 1000.0
        f.write(f"        ({val}, {ms})")
    f.write("""    };
    """)

    f.write("\\legend{Miranda-style Engine, TIGRE-style Engine}")

with open(BASE_DIR + "/bracket_vs_kiselyov.tex", "w") as f:
    for c in COMPILERS.values():
        f.write("""
        \\addplot
            coordinates {""")
        for (val, ns) in compiler_coords[c]:
            ms = ns / 1000.0 / 1000.0
            f.write(f"        ({val}, {ms})")
        f.write("""    };
        """)


    legend = ",".join(COMPILERS.values())
    f.write(f"\\legend{{{legend}}}")