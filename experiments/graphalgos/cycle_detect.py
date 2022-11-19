#!/usr/bin/env python3

"""
Classic Cycle detection using DFS.

Requires:
- set and stack data structures
"""

def cycle_detect(G):
    visited = set()
    stack = list() 

    cycles = list()

    def visit(n):
        if n in stack:
            cycles.append(stack[stack.index(n):] + [n])
            return
        if n in visited:
            return
        visited.add(n)
        stack.append(n)
        for c in neighbours(G, n):
            visit(c)
        stack.pop()
    for n in nodes(G):
        stack = list()
        visit(n)
    
    return cycles

test_graph = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),

    (1, 4),
    (4, 1),
]

# Unrealistic implementation
def nodes(G):
    return set(s for (s, t) in G) | set(t for (s, t) in G)

# Unrealistic implementation
def neighbours(G, n):
    for (s, t) in G:
        if s == n:
            yield t

print(cycle_detect(test_graph))