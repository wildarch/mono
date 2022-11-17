#!/usr/bin/env python3

"""
Local triangle counting. 

Many databases tend to estimate it rather than getting an absolute answer.
Neo4J uses this one: https://doi.org/10.1145/1401890.1401898
Another one that is more recent is: https://arxiv.org/abs/1011.0468

If we want an absolute answer, we have to look for cycles of length 3.

Requires an undirected graph!
"""

"""
This is the slow but precise algorithm.
"""
def triangle_count(G):
    # Number of triangles per vertex
    triangles = {}

    # Look for a path node -> l1 -> l2 -> node
    # O(V^4)
    for node in nodes(G):
        # O(V^3)
        for l1 in neighbours(G, node):
            if l1 in [node]:
                continue
            # O(V^2)
            for l2 in neighbours(G, l1):
                if l2 in [node, l1]:
                    continue
                # O(V)
                for l3 in neighbours(G, l2):
                    if l3 == node:
                        triangles[node] += 1
