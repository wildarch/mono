#!/usr/bin/env python3

"""
Computes the core numbers of the vertices in the graph.

https://arxiv.org/abs/cs/0310049

I don't really understand this algorithm yet, I have to look into it some more.
It is possible to make it much faster by storing ns in a format that is easier to sort after an update. 
"""
def core(G):
    degrees = { node: degree(G, node) for node in nodes(G) }
    ns = sorted(nodes(G), key=degrees.get)

    for i in range(len(ns)):
        u = ns[i]
        core[u] = degree[u]
        for v in neighbours(G, u):
            if degrees[u] > degrees[v]:
                degrees[u] -= 1
                ns.sort(key=degrees.get)
    return degrees



def core_old(G, k):
    removed = set()
    visited = set()

    # Tracks if we removed any nodes from the graph this iteration
    dirty = True
    def visit(node):
        # Skip nodes that have been removed or already visited
        if node in removed and node in visited:
            return
        visited.add(node)
        ns = set(neighbours(G, node)) - removed
        if len(ns) < k:
            removed.add(node)
            dirty = True
            return
        for n in ns:
            visit(n)

    while dirty:
        dirty = False
        visited.clear()
        for node in nodes(G):
            visit(node)
