#!/usr/bin/env python3

"""
Hopcroft and Tarjan's connected components algorithm based on depth-first search.
https://doi.org/10.1145%2F362248.362272

Note: requires an undirected graph!

Needed to represent the algorithm:
- Iterate vertices of the graph
- Access neighbours of nodes
- Recursion (or a stack), or DFS/BFS primitive
- New vertex label 
"""

test_graph = [
    (0, 1),
    (0, 2),
    (3, 4),
]

# Unrealistic implementation
def nodes(G):
    return set(s for (s, t) in G) | set(t for (s, t) in G)

# Unrealistic implementation
def neighbours(G, n):
    for (s, t) in G:
        if s == n:
            yield t
        elif t == n:
            yield s

def connected_components(G):
    # Maps each vertex to its component number
    components = {}

    component = 0

    def mark_component(node):
        components[node] = component
        for child in neighbours(G, node):
            # In case the graph has a cycle
            if child in components:
                continue
            # Mark all reachable nodes as part of the same component
            mark_component(child)

    for node in nodes(G):
        if node in components:
            # Already seen, skip
            continue
        # Mark this node and all nodes reachable from it as visited
        mark_component(node)
        component += 1
    return components

print(connected_components(test_graph))

