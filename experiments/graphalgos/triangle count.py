#!/usr/bin/env python3

"""
Local triangle counting. 

Many databases tend to estimate it rather than getting an absolute answer.
Neo4J uses this one: https://doi.org/10.1145/1401890.1401898
Another one that is more recent is: https://arxiv.org/abs/1011.0468

If we want an absolute answer, we have to look for cycles of length 3.
Algorithms for this are based on 'CS167: Reading in Algorithms Counting Triangles' by Tim Roughgarden.

Requires an undirected graph!

To implement all these algorithms, we need:
- temporary hashmaps
- neighbours, nodes, edges accessors
- degree of a node
- edge existance check
- (possibly a builtin distinct_pairs)
- random number generation
"""

"""
This is a slow but precise algorithm.
"""
def triangle_count(G):
    # Number of triangles per vertex
    triangles = {}

    # Look for a path node -> l1 -> l2 -> node
    # O(V * deg(V^2))
    for node in nodes(G):
        # O(deg(V)^3)
        for l1 in neighbours(G, node):
            if l1 in [node]:
                continue
            # O(deg(V)^2)
            for l2 in neighbours(G, l1):
                if l2 in [node, l1]:
                    continue
                # O(deg(V))
                for l3 in neighbours(G, l2):
                    if l3 == node:
                        triangles[node] += 1
    return triangles

"""
Another slow precise algorithm.
It improves on the previous algorithm in a few ways:
- It skips any vertex with an out-degree less than 2.
- We only iterate over one set of neighbours per vertex.
"""
def triangle_count2(G):
    # Number of triangles per vertex
    triangles = {}

    # O(V * deg(V)^2)
    for node in nodes(G):
        # O(deg(node)^2)
        for c1,c2 in distinct_pairs(neighbours(G, node)):
            # Self-loops
            if c1 == node or c2 == node:
                continue
            # Assume this lookup is in constant time
            if has_edge(G, c1, c2):
                triangles[node] += 1
    return triangles

"""
A slightly different algorithm that only counts the total number of triangles in the graph
It is particularly good in very unbalanced graphs: if you give it a star graph with one center node that connects to all other nodes, 
but other nodes is not connected to each other, it will very quickly figure out that there are 0 triangles.
The previous algorithm on the other hand will run many edge checks between neighbours of the center node.
"""
def triangle_count3(G):
    triangles = 0

    for node in nodes(G):
        node_degree = degree(G, node)
        for c1,c2 in distinct_pairs(neighbours(G, node)):
            # Count only from the vertex with the lowest degree
            c1_degree = degree(G, c1)
            c2_degree = degree(G, c2)
            if c1_degree > node_degree or c2_degree > node_degree:
                continue
            # Tie break on node id
            if c1_degree == node_degree and c1 > node:
                continue
            if c2_degree == node_degree and c2 > node:
                continue
            # Self-loops
            if c1 == node or c2 == node:
                continue
            # Assume this lookup is in constant time
            if has_edge(G, c1, c2):
                triangles += 1
    return triangles

"""
https://doi.org/10.1145/1401890.1401898

The key insight here is that for every edge (u, v), the number of triangles containing this edge can be computed with |S(u) ∩ S(v)|,
where S returns the set of neighbours. 
If we have (u, v), we need to find a w such that there exists a (u, w) and (v, w) to finish the cycle.
(u, w) exists iff w ∈ S(u) and , (v, w) exists iff w ∈ S(v).

Consider one iteration of the algorithm:
We start with vertices u, v, w in a cycle.
Assume the hash of w is the lowest among them.
Then after the minima compute step, we have min_h[u] = hashes[w] and min_h[v] = hashes[w], and we increment Z[(src, dst)].
What remains is just some math jargon to extract a good estimate for the count.
"""
def triangle_count_approx(G, iterations):
    Z = {}

    for i in range(iterations):
        hashes = {}
        min_h = {}
        # Initialize
        for node in nodes(G):
            hashes[node] = random_int()
            min_h[node] = MAX_INT
        # Compute minima
        for src, dst in edges(G):
            min_h[src] = min(min_h[src], hashes[dst])
        
        # Compare minima
        for src, dst in edges(G):
            if min_h[src] == min_h[dst]:
                Z[(src, dst)] = Z.get((src, dst), 0) + 1
    # Compute number of triangles
    T = {}
    for src, dst in edges(G):
        T[src] = T.get(src, 0) + (Z.get((src, dst), 0) / (Z.get((src, dst), 0)+iterations)) * (len(neighbours(G, src)) + len(neighbours(G, dst)))
    
    return {k: v / 2 for k, v in T}

# TODO the main-memory version of the above algorithm, from the same paper