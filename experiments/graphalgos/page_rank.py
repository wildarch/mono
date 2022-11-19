#!/usr/bin/env python3
"""
Pagerank algorithm.
It is easiest to view as an operation on matrices.

Mapping it to a sparse matrix is quite simple, if we leave out the damping factor
"""

import numpy as np

def pagerank(M, num_iterations: int = 100, d: float = 0.85):
    """
    This is taken verbatim from https://en.wikipedia.org/wiki/PageRank#Python
    
    PageRank: The trillion dollar algorithm.

    Parameters
    ----------
    M : numpy array
        adjacency matrix where M_i,j represents the link from 'j' to 'i', such that for all 'j'
        sum(i, M_i,j) = 1
    num_iterations : int, optional
        number of iterations, by default 100
    d : float, optional
        damping factor, by default 0.85

    Returns
    -------
    numpy array
        a vector of ranks such that v_i is the i-th rank from [0, 1],
        v sums to 1

    """
    N = M.shape[1]
    v = np.ones(N) / N
    # Apply damping
    M_hat = (d * M + (1 - d) / N)
    for i in range(num_iterations):
        # Matrix multiplication
        v = M_hat @ v
    return v

M = np.array([[0, 0, 0, 0, 1],
              [0.5, 0, 0, 0, 0],
              [0.5, 0, 0, 0, 0],
              [0, 1, 0.5, 0, 0],
              [0, 0, 0.5, 1, 0]])
v = pagerank(M, 100, d = 1)

print(v)

"""
Same thing, but with a sparse graph
"""
def pagerank_graph(G, num_iterations = 100):
    N = len(nodes(G))
    v = [1.0/N for i in range(N)]

    for i in range(num_iterations):
        nv = [0.0 for i in range(N)]

        for s, t, w in G:
            # This does not work because we also need to add edges with weight 
            # (1-d)/N for any disconnected pair of nodes in the original graph
            # dw = d * w + (1 - d) / N
            # nv[t] += dw * v[s]

            # No support for damping factor
            nv[t] += w * v[s]
        v = nv
    
    return v

def nodes(G):
    return set(s for s, t, w in G) | set(t for s, t, w in G)

test_graph = [
    (0, 1, 0.5),
    (0, 2, 0.5),

    (1, 3, 1.0),

    (2, 3, 0.5),
    (2, 4, 0.5),

    (3, 4, 1.0),

    (4, 0, 1.0),
]

v = pagerank_graph(test_graph, 100)

print(v)