#!/usr/bin/env python3

"""
Single source shortest path.

Common options here are:
- Dijkstra
- A* ()
- Bellman-ford (for graphs with negative edge weights)
- BFS (if graph is unweighted).

Here is a list of variants that Neo4J supports: https://neo4j.com/docs/graph-data-science/current/algorithms/pathfinding/

Requirements:
- Dijkstra and derivatives use a priority heap (min-heap).
- BFS uses a FIFO queue.
"""

# TODO