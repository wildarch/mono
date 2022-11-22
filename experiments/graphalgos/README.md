# Graph algorithms
For my Master thesis I need to analyse a bunch of graph algorithms.
Under this folder I am implementing a selection of common algorithms to get a feel for the space.

Algorithms I would like to implement:
- [Connected components](https://en.wikipedia.org/wiki/Component_(graph_theory)). 
  A subgraph is a strongly connected component if every vertex is reachable from every other vertex. 
  A directed graph is weakly connected if replacing all directed edges with undirected edges makes it strongly connected.

- [Cycle detection](https://en.wikipedia.org/wiki/Cycle_detection)
- [Triangle counting](https://math.stackexchange.com/questions/117024/complexity-of-counting-the-number-of-triangles-of-a-graph/117030#117030)
- [Cores](https://en.wikipedia.org/wiki/Core_(graph_theory)). This is just a concept, I need to figure out what algorithms are closely related.
- Shortest path
- Pagerank
- [Centrality](https://en.wikipedia.org/wiki/Centrality). There are different definitions of centrality, we can start with degree centrality. Betweenness centrality would be nice as well.
    edit: pagerank is also a centrality measure. Degree centrality is just the degree of each node, trivial.

Wikipedia has a bunch more listed [here](https://en.wikipedia.org/wiki/Category:Graph_algorithms).
[Graph Algorithms book by O'Reilly/Neo4J](https://go.neo4j.com/rs/710-RRC-335/images/Neo4j_Graph_Algorithms.pdf) lists some algorithms that are commonly used with graph databases, but only gives code for use with Neo4J.


Rules for implementations:
- No in-place mutation of the input graph
- All graphs are directed.
- We can iterate over nodes in the graph. Nodes have a unique integer ID.
- We can iterate over all edges in the graph, which are pairs of node IDs. The list is not ordered.
- We can efficiently iterate over the out and in neighbors of a graph.
- Nodes and edges can both have arbitrary associated properties.

## Notes
A graph homomorphism is a mapping from the vertex set of one graph to the vertex set of another graph that maps adjacent vertices to adjacent vertices.
Two vertices are adjacent if there exists an edge between them, and they are not the same vertex.
A graph colouring is the same as a homomorphism to a complete (fully connected) graph.

A graph isomorphism is a one-to-one mapping between vertices and edges of two graphs.
If two graphs are isomorphic, they are structurally equivalent.

An induced subgraph is one formed from a subset of the edges of the original graph, but with all of the edges preserved.
For endpoints of edges where the original vertex is no longer present, another vertex in the subset takes its place.
It is a 'simplified' version of the original graph, in the sense that it has fewer vertices but the same number of edges.

The core `C` of graph `G` is the smallest subgraph such that there exists a homomorphism from `C` to `G` and back. 
A simple example starts with graph `G`:

```
A ---- B
  \
   \
    C
```

The core of this graph is: 

```
A ---- B
```

`B` and `C` are not adjacent in `G`, so in the simplified core, we can merge `B` and `C`.
Alternatively, we can give the homomorphisms between them:

```
G -> C =
  A -> A
  B -> B
  C -> B

C -> G =
  A -> A
  B -> B
```

A k-core sounds similar but is a different thing entirely.
Starting from graph `G`, the k-core is formed by repeatedly deleting vertices with a degree lower than `k`.
If the deletion of a vertex reduces the degree of a node below `k`, that node should be removed as well, even if in the original graph it had a higher degree.