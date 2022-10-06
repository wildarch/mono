# Graph algorithms
For my Master thesis I need to analyse a bunch of graph algorithms.
Under this folder I am implementing a selection of common algorithms to get a feel for the space.

Algorithms I would like to implement:
- [Connected components](https://en.wikipedia.org/wiki/Component_(graph_theory)). 
  A subgraph is a strongly connected component if every vertex is reachable from every other vertex. 
  A directed graph is weakly connected if replacing all directed edges with undirected edges makes it strongly connected.

- [Cycle detection](https://en.wikipedia.org/wiki/Cycle_detection)
- [Triangle counting](https://math.stackexchange.com/questions/117024/complexity-of-counting-the-number-of-triangles-of-a-graph/117030#117030)
- [Cores](https://en.wikipedia.org/wiki/Core_(graph_theory)). This is really just a concept, I need to figure out what algorithms are closely related.
- Shortest path
- Pagerank
- [Centrality](https://en.wikipedia.org/wiki/Centrality). There are different definitions of centrality, we can start with degree centrality. Betweenness centrality would be nice as well.

Wikipedia has a bunch more listed [here](https://en.wikipedia.org/wiki/Category:Graph_algorithms).
[Graph Algorithms book by O'Reilly/Neo4J](https://go.neo4j.com/rs/710-RRC-335/images/Neo4j_Graph_Algorithms.pdf) lists some algorithms that are commonly used with graph databases, but only gives code for use with Neo4J.