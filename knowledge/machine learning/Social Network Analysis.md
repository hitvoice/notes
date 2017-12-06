# Social Network Analysis

Convert any data type to neighborhood graph for applications based on pairwise similarity:
1. For a given set of data objects $\mathcal{O} = \{O_1 \dots O_n\}$, a single node is defined for each object in $\mathcal O$.
2. An edge exists between $O_i$ and $O_j$, if the distance $d(O_i,O_j)$ is less than a particular threshold $\epsilon$, or $O_i$ is the k-nearest neighbor of $O_j$.
3. The weight $w_{ij}$ of the edge $(i,j)$ is equal to a kernelized function of the distance. An example is the _heat kernel_:
$$
w_{ij}=e^{-d(O_i,O_j)^2/t^2}.
$$

### General
**modularity**
Modularity is the fraction of the edges that fall within the given groups minus the expected such fraction if edges were distributed at random. The value of the modularity lies in the range [−1/2,1). It is positive if the number of edges within groups exceeds the number expected on the basis of chance.

**geodesic distance**
The distance between two vertices in a graph is the number of edges in a shortest path (also called a graph geodesic) connecting them. This is also known as the geodesic distance.

**radius**
The radius of a graph is the minimum eccentricity of any vertex.

**diameter**
The diameter of a graph is the maximum eccentricity of any vertex in the graph.

---

### Node Property
**degree**
The number of edges incident to the vertex, with loops counted twice.

**authority & hub**
In Hyperlink-Induced Topic Search(HITS) algorithm, a good hub represented a page that pointed to many other pages, and a good authority represented a page that was linked by many different hubs.

**betweenness centrality**
It is equal to the number of shortest paths from all vertices to all others that pass through that node.

**closeness centrality**
the average of the length of the shortest paths between the node and all other nodes in the graph. Thus the more central a node is, the closer it is to all other nodes.

**eigenvector centrality**
corresponding eigen value in the eigen vector of the adjacency matrix of the graph.

**eccentricity**
The eccentricity of a vertex is the greatest geodesic distance between this vertex and any other vertex. It can be thought of as how far a node is from the node most distant from it in the graph.

**PageRank**
The propbability that a person randomly clicking on links will arrive at any particular page. 

**(local) clustering coefficient**
The proportion of links between the vertices within its neighbourhood divided by the number of links that could possibly exist between them. It quantifies how close its neighbours are to being a clique (complete graph).

---

### Reference
- Book: Data Mining-The Textbook
- <https://en.wikipedia.org/wiki/Degree_(graph_theory)>
- <https://en.wikipedia.org/wiki/Modularity_(networks)>
- <https://en.wikipedia.org/wiki/HITS_algorithm>
- https://en.wikipedia.org/wiki/Betweenness_centrality
- <https://en.wikipedia.org/wiki/Distance_(graph_theory)>
- https://en.wikipedia.org/wiki/PageRank
- https://en.wikipedia.org/wiki/Eigenvector_centrality
- https://en.wikipedia.org/wiki/Clustering_coefficient