# Word Association Mining

### Types
* Paradigmatic: A & B have paradigmatic relation if they can be substituted for each other
* Syntagmatic: A & B have syntagmatic relation if they can be combined with each other

Syntagmatic associates tend to be semantically associated (bee and honey are often neighbours); paradigmatic parallels tend to be taxonomically similar (doctor and nurse have similar neighbours).

### Applications
* Text retrieval (e.g., use word associations to suggest a variation of a query)
* Automatic construction of topic map for browsing: words as nodes and associations as edges
* Compare and summarize opinions (e.g., what words are most strongly associated with “battery” in positive and negative reviews about iPhone 6, respectively?)

### Algorithm: Paradigmatic Relation Mining
* Represent each word by its context (e.g. concat words in left_1, right_1, neighbors\[4\]... to generate psudo documents)
* Compute context [similarity](Text%20Similarity.md) (e.g. sum of similarity of corresponding pseudo documents)
* Words with high context similarity are likely to have paradigmatic relations

### Algorithm: Syntagmatic Relation Mining
* by computing [mutual information](../math/Mutual%20Information.md)

### Additional Readings
* Identify lexical atoms like "hot dog":
Chengxiang Zhai, Exploiting context to identify lexical atoms: A statistical view of linguistic context. Proceedings of the International and Interdisciplinary Conference on Modelling and Using Context (CONTEXT-97), Rio de Janeiro, Brzil, Feb. 4-6, 1997. pp. 119-129.

* Use word graphs to find paradigmatic and syntagmatic relations
Shan Jiang and ChengXiang Zhai, Random walks on adjacency graphs for mining lexical relations from big text data. Proceedings of IEEE BigData Conference 2014, pp. 549-554.

### Reference
- Text Mining: [https://www.coursera.org/learn/text-mining](https://www.coursera.org/learn/text-mining)
- Turney, Peter D., and Patrick Pantel. "From frequency to meaning: Vector space models of semantics." Journal of artificial intelligence research 37.1 (2010): 141-188.