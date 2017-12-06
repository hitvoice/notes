# Vector Space Model

Novelty: use **frequencies** in a corpus of text as a clue for discovering semantic information.

The statistical semantic hypothesis:
> Statistical patterns of human word usage can be used to figure out what people mean.

### Similarity of Documents: The Term-Document Matrix
The bag(multiset) of words hypothesis:
> The frequencies of words in a document tend to indicate the relevance of the document to a query.

In terms of vectors:
> If documents and pseudo-documents (queries) have similar column vectors in a term–document matrix, then they tend to have similar meanings.

### Similarity of Words (Attributional Similarity): The Word-Context Matrix
The distributional hypothesis:
> Words that occur in similar contexts tend to have similar meanings.

In terms of vectors:
> If words have similar row vectors in a word–context matrix, then they tend to have similar meanings.

A word may be represented by a vector in which the elements are derived from the occurrences of the word in various contexts, such as windows of words and grammatical dependencies. See [2] for a comprehensive study of various contexts.
We might not usually think that antonyms are similar, but antonyms have a high degree of attributional similarity.

### Similarity of Relations: The Pair-Pattern Matrix
The extended distributional hypothesis:
> Patterns that co-occur with similar pairs tend to have similar meanings.

In terms of vectors:
> If patterns have similar column vectors in a pair–pattern matrix, then they tend to express similar semantic relations.

For example: "X solves Y" and "Y is solved by X" are similar patterns that co-occur with similar pairs.

The latent relation hypothesis:
> Pairs of words that co-occur in similar patterns tend to have similar semantic relations.

In terms of vectors:
> If word pairs have similar row vectors in a pair–pattern matrix, then they tend to have similar semantic relations.

For example, the pairs mason:stone, carpenter:wood, potter:clay, and glassblower:glass share the semantic relation artisan:material.
Note that relational similarity does not reduce to attributional similarity. For example, we cannot infer that mason:glass and carpenter:clay have similar relations.

### References
[1] Turney, Peter D., and Patrick Pantel. "From frequency to meaning: Vector space models of semantics." Journal of artificial intelligence research 37.1 (2010): 141-188.
[2] Sahlgren, M. (2006). The Word-Space Model: Using distributional analysis to represent syntagmatic and paradigmatic relations between words in high-dimensional vector spaces. Ph.D. thesis, Department of Linguistics, Stockholm University.