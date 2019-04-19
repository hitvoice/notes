## Basics
A word embedding is a continuous vector representation of words.

Ways to train word embeddings:
- Initialize randomly, train jointly with the task 
- Pre-train on a supervised task (e.g. POS tagging) and test on another, (e.g. parsing)
- Pre-train on an unsupervised task

Distributional vs non-distributional representations:
- Words are similar if they appear in similar contexts; distribution of words indicative of usage 
- non-distributional representations created from lexical resources such as WordNet, etc.

Distributed vs local representations:
- Distributed representations are represented by a vector of values
- local representations are represented by a discrete symbol (one-hot vector)

## GloVe
GloVe is a matrix factorization approach motivated by ratios of $P(\text{word} | \text{context})$ probabilities.

(TBD: assumptions and formulations of GloVe)

Contexts:
- Small context window: more syntax-based embeddings 
- Large context window: more semantics-based, topical embeddings 
- Context based on syntax: more functional, grouping words with same inflection

Limitations and improvements:
- Sensitive to superficial differences (dog/dogs): [+morphology](http://aclweb.org/anthology/W/W13/W13-3512.pdf)/[+character](http://aclweb.org/anthology/D15-1176)/[+subword](http://aclweb.org/anthology/D/D16/D16-1157.pdf)
- Insensitive to context (financial bank, bank of a river): multi-prototype embeddings [1](http://aclweb.org/anthology/N/N10/N10-1013.pdf) [2](http://aclweb.org/anthology/D/D14/D14-1113.pdf)
- Not necessarily coordinated with knowledge or across languages: cross lingual embeddings [1](http://aclweb.org/anthology/E/E14/E14-1049.pdf) [2](http://aclweb.org/anthology/P/P17/P17-1042.pdf) [3](http://aclweb.org/anthology/D/D17/D17-1207.pdf)
- Biased (encode stereotypical gender roles, racial biases): [de-biasing](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)

## Resources
- [GloVe](https://nlp.stanford.edu/projects/glove/)
- [FastText](https://github.com/facebookresearch/fastText)

## References
- CMU CS 11-747, Spring 2019: [Distributional Semantics and Word Vectors](http://phontron.com/class/nn4nlp2019/schedule/word-vectors.html)
