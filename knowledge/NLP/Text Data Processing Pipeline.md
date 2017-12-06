### Tokenization
An accurate English tokenizer must know how to handle punctuation (e.g., don’t, Jane’s, and/or), hyphenation (e.g., state-of-the-art versus state of the art), and recognize multi-word terms (e.g., Barack Obama and ice hockey) (Manning et al., 2008). We may also wish to ignore stop words, high-frequency words with relatively low information content, such as function words (e.g., of, the, and) and pronouns (e.g., them, who, that). To replace digits with a single token or treat them differently also needs consideration.

Chinese tokenization: [结巴中文分词](https://github.com/fxsjy/jieba/)([code example](resources/tokenize.py) [POS tags](resources/pos.doc)) [THULAC](http://thulac.thunlp.org)

### Normalization
* case folding: converting all words to lower case
* stemming: reducing inflected words to their stem or root form

In general, normalization increases recall and reduces precision. If we have a small corpus, we may not be able to afford to be overly selective, and it may be best to aggressively normalize the text, to increase recall. If we have a very large corpus, precision may be more important, and we might not want any normalization.

### Annotation
* POS tagging: marking words according to their parts of speech
* word sense tagging: marking ambiguous words according to their intended meanings
* parsing: analyzing the grammatical structure of sentences and marking the words in the sentences according to their grammatical roles

We expect annotation to decrease recall and increase precision.

### Building the Frequency Matrix
A typical approach to building a frequency matrix involves two steps. First, scan sequentially through the corpus, recording events and their frequencies in a hash table, a database, or a search engine index. Second, use the resulting data structure to generate the frequency matrix, with a sparse matrix representation.

A Row-based linked list sparse (LIL) matrix is preferred for constructing, and a Compressed Sparse Row (CSR) matrix is preferred for storage.
```python
from scipy import sparse
X = sparse.lil_matrix((nrows,ncols)) 
# constructing...
X = X.tocsr()
```
Check this [file-io code](resources/io.py) for scipy sparse matrix.

Scikit-learn models compatible with sparse matrix:
* linear_model.LogisticRegression()
* svm.SVR()
* svm.NuSVR()
* linear_model.LinearRegression()
* neighbors.KNeighborsRegressor()
* naive_bayes.MultinomialNB()
* naive_bayes.BernoulliNB()
* linear_model.PassiveAggressiveRegressor()
* linear_model.PassiveAggressiveClassifier()
* linear_model.Perceptron()
* linear_model.Ridge()
* linear_model.Lasso()
* linear_model.ElasticNet()

### Weighting the Elements
The idea of weighting is to give more weight to surprising events and less weight to expected events. The hypothesis is that surprising events, if shared by two vectors, are more discriminative of the similarity between the vectors than less surprising events.
* tf-idf (usually for term-document matrix)
```python
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X)
```
* Positive PMI (of word and context, usually for word-context matrix)
* [BNS Scaling](BNS%20Scaling.md)
* binary: `X = X.sign()`


While the tf–idf normalization is often very useful, there might be cases where the binary occurrence markers might offer better features. In particular, some estimators such as Bernoulli Naive Bayes explicitly model discrete boolean random variables. Also, very short texts are likely to have noisy tf–idf values while the binary occurrence info is more stable.

### Dimension Reduction
* truncated-SVD (LSI, LSA)
Although traditional text mining algorithms are directly applied to the reduced representation obtained from LSA, it's sometimes necessary to scale each document $\bar X=(x_1\dots x_d)$ to $\frac{1}{\sqrt{\sum_{i=1}^d x_i^2}}(x_1\dots x_d)$ to ensure that documents of varying length are treated in a uniform way. After this scaling, traditional numeric measures, such as the Euclidean distance, work more effectively.
* NMF
* PLSI
* IS (Iterative-Scaling)
* KPCA
* LDA
* DCA

### reference
- [Scikit-learn models compatible with sparse matrix](https://www.kaggle.com/c/amazon-employee-access-challenge/forums/t/5128/scikit-learn-models-compatible-with-sparse-matrix)
- [scipy.sparse](https://docs.scipy.org/doc/scipy-0.18.1/reference/sparse.html)
- [scikit-learn: Working With Text Data](http://scikit-learn.org/dev/tutorial/text_analytics/working_with_text_data.html)
- [Save / load scipy sparse csr_matrix in portable data format](http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format)
- Turney, Peter D., and Patrick Pantel. "From frequency to meaning: Vector space models of semantics." Journal of artificial intelligence research 37.1 (2010): 141-188.
- Book: Data Mining-The Textbook
- Udemy course: Data Mining & Machine Learning Bootcamp