# Probabilisitc Latent Semantic Analysis

![img](resources/plsa.png)

### Parameters
* $K$: number of topics
* $M$: number of documents
* $N_m$: length of m-th document
* $V$: size of vocabulary
* $\varphi$: $K\times V$, word distribution of topics
* $\theta$: $M\times K$, topic distribution of documents

### Likelyhood Function
$$
\log p(C) = \sum_{d\in C} \log p(d)
$$
where
$$
\log p(d) = \sum_{w\in V} c(w,d)\log p_d(w)
$$
$$
p_d(w)=\sum_{k=1}^K \theta_{d,k}p(w|\varphi_k)
$$
ML is done by EM algorithm 

### Criticism
* number of parameters: MK+KV, grows linearly by the number of documents. So pLSA is prone to overfitting and can't easily handle unseen documents (But sometimes what we want is just to "overfit" our dataset)

### Reference
Text Mining: [https://www.coursera.org/learn/text-mining](https://www.coursera.org/learn/text-mining)