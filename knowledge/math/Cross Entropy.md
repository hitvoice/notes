# Cross Entropy

$$
H(p,q)=\operatorname {E}\_{p}[-\log q]=H(p)+D_{{{\mathrm  {KL}}}}(p\|q)
$$

For discrete $p$ and $q$ this means:

$$
H(p,q)=-\sum_{x}p(x)\,\log q(x)
$$
As a distance measure between probability distributions, cross entropy has the unfortunate property that distributions with long tails are often modeled poorly with too much weight given to the unlikely events. Furthermore, for the measure to be bounded it requires that distribution Q be properly normalized. This sometimes presents a computational bottleneck if normalizer of Q is expensive to compute.

Logistic loss in the logistic regression is sometimes called cross-entropy loss, which measures the similarity between the prection and actual data labels:
$$
\begin{aligned}
L({\mathbf {w}})\ &=\ {\frac  1N}\sum_{{n=1}}^{N}H(p_{n},q_{n})\ =\ -{\frac  1N}\sum_{{n=1}}^{N}\ 
{\bigg [}y_{n}\log {\hat y}\_{n}+(1-y_{n})\log(1-{\hat  y}\_{n}){\bigg ]}
\end{aligned}
$$

Because the probability of the data label $y_i$ is 0 or 1 and is fixed, so in the softmax regression, the cross-entropy loss is expressed as:
$$
J(\theta) = - 
\bigg [
\sum_{i=1}^{m} \sum_{k=1}^{K} 1 \left{ y^{(i)} = k \right} \log \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}
\bigg ]
$$

### reference
https://en.wikipedia.org/wiki/Cross_entropy
http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
Pennington, Jeffrey, Richard Socher, and Christopher D. Manning. "Glove: Global Vectors for Word Representation." EMNLP. Vol. 14. 2014.
