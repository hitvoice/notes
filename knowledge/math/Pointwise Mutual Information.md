# Pointwise Mutual Information

### Specification
In contrast to mutual information (MI), pointwise mutual information refers to single events, whereas MI refers to the average of all possible events.
$$
\operatorname {pmi} (x;y)\equiv \log {\frac {p(x,y)}{p(x)p(y)}}=\log {\frac {p(x|y)}{p(x)}}=\log {\frac {p(y|x)}{p(y)}}.
$$
The mutual information (MI) of the random variables X and Y is the expected value of the PMI over all possible outcomes.

### Properties
If $x$ and $y$ are independent, $\operatorname{pmi}(x;y) = 0$, indicating that the cooccurrence is uninformative.
PMI maximizes when $x$ and $y$ are perfectly associated (i.e. $p(y|x)=1$ or $p(x|y)=1$). The upper bound is $\min \left[-\log p(x),-\log p(y)\right]$.

A well-known problem of PMI is that it is biased towards infrequent events. A way to deal with this is Laplace smoothing. A constant positive value is added to the raw frequencies before calculating the probabilities. The larger the constant, the greater the smoothing effect. Laplace smoothing pushes the pmi values towards zero. The magnitude of the push depends on the raw frequency. If the frequency is large, the push is small; if the frequency is small, the push is large. Thus Laplace smoothing reduces the bias of PMI towards infrequent events.


### Reference
- https://en.wikipedia.org/wiki/Pointwise_mutual_information
- Turney, Peter D., and Patrick Pantel. "From frequency to meaning: Vector space models of semantics." Journal of artificial intelligence research 37.1 (2010): 141-188.