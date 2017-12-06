# Variational Bayes

## Discussions
### Design
* EM is a special case of mean field VB, in which we assume some $q_i(x_i)$ are infitely narrow. Mean field VB is more general with fewer assumptions
* You don't have to completely maximize $L(q)$ at each step. Just increase it. $\Rightarrow$ stochastic gradient VB
* You don't have to decompose completely. One $q$ can contain several $x_i$, especially when some posterior is computationally easy with mature methods (e.g. HMM)

### Comparison with MCMC method
**advantages of VI**
* for small to medium problems, it is usually faster
* Variational inference is trivial to parallelize, since you can create a variational distribution that matches your computational topology
* Variational inference is also trivial to turn into an online algorithm, which can also be faster if you don't have access to a cluster 
*  The sample variance of an MCMC estimate (or a rejection sampling estimate) usually approaches 0 as you draw more and more samples, while a variational estimate already has sample variance of exactly 0 because it's deterministic.  
*  is it easy to determine when to stop

**advantages of MCMC**
* Variational inference is irredeemably biased, whereas MCMC's bias approaches 0 as you run the Markov chain for longer and longer.
* it is often easier to implement
* it is applicable to a broader range of models, such as models whose size or structure changes depending on the values of certain variables (e.g., as happens in matching problems), or models without nice conjugate priors; 
* sampling can be faster than variational methods when applied to really huge models or datasets. The reason is that sampling passes specific values of variables (or sets of variables), whereas in variational inference, we pass around distributions. Thus sampling passes sparse messages, whereas variational inference passes dense messages For comparisons of the two approaches

### Reference
- [https://www.quora.com/When-should-I-prefer-variational-inference-over-MCMC-for-Bayesian-analysis](https://www.quora.com/When-should-I-prefer-variational-inference-over-MCMC-for-Bayesian-analysis)
- Book: Machine Learning - A Probabilistic Perspective(Chapter 24)