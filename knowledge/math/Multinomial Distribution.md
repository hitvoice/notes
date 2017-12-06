# Multinomial Distribution

For n independent trials each of which leads to a success for exactly one of k categories, with each category having a given fixed success probability, the multinomial distribution gives the probability of any particular combination of numbers of successes for the various categories. For example, it models the probability of counts for rolling a k sided die n times. 
## Specification
$$
\begin{align}
f(x_1,\ldots,x_k;n,p_1,\ldots,p_k) 
& {} = \Pr(X_1 = x_1\mbox{ and }\dots\mbox{ and }X_k = x_k) \\
& {} = \begin{cases} { \displaystyle {n! \over x_1!\cdots x_k!}p_1^{x_1}\cdots p_k^{x_k}}, \quad &
\mbox{when } \sum_{i=1}^k x_i=n \\  \\
0 & \mbox{otherwise,} \end{cases} \\
& {} = \frac{\Gamma(\sum_i x_i + 1)}{\prod_i \Gamma(x_i+1)} \prod_{i=1}^k p_i^{x_i}.
\end{align}
$$
for non-negative integers $x_1,\dots, x_k$.
### Related Distributions
* The last form expressed using the [Gamma function](Gamma%20Function.md) shows its resemblance to the [Dirichlet Distribution](./Dirichlet%20Distribution.md) which is its [Conjugate Prior](Conjugate%20Prior.md).
* When k = 2, the multinomial distribution is the binomial distribution.
* Categorical distribution, the distribution of each trial; for k = 2, this is the Bernoulli distribution.

Although it's imprecise, in many fields, especially NLP, categorical distribution is often confused with multinomial distribution.

## Properties
### Expectation
$$
\operatorname{E}(X_i) = n p_i
$$
### Covariance matrix
Each diagonal entry is the variance of a binomially distributed random variable, and is therefore
$$
\operatorname{var}(X_i)=np_i(1-p_i)
$$
The off-diagonal entries are the covariances:
$$
\operatorname{cov}(X_i,X_j)=-np_i p_j
$$
for i, j distinct.

## Reference
Multinomial distribution: [https://en.wikipedia.org/wiki/Multinomial_distribution](https://en.wikipedia.org/wiki/Multinomial_distribution)