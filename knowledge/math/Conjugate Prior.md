# Conjugate Prior

In Bayesian probability theory, if the posterior distributions p(θ|x) are in the same family as the prior probability distribution p(θ), the prior and posterior are then called conjugate distributions, and the prior is called a conjugate prior for the likelihood function. 

To see a conclusion of conjugate distributions of all frequently-used likelyhood functions, their posterior hyperparameters and interpretation of hyperparameters, check [this page](https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions).

To see why we tend to choose the conjugate prior as the prior, consider the proof of the conjugate prior:
$$
\begin{align}
p(\theta\mid x)
&=\frac{p(\theta,x)}{p(x)}\\
&=\frac{p(\theta)p(x\mid \theta)}{p(x)}\quad \text{(conditional prob.)}\\
&=\frac{p(\theta)p(x\mid \theta)}{\int{p(\theta,x)\mathrm{d}\theta}}\quad \text{(marginal distribution)}\\
&=\frac{p(\theta)p(x\mid \theta)}{\int{p(\theta)p(x\mid \theta)\mathrm{d}\theta}}\quad \text{(conditional prob.)}
\end{align}
$$
Recall that prior distribution is the probability distribution over probabilites. So $p(\theta)$ is the probability of seeing $\theta$ (the model parameter) in the prior distribution. If prior distribution is not the conjugate prior, the form of posterior distribution (can be told from the equation above) will be very complex and computationally intractable.

### reference
Conjugate Prior: [https://en.wikipedia.org/wiki/Conjugate_prior](https://en.wikipedia.org/wiki/Conjugate_prior)