# Dirichlet-multinomial Distribution

It is a compound probability distribution, where a probability vector **p** is drawn from a Dirichlet distribution with parameter vector ${\boldsymbol {\alpha }}$, and an observation drawn from a multinomial distribution with probability vector **p** and number of trials N. 
### Specification
$$
\Pr({\mathbf  {x}}\mid {\boldsymbol  {\alpha }})=\int _{{{\mathbf  {p}}}}\Pr({\mathbf  {x}}\mid {\mathbf  {p}})\Pr({\mathbf  {p}}\mid {\boldsymbol  {\alpha }}){\textrm  {d}}{\mathbf  {p}}
$$
which results in the following explicit formula:
$$
\begin{align*}
\Pr(\mathbf {x} \mid {\boldsymbol {\alpha }})
&{}={\frac {\left(n!\right)\Gamma \left(\alpha _{0}\right)}{\Gamma \left(n+\alpha _{0}\right)}}\prod _{k=1}^{K}{\frac {\Gamma (x_{k}+\alpha _{k})}{\left(x_{k}!\right)\Gamma (\alpha _{k})}}\\\\
&{}={\frac {nB\left(\alpha _{0},n\right)}{\prod _{k:x_{k}>0}x_{k}B\left(\alpha _{k},x_{k}\right)}}
\end{align*}
$$
where $\alpha _{0}$ is defined as the sum ${\displaystyle \alpha _{0}=\sum \alpha _{k}}$ The latter form emphasizes the fact that zero count categories can be ignored in the calculation.
### Related distributions
* It reduces to the Categorical distribution as a special case when n = 1: $$\Pr({\mathbf  {x}}\mid {\boldsymbol  {\alpha }}) = \frac{\alpha_k}{\alpha_0}$$ where $\alpha_k$ is seen as the unnormalized probability of each category.
* It approximates the multinomial distribution arbitrarily well for large α. 

### Reference
Dirichlet-multinomial Distribution: [https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution](https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution)
