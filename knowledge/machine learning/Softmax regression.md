Also called multinomial logistic regression, is a generalization of logistic regression to the case where we want to handle multiple classes.
### Specification
Hypothesis:
($x$: $m\times d$, $\theta$: $d\times K$)

$$
\begin{align*}
h_\theta(x) =
\begin{bmatrix}
P(y = 1 | x; \theta) \\\\
P(y = 2 | x; \theta) \\\\
\vdots \\\\
P(y = K | x; \theta)
\end{bmatrix}
= \frac{1}{ \sum_{j=1}^{K}{\exp(\theta^{(j)\top} x) }}
\begin{bmatrix}
\exp(\theta^{(1)\top} x ) \\\\
\exp(\theta^{(2)\top} x ) \\\\
\vdots \\\\
\exp(\theta^{(K)\top} x ) \\\\
\end{bmatrix}
\end{align*}
$$

Cost function ([cross entropy loss](../math/Cross%20Entropy.md)):

$$
\begin{align*}
J(\theta) = -  \sum_{i=1}^{m} \sum_{k=1}^{K}  1\left\{y^{(i)} = k\right\} \log \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}
\end{align*}
$$
Optimize with gradient descent:
$$
\begin{align*}
\nabla_{\theta^{(k)}} J(\theta) = - \sum_{i=1}^{m}{ \left [ x^{(i)} \left( 1\{ y^{(i)} = k\}  - P(y^{(i)} = k | x^{(i)}; \theta) \right) \right ]  }
\end{align*}
$$

### Redundancy of Parameters
The actual number of parameters are just $(K-1)n$, rather than $Kn$, because probabilities always sum to 1. We can find that subtracting $\psi$ from every $\theta(j)$ does not affect our hypothesis’ predictions at all:
$$
\begin{align*}
P(y^{(i)} = k | x^{(i)} ; \theta)
&= \frac{\exp((\theta^{(k)}-\psi)^\top x^{(i)})}{\sum_{j=1}^K \exp( (\theta^{(j)}-\psi)^\top x^{(i)})}  \\\\
&= \frac{\exp(\theta^{(k)\top} x^{(i)}) \exp(-\psi^\top x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)}) \exp(-\psi^\top x^{(i)})} \\\\
&= \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}.
\end{align*}
$$
So one can instead set $\theta(K)=\vec 0$ and optimize only with respect to the remaining parameters.
Note that $J(\theta)$ is still convex, but the Hessian is singular, which causes a straightforward implementation of Newton’s method to run into numerical problems.

### Invariant to Constant Offset
Softmax is invariant to constant offsets in the input, that is, for any input vector $x$ and any constant $c$, 
$$
\operatorname{softmax}(x) = \operatorname{softmax}(x + c)
$$
In practice, we make use of this property and choose $c = − \max_i x_i$ when computing softmax probabilities for numerical stability (i.e. subtracting its maximum element from all elements of $x$) so that all the values of $\exp(\cdot)$ will be in range $(0,1]$, not exploding with $x$. 
### Code Example
In numpy:
```python
def softmax(x):
  """
  Args:
    x:   np.array with shape (n_samples, n_features).
  Returns:
    out: np.array with shape (n_sample, n_features). 
  """
  maxes = np.max(x, axis=1, keepdims=True)
  x_red = x - maxes
  x_exp = np.exp(x_red)
  sums = np.sum(x_exp, axis=1, keepdims=True)
  out = x_exp / sums
  return out 
```
In tensorflow:
```python
def softmax(x):
  """
  Args:
    x:   tf.Tensor with shape (n_samples, n_features).
  Returns:
    out: tf.Tensor with shape (n_sample, n_features). 
  """
  maxes = tf.expand_dims(tf.reduce_max(x, reduction_indices=[1]), 1)
  x_red = x - maxes
  x_exp = tf.exp(x_red)
  sums = tf.expand_dims(tf.reduce_sum(x_exp, reduction_indices=[1]), 1)
  out = x_exp / sums
  return out 
```

It often comes up in neural networks, generalized linear models, topic models and many other probabilistic models that one wishes to convert an unconstrained vector of numbers to a discrete probabilistic distribution. A very common way to address this is to use the softmax transformation.  If we want to perform the softmax transformation and then draw from a discrete distribution, it turns out that it doesn’t actually require constructing the discrete distribution (Gumbel-Max Trick):
>  add Gumbel noise to each $x_k$ and then take the argmax

It can be proved that this is exactly the softmax probability.



### Reference
- [UFLDL Tutorial: Softmax Regression](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)
- [Stanford CS224d PSet1 Solutions](http://cs224d.stanford.edu/assignment1/assignment1_soln)
- [Stanford CS224d PSet2 Solutions Code](http://cs224d.stanford.edu/assignment2/assignment2_dev.zip)
- [The Gumbel-Max Trick for Discrete Distributions](https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/)
