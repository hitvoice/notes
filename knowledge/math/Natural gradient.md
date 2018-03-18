# Natural gradient

### Background Story

Gradient descent is not efficient in variational inference, because probability distributions do not naturally live in Euclidean space but rather on a statistical manifold. There are better ways of defining the distance between distributions, one of the simplest being the **symmetrized Kullback-Leibler divergence**:

$$
KL_{sym}(p_1,p_2) = \frac{1}{2}(KL(p_1||p_2) + KL(p_2||p_1))
$$

In differential geometry, distance on a manifold is given by the bilinear form
$$
\|\mathrm{d}\phi\|^2 = \left < \mathrm{d}\phi, G(\phi)\mathrm{d}\phi \right > = \sum_{ij} g_{ij}(\phi)\mathrm{d}\phi_i \mathrm{d}\phi_j 
$$

The matrix $ G(\phi) = [g_{ij} (\phi)]$ is called the **Riemannian metric tensor**. 
In Euclidean space with an orthonormal basis $G(\phi)$ is simply the identity matrix. When  $\Phi$ is a space of parameters of probability distributions and the symmetrized KL divergence is used to measure the distance between distributions then $G(\phi)$ turns out to be the **Fisher information matrix**:

$$
{\left(\mathcal{I} \left(\theta \right) \right)}\_{i, j}=
\operatorname{E}
\left\\\[\left.
 \left(\frac{\partial}{\partial\theta_i} \log f(X;\theta)\right)
 \left(\frac{\partial}{\partial\theta_j} \log f(X;\theta)\right)
\right|\theta\right\\\].
$$
### The Story
In gradient ascent (of the evidence lower bound in variational inference), we want to maximize:
$$
\begin{align*} L(\phi + \mathrm d\phi) = L(\phi) + \epsilon \nabla L(\phi)^T v \end{align*}
$$
with constraint: $\|v\|^2 = \langle v,G(\phi)v \rangle = 1$. Solve with Lagrange mulitpliers, we get the **natural gradient** by multiplying the inverse of the Fisher information matrix and the first derivative:
$$
G(\phi)^{-1}\nabla L(\phi) 
$$

### reference
The Natural Gradient: [https://hips.seas.harvard.edu/blog/2013/01/25/the-natural-gradient/](https://hips.seas.harvard.edu/blog/2013/01/25/the-natural-gradient/)
