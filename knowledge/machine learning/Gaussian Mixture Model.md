# Gaussian Mixture Model

Guassian mixure model. Exceeds k-means by the ability of handling oval-shaped clusters.

```
randomly initialize k centroids
loop
  assign cluster label with maximun likelyhood
  update mean and covariance matrix
until convergence
```
likelyhood (on the assumption that covariance matrix is diagonal):
$$
\begin{align}
p(x|c=k)
&{}\propto N(\mu_k,\Sigma_k)\\
&{}\propto-\|x-\mu_k\|^2
\end{align}
$$

### Reference
"Machine Learning" Lecture 16: [http://www.umiacs.umd.edu/~jbg/teaching/CSCI_5622/](http://www.umiacs.umd.edu/~jbg/teaching/CSCI_5622/)