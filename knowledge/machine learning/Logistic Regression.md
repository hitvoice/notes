# Logistic Regression

A special case of [softmax regression](Softmax%20Regression.md) with K=2: (due to the overparameterization property)
$$
\begin{align}
h_\theta(x) &=

\frac{1}{ \exp(\theta^{(1)\top}x)  + \exp( \theta^{(2)\top} x^{(i)} ) }
\begin{bmatrix}
\exp( \theta^{(1)\top} x ) \\
\exp( \theta^{(2)\top} x )
\end{bmatrix}\\

&=

\frac{1}{ \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) + \exp(\vec{0}^\top x) }
\begin{bmatrix}
\exp( (\theta^{(1)}-\theta^{(2)})^\top x )
\exp( \vec{0}^\top x ) \\
\end{bmatrix} \\

&=
\begin{bmatrix}
\frac{1}{ 1 + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
\frac{\exp( (\theta^{(1)}-\theta^{(2)})^\top x )}{ 1 + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) }
\end{bmatrix} \\

&=
\begin{bmatrix}
\frac{1}{ 1  + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
1 - \frac{1}{ 1  + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
\end{bmatrix}
\end{align}
$$
Finally:
$$
\begin{align}
P(y=1|x) &= h_\theta(x) = \frac{1}{1 + \exp(-\theta^\top x)}\\
P(y=0|x) &= 1 - P(y=1|x) = 1 - h_\theta(x).
\end{align}
$$
Sigmoid function... (left to be expanded)
$$
f(z) = \frac{1}{1+\exp(-z)}.
$$

### Reference
- http://ufldl.stanford.edu/tutorial/supervised/LogisticRegression/
- http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/