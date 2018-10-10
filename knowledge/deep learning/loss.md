### Loss functions
#### Square Loss
$$
J(\theta)=
\frac{1}{m} \sum_{i=1}^m \left( \frac{1}{2} \left\| h_{\theta}(x^{(i)}) - y^{(i)} \right\|^2 \right)
$$
It's equivalant to maximizing the log-likelihood of a conditional Gaussian distribution.
#### Cross-Entropy Loss
$$
J(\theta)=
 \frac{1}{m} \sum_{i=1}^m 
 -\log\left(\hat y_t^{(i)}\right)
$$
where $t$ is the index of the correct class. When using cross-entropy loss, it is assumed that the network’s output is transformed using the softmax transformation, in which case increasing the mass assigned to the correct class means decreasing the mass assigned to all the other classes.

Label smoothing regularizes a model based on a softmax with $k$ output values by replacing the hard 0 and 1 classification targets with targets of $\frac{\epsilon}{k-1}$ and $1−\epsilon$, respectively. Besides this uniform smoothing, "unigram smoothing" distributes the remaining probability mass proportionally to the marginal probability of classes. It prevents the model to learn larger and larger weights, making more extreme predictions forever to reach the unreachable 0 and 1 target without discouraging correct classification (unlike weight decay). Label smoothing is equivalent to adding the KL divergence between the uniform distribution $u$ and the network's predicted distribution $p_{\theta}$ to the negative log-likelihood:
$$
\mathcal{L}(\theta) = -\sum\log p_\theta(y\mid x) - D_{KL}(u\|p_\theta(y\mid x)).
$$
By reversing the direction of the KL divergence, we get confidence penalty by adding the negative entropy:
$$
\mathcal{L}(\theta) = -\sum\log p_\theta(y\mid x) - \beta H(p_\theta(y\mid x)).
$$
For reference see [this paper](https://arxiv.org/pdf/1701.06548.pdf)(ICLR'17).

When applying cross entropy loss for generating discrete policy in reinforcement learning or for some other generation tasks, it's sometimes useful to tune the "temperature" $t$ in softmax function. With the temperature parameter, $x$ becomes $x/t$ before the softmax tranformation. As the result, the lower the temperature is, the higher the entropy of the generated distribution will be. It's sometimes useful to set the temperature high in the begining and then "anneal" it repeatedly to encourage exploration first and then put emphasis on exploitation in reinforcement learning. In contrast, in supervised learning, it's desired to have a quick convergence at the beginning while preventing overfitting near the end of training.
#### Hinge Loss
$$
J(\theta)=
 \frac{1}{m} \sum_{i=1}^m 
 \max\left(0,1-\left(\hat y_t^{(i)} - \hat y_k^{(i)}\right) \right)
$$
where $k=\operatorname{argmax}_{j\neq t}\hat y_j$. Hinge loss attempts to score the correct class above all other classes with a margin of at least 1.
#### Ranking Loss (hinge)
The goal is to score correct items above incorrect ones, given pairs of correct and incorrect items $x_p$ and $x_n$. Such training situations arise when we have only positive examples, and generate negative examples by corrupting a positive example.

<div>
$$
J(\theta)=
\frac{1}{m} \sum_{i=1}^m\max \left( 0,1- \left( h_{\theta}(x^{(i)}_p)-h_{\theta}(x^{(i)}_n)\right) \right)
$$
</div>

The objective is to score correct inputs over incorrect ones with a margin of at least 1.
#### Ranking Loss (log)

<div>
$$
J(\theta)=
\frac{1}{m} \sum_{i=1}^m\log \left( 1+ \exp\left(-\left( h_{\theta}(x^{(i)}_p)-h_{\theta}(x^{(i)}_n)\right)\right) \right)
$$
</div>

#### Distance Loss
The objective is to make the representation of an input sample closer to the class centroid of its correct class. ([reference](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf))

<div>
$$
J(\theta)=
\frac{1}{m} \sum_{i=1}^m-\log \left(\frac{\exp(-d(x, c'))}{\sum_{c}\exp(-d(x, c))} \right)
$$
</div>

#### Rating Prediction
Assume that all the ratings lie in $[1,K]$. Real-valued scores are allowed for ground-truth ratings that are an average over the evaluations of several human annotators. 
Using a softmax layer as the last layer, we get a probability output $\hat p$, which is seen as the probability of each discrete integer rating. We predict the scoring by $\hat y = r^T\hat p$, where $r^T=[1\ 2\ \dots\ K]$, and optimize the model over the following cost function:
$$
J(\theta)=\frac{1}{m}\sum_{k=1}^m 
\operatorname{KL}\left(
  p^{(k)}\left\|\hat p^{(k)}
\right)\right.
+\frac{\lambda}{2}\|\theta\|_2^2\ .
$$
The sparse target distribution $p$ is defined as:
$$
\begin{align*}
p_i = 
\begin{cases} 
1-\left(y-\lfloor y\rfloor\right), \ &i=\lfloor y\rfloor \\\\
y-\lfloor y\rfloor,    &i=\lfloor y\rfloor+1\\\\
0    &\mbox{otherwise }
\end{cases}
\end{align*}
$$
for $1\leq i\leq K$. For example, if $K=5$ and $y=3.6$, $p=[0, 0, 0.4, 0.6, 0]$. See [this paper](http://arxiv.org/pdf/1503.00075.pdf).

## Appendix
### prove for the relationship between label smoothing and entropy
$$
\begin{align*}
\underset{\theta}{\operatorname{argmin}}\mathcal{L}(\theta)&=
\underset{\theta}{\operatorname{argmin}}\sum H(p(y'),p_\theta(y_t|x))\\\\
&=\underset{\theta}{\operatorname{argmin}}\sum 
-(1-\epsilon)\log p_\theta(y_t|x)-\sum_{i\neq t}\frac{\epsilon}{c-1}\log p_\theta(y_i|x)\\\\
&=\underset{\theta}{\operatorname{argmin}}\sum -(1-\epsilon+\frac{\epsilon}{c-1})\log p_\theta(y_t|x)-\sum_i\frac{\epsilon}{c-1}\log p_\theta(y_i|x)\\\\
&=\underset{\theta}{\operatorname{argmin}}\sum
-\log p_\theta(y_t|x)-\frac{\epsilon}{c-1-(c-2)\epsilon}\sum_i\log p_\theta(y_i|x)
\end{align*}
$$
Let
$$
\alpha=\frac{\epsilon c}{c-1-(c-2)\epsilon}
$$
Then
\begin{align*}
\underset{\theta}{\operatorname{argmin}}\mathcal{L}(\theta)
&=\underset{\theta}{\operatorname{argmin}}\sum
-\log p_\theta(y_t|x)-\frac{\alpha}{c}\sum_i\log p_\theta(y_i|x)\\\\
&=\underset{\theta}{\operatorname{argmin}}\sum
-\log p_\theta(y_t|x)-\frac{\alpha}{c}\sum_i\log p_\theta(y_i|x)-\log c\\\\
&=\underset{\theta}{\operatorname{argmin}}\sum
-\log p_\theta(y_t|x)-\frac{\alpha}{c}\sum_i\log p_\theta(y_i|x)c\\\\
&=\underset{\theta}{\operatorname{argmin}}\sum
-\log p_\theta(y_t|x)+\frac{\alpha}{c}\sum_i\log \frac{1/c}{p_\theta(y_i|x)}\\\\
&=\underset{\theta}{\operatorname{argmin}}\sum
-\log p_\theta(y_t|x)+\alpha D_{KL}(u\|p_\theta(y\mid x))\\\\
&=\underset{\theta}{\operatorname{argmin}}-\sum
\log p_\theta(y_t|x)-\alpha D_{KL}(u\|p_\theta(y\mid x))\\\\
\end{align*}

### reference
- [Deep Learning Book](http://www.deeplearningbook.org/)
- [Oxford Deep NLP 2017 Lecture 4](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%204%20-%20Language%20Modelling%20and%20RNNs%20Part%202.pdf)
- [A primer on neural network models for natural language processing](http://arxiv.org/pdf/1510.00726)
- [Regularizing Neural Networks By Penalizing Confident Output Distributions](https://arxiv.org/pdf/1701.06548.pdf)
- [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://arxiv.org/pdf/1503.00075.pdf)
