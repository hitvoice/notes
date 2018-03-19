# Stochastic Gradient Descent

### Motivation
Although batch methods are straight forward to get working provided a good off the shelf implementation because they have very few hyper-parameters to tune,
* Computing the cost and gradient for the entire training set can be very slow and sometimes intractable on a single machine if the dataset is too big to fit in main memory.
* There's no easy way to incorporate new data in an "online" setting.

### Tips
**Use a minibatch rather than a single example**
* reduces the variance in the parameter update and can lead to more stable convergence, 
* takes the advantage of parallelized computation. 

A typical minibatch size is 256, although the optimal size of the minibatch can vary for different applications and architectures. We use powers of 2 in practice because many vectorized operation implementations work faster when their inputs are sized in powers of 2. 

**Choose proper learning rate**
* use a small enough constant learning rate that gives stable convergence in the initial epoch or two (an epoch is a full pass through the training set)
* halve the value of the learning rate after each epoch.

There are many more advanced and sophisticated methods, some of which even allow no hand-set learning rates with no hyper-parameters, e.g. AdaGrad.

**Shuffle the data prior to each epoch**
avoids bias in the gradient and lead to better convergence.

**Implement gradient check**
Analytical gradient is more error prone to implement, though it's much faster than numerical approximate gradient. Check the correctness of your implementation by comparing the two gradients.


### Momentum
The objectives of deep architectures have a long shallow ravine leading to the optimum and steep walls on the sides. The negative gradient of standard SGD will point down one of the steep sides rather than along the ravine towards the optimum and thus can lead to very slow convergence. 
Momentum is one method for pushing the objective more quickly along the shallow ravine. The momentum update is given by

\begin{align*}
v &= \gamma v+ \alpha \nabla_{\theta} J(\theta; x^{(i)},y^{(i)}) \\\\
\theta &= \theta - v
\end{align*}

In the above equation $v$ is the current velocity vector which is of the same dimension as the parameter vector $\theta$, which is initialized to 0. The learning rate $\alpha$ is as described above, although when using momentum $\alpha$ may need to be smaller since the magnitude of the gradient will be larger. Finally momentum $\gamma \in (0,1]$ determines for how many iterations the previous gradients are incorporated into the current update. Generally $\gamma$ is set to 0.5 until the initial learning stabilizes and then is increased to 0.9 or higher.

### reference
- http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/
- http://cs231n.github.io/optimization-1/
- [Stanford CS224d Lecture 6](http://cs224d.stanford.edu/lectures/CS224d-Lecture6.pdf)
