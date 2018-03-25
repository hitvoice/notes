The basic update rule for gradient descent:
$$
\begin{align*}
W_{ij}^{(l)} &= W_{ij}^{(l)} - \alpha \frac{\partial}{\partial W_{ij}^{(l)}} J(W,b) \\\\
b_{i}^{(l)} &= b_{i}^{(l)} - \alpha \frac{\partial}{\partial b_{i}^{(l)}} J(W,b)
\end{align*}
$$

where $\alpha$ is the learning rate. The **back propagation** algorithm gives an efficient way to compute these partial derivatives. The idea behind back propagation is that we can re-use derivatives computed for higher layers in computing derivatives for lower layers (in terms of back propagated error).
The computation of the partial derivative for a single example is as follows (and computation is usually parallelized):

1. Perform a feedforward pass, computing the activations for layers $L_2$, $L_3$, up to the output layer $L_{nl}$, using the equations defining the forward propagation steps
2. For the output layer (layer $n_l$), set $$ \delta^{(n_l)} =a^{(n_l)}-y$$
3. For $l=n_l−1,n_l−2,n_l−3,\dots,2$ (there's no "error" in input layer), set
   $$\delta^{(l)} = \left((W^{(l)})^T \delta^{(l+1)}\right) \bullet f'(z^{(l)})$$
4. Compute the desired partial derivatives for a single training example:

$$
\begin{align*}
\nabla_{W^{(l)}} J(W,b;x,y) &= \delta^{(l+1)} (a^{(l)})^T+\lambda W^{(l)}, \\\\
\nabla_{b^{(l)}} J(W,b;x,y) &= \delta^{(l+1)}.
\end{align*}
$$

The $\bullet$ denotes the element-wise product operator (also called the Hadamard product). The same extension goes for $f(\cdot)$ and $f'(\cdot)$. $\lambda$ is the regularization parameter. The partial derivatives are then sum up (and normalized by the number of training examples if they should match the cost function) to get the partial derivative w.r.t a whole batch (or minibatch). Since the back propagation is hard to implement and prone to tiny errors and bugs, remember to perform [gradient checking](http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization) after implementation. View [this](resources/gradient-checking.png) common asked question about gradient checking.

A clean python implementation and illustration for neural network can be found [here](resources/nn-implement.html).

## tips for initialization
Because of gradient-based optimization and early stopping, the final parameters are usually  close to the initial parameters. Initialization acts like some kind of prior. A general principle for initialization is to initialize with values close to 0. This prior says that it is more likely that units do not interact with each other than that they do interact. Units interact only if the likelihood term of the objective function expresses a strong preference for them to interact. In practice, it also prevents saturation of tanh units and thus accelerate convergence. 

A typical recipe is to initialize all the parameters $\propto\mathcal N(0,\epsilon^2)$ for some small $\epsilon$, say 0.01 or $\sqrt{2/\operatorname{fan-in}}$; another choice is initializing hidden layer biases to 0, weights $\propto \operatorname{Uniform}(-r,r)$, $r=\sqrt{6/(\operatorname{fan-in}+\operatorname{fan-out})}\ $ for tanh units (4x bigger for sigmoid units), where "fan-in" is the size of the previous layer and "fan-out" is the size of the next layer, which is equivalant to the number of rows and columns of this weight matrix (this is called Xavier initialization or Glorot initialization, which is initially described in [this paper](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)). 

The bias term is usually set to zero, with the following exceptions:
* Set the bias of a ReLU hidden unit to 0.1 rather than 0 to avoid saturating the ReLU at initialization.
* It is often beneficial to initialize the bias to obtain the right marginal statistics of the output. We can set the bias vector $b$ by solving the equation $\operatorname{softmax}(b) = c$, where $c$ is the marginal distribution of the classes.
* Set proper biases for gate units to "open" it at the beginning, otherwise units behind the gate do not have a chance to learn.

Unsupervised pre-training, supervised training on a related (or even unrelated) task are also good choices for initialization. In this case our prior specifies which units should interact with each other, and how they should interact. Some of these initialization strategies may yield faster convergence and better generalization than random initialization because they encode information about the distribution in the initial parameters of the model. They also benefits from larger initial weights, which will yield a stronger symmetry breaking effect, helping to avoid redundant units, and help to avoid losing signal during forward or back-propagation through the linear component of each layer.

For embeddings, one will often use the random initialization approach to initialize the embedding vectors of commonly occurring features, such as part-of-speech tags or individual letters, while using some form of supervised or unsupervised pre-training to initialize the potentially rare features, such as features for individual words. If you have little task-specific training data, or poor coverage of the vocabulary, it's better to fix the embedding and rely on task-specific projections to capture information salient to the task (pretrained word embeddings can be found in [Pre-trained word vectors for 294 languages](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md).

## optimization algorithm
On smaller datasets (or a small optimization problem like neural style transfer) L-BFGS or Conjugate Gradients win. On large datasets, mini-batch SGD usually wins over all batch methods.
Typical size of each mini batch : 20 to 1000. If full-batch training is affordable, use L-BFGS. If mini-batch is the case, Adam is the most common choice. Other choices includeSGD with momentum, RMSProp, AdaDelta and so on.

<img src="resources/adam.png" width="700">

Adam is the abbreviation for "adaptive moment estimation".Because $(1-\epsilon)^{1/\epsilon} \approx 1/e$, if beta1=0.99, it takes $1/(1-0.99)=100$ iterations to make the first_moment small enough (<37% of the original value). The first_moment acts as exponential weighted averages of derivatives, and the second_moment aims to damp out the value in the oscillating directions. In practice, beta1 and beta2 are rarely tuned, while different choices of the learning rate (and learning rate scheduling) may affect the optimization process in a more noticable way.

Most mini-batch optimization algorithms have the learning rate hyperparameter. Strategies include fixed learning rate, learning rate decay, cyclic learning rate (with snapshot ensemble). Common learning rate decay schemes:
- step decay, say decaying learning rate by half every few epochs
- exponential decay
$$
\alpha=\alpha_0\mathrm{e}^{-kt}\ (k>0)
$$
or 
$$
\alpha=\alpha_0 k^t\ (k<1)
$$
- linear decay
$$
\alpha = \alpha_0 \max(\epsilon, k-ct)
$$
$\alpha_0\epsilon$ is the minimun amount of learning rate. $k$ and $c$ provide the offset and slope of the decay.

- 1/t decay
$$\alpha=\alpha_0/(1+kt)$$
In the definitions above, $k$ is called the decay rate and $t$ is the number of epochs.

- Inversed sigmoid decay
$$
\alpha = \frac{k}{k+\exp (i/k)}\ (k \geq 1)
$$

Relation between the learning rate and convergence:
<div align="center">
 <img src="resources/lr.jpg" width="300">
</div>

**Batch normalizaion** is a standard strategy for optimization. Let $H$ be a minibatch of activation of a layer, we replace it with
$$
H' = \gamma\frac{H-\mu}{\sigma}+\beta
$$
At training time, we set
$$
\mu = \frac{1}{m}\sum_iH_i
$$
and
$$
\sigma=\sqrt{\varepsilon+\frac{1}{m}\sum_i \left(H-\mu\right)_i^2},
$$
where $\varepsilon$ is a small positive value (e.g. $10^{-8}$) to avoid undefined gradient. All the 4 parameters ($\mu, \sigma, \gamma, \beta$) are trainable during back propagation. At test time, we use the exponentially weighted averages of means and variances.

The gradient tells how to update each parameter, under the assumption that the other layers do not change. In practice, we update all of the layers simultaneously. The effect of an update to the parameters for one layer depends so strongly on all of the previous layers and unexpected results can happen because of this (this is called "covariate shift"). Batch normalization acts to standardize the mean and variance of each unit in order to stabilize learning, while allowing the relationships between units and the nonlinear statistics of a single unit to change. The new parameters $\gamma$ and $\beta$ have much lower learning dynamics, because they are not determined by a complicated interaction between the parameters in the previous layers. The estimation of mean and variance is based simply on a mini-batch and is not accurate. This adds some noise to the $z$ values and has a slight regularization effect (the larger the batch size, the smaller the regularization effect). With batch normalization, the model can have higher learning rates, more resiliant training, and less need for dropout.

Instance normalization can be seen (and implemented) as batch normalization with batch size 1 and dimension $m\times d$, except that running average and variance originally in the same dimension are collected together and copied to corresponding dimensions when a new batch comes.

Other optimization strategies and meta-algorithms, like supervised pretraining, curriculum learning, etc. can be found in section 8.7 (Page 322/PDF 337) of Deep Learning Book.

## hyper-parameters
* learning rate schedule
* minibatches
* parameter initialization
* number of hidden units
* number of layers
A network with even one hidden layer is sufficient to fit the training set. Deeper networks often are able to use far fewer units per layer and far fewer parameters and often generalize to the test set, but are also often harder to optimize. 
* regularization/earlystopping
* ...

Efficiently search for hyper-parameter configurations: random hyperparameter search.

<div align="center">
 <img src="resources/random-search.png" width="500">
</div>

## regulariazation
A visualization of training/validation error with time is helpful for diagnostics. When the training error keeps going down and down, even &lt;1%, and the validation error gets stuck, it's likely that the model is memorizing all the training data and is overfitting.

* Simple first step: Reduce model size by lowering number of units and layers and other parameters
* Standard L1 or L2 regularization on weights ($\lambda$ can be different on different layers if the computational cost is acceptable). L2 regularization is also called "weight decay" because parameters will be multiplied by $1-\alpha\lambda/m$ after each update ($\alpha$ is the learning rate and $m$ is the batch size).
* Constrain L2 norms of weight vectors to fixed number $s$. If $\|W\|_2^2>s$, then rescale it so that $\|W\|_2^2=s$.
* Sparse activation by L1 regularizer.
* Dataset augmentation (discussed later)
* Label smoothing for softmax output.
* Early Stopping (discussed later)
* If the network is small, ensemble multiple (e.g 5) networks
* Dropout

### Dataset Argumentation
* Easiest for classification: apply (hand-designed) transformations that would not change the correct class. It's something you should always do. Just a matter of what kind and how much.
* Injecting noise in the input or the hidden units can also be seen as a form of data augmentation. (The dropout algorithm is the main development of this approach.) But injecting noise in the input is mostly not a good idea because it causes direct information loss.
* Adding noise to the weights can be interpreted as a stochastic implementation of a Bayesian inference over the weights, which reflect the uncertainty of a probability distribution. It can also be interpreted as pushing the model to points that are not merely minima, but minima surrounded by flat regions. It has been shown to be an effective in the context of RNN.

### Early Stopping
Save parameters after each, say 3 epochs (or 1/3 epoch for a large dataset like ImageNet) and use parameters that give best validation error. To reuse data in validation set, there are 2 strategies:
* initialize the model again and retrain on all of the data for the same number of steps determined by early stopping (not optimal).
* keep the parameters and continue training using all of the data until the average loss function on the validation set falls below the value of the training set objective when early stopping (not guaranteed to terminate).

### Dropout
#### How
* Training time: at each instance of evaluation (in online SGD training), randomly set $p\in [0,1]$ of the inputs to each neuron to 0 (say an input unit is included with probability 0.8 and a hidden unit is included with probability 0.5).
* Test time: scale the model weights, multiply them by $p$.
#### Why
* This prevents feature co-adaptation: A feature cannot only be useful in the presence of particular other features.
* In a single layer: A kind of middle-ground between Naïve Bayes (where all feature weights are set independently) and logistic regression models (where weights are set in the context of all others).
* It can be seen as a process of constructing new inputs by multiplying by noise (data augmentation)
* Can be seen as a form of model bagging (but not quite the same): training a network with stochastic behavior and making predictions by averaging (with weight sharing). A multi-layer perceptron with dropout applied at every layer can be interpreted as Bayesian model averaging. When applied to linear models, dropout is equivalent to L2 weight decay.

#### Tips
* Because dropout is a regularization technique, it reduces the effective capacity of a model. To offset this effect, the size of the model and the number of iterations should be larger. For very large datasets, regularization confers little reduction in generalization error.
* If the training accuracy is always lower than the validation accuracy, you may regulate too much and the model goes underfitting. Try reducing the dropout rate or removing dropout in early layers (Since the general rule of thumb is to gradually increase dropout rate from beginning to end). When finetuning a model with the dropout rate adjusted, relevant weights should be rescaled. If the implementation is "inverted dropout" (e.g. in Keras), there's no need to adjust the weights.
* For networks containing recurrent units, do not apply dropout on recurrent connections. (for design of recurrent dropout mask, refer to [Baysian Dropout](https://arxiv.org/abs/1512.05287), NIPS 2016)
* When extremely few labeled training examples are available, dropout is less effective. Unsupervised feature learning can gain an advantage over dropout.
* multiplying the weights by $\mu\sim N(1,\sigma)$ (instead of a stochasitic binary vector) is another possible choice


### vanishing & exploding gradients in deep networks
Vanishing gradients result in slow convergence. A live code illustration for vanishing gradient problem can be found [here](resources/vanish.html).
Dealing with the vanishing gradients problem is still an open research question. Solutions include:
* making the networks shallower
* step-wise training
* specialized architectures that are designed to assist in gradient flow (e.g. LSTM & GRU)

Exploding gradients can be easily and effectively solved by gradient clipping.
### saturation and dead neurons
If your network does not train well, it is advisable to monitor the network for layers with many saturated or dead neurons.
It matters for non-linearities that flatten in higher regimes (softmax, sigmoid, tanh etc.) to normalize all inputs magnitude-wise (be cautious to check this when using word vectors). Large-magnitude inputs will cause the non-linearity units to saturate. At saturation regimes the gradient is close to zero. The model will not learn effectively as the error would not propagate.
Solutions for saturation:
* changing the initialization
* scaling the range of the input values
* changing the learning rate
* normalize the values in the saturated layer after the activation (effective but expensive in terms of gradient computation)

Layers with the ReLU activation cannot be saturated, but can “die” – most or all values are negative and thus clipped at zero for all inputs, resulting in a gradient of zero for that layer. This can happen after a large gradient update. 
Solutions for dead neurons:
* changing the initialization
* reducing the learning rate

### Issues for RNN
* Long range dependencies -> vanishing (or exploding) gradients: additive gated architectures (LSTM, GRU..)
* Increasing the size of the recurrent layer cost a quadratic slow down: deep RNN in both direction scales linearly
* Large vocabularies -> slow softmax calculations: factorising the softmax or sampling

### model ensemble
- train N models independently and average the output
- snapshot ensemble
- Polyak averaging: keep a moving average of the parameters and use that at test time

### reference
- [Stanford CS231n Lecture 6](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture6.pdf)
- [Stanford CS231n Lecture 7](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture7.pdf)
- [Stanford CS224d Lecture 6](http://cs224d.stanford.edu/lectures/CS224d-Lecture6.pdf)
- [Stanford CS224d Lecture 8](http://cs224d.stanford.edu/lectures/CS224d-Lecture8.pdf) ([Vanishing gradient example](http://cs224d.stanford.edu/notebooks/vanishing_grad_example.html))
- [Stanford CS224d Midterm Solutions](http://cs224d.stanford.edu/midterm/midterm_solutions.pdf)
- [Oxford Deep NLP 2017 Lecture 4](https://github.com/oxford-cs-deepnlp-2017/lectures/blob/master/Lecture%204%20-%20Language%20Modelling%20and%20RNNs%20Part%202.pdf)
- [Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)
- [A primer on neural network models for natural language processing](http://arxiv.org/pdf/1510.00726)
- [Deep Learning Book](http://www.deeplearningbook.org/)
- [deeplearning.ai Course 2: Hyperparameter tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network/home/welcome)

