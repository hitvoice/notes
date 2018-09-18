## Table of contents
<!--ts-->
   * [Convolutional Neural Network](#convolutional-neural-network)
   * To be continued...   
<!--te-->
## Projection
### dense layer/perceptron
$$
x'=\sigma(Wx+b)
$$
### highway layer
#### version 1
$$
\begin{align*}
z &= \operatorname{relu}(W_p x+b) \\\\
g &= \operatorname{sigmoid}(W_g x+b)\\\\
x'&=g\circ z + (1-g)\circ x
\end{align*}
$$
#### version 2
$$
\begin{align*}
z &= \tanh(W_1x + b_1) \\\\
r &= \operatorname{sigmoid}(W_2x + b_2)\\\\
f &= \operatorname{sigmoid}(W_3x + b_3)\\\\
x'&=r\circ x + f\circ z
\end{align*}
$$
### factorized bilinear layer
$$
x_i'=x^TF^TFx + Wx + b
$$
$F\in \mathcal{R}^{k\times d}$ ($k \ll d$). Dropout in factorized bilinear layer (DropFactor[9]): each **factor** is retained with a fixed probability $p$.
## Embedding of Discrete Variables
### Usage of pretrained embeddings
- fix
  - fix pretrained embeddings and train the outside-pretrained ones
  - fix pretrained embeddings and set others to zero (when pretrained embeddings have very high coverage)
  - fix pretrained embeddings and hash others to one of N random embeddings [4]
- make them trainable/finetunable
- fix embeddings and train a projection layer
- concat fixed and trainable embeddings [6]
- fix embeddings and train a highway layer [8]
### Handle unknown categories/out-of-vocabulary words
What are treated as OOV
- outside pretrained
- document frequency < threshold

What to do with them
- Set to zero (when unknown categories do not matter)
- Use one learnable vector as the embedding of OOV
- hash to one of N random embeddings
## Sequence Encoding
### Recurrent Neural Networks
#### Vanilla RNN
#### LSTM
#### GRU
### Convolutional Neural Networks
#### 1d Convolution
### Self-attention
### Positional Encoding
#### postional Embedding
#### Sine Wave-like
#### Distance-sensitive bias/mask
distances greater than N share the same bias/mask [4]
## Sequence Aggregation
### Pooling
Pooling operators have no learnable parameters.
#### Max pooling
pad with -inf
#### Mean pooling/summation
pad with zeros and / actual sequence length
### Aggregation models
#### Self-attention 
##### version 1
$$
\begin{align*}
\alpha_i &=\frac{\exp(Wx_i + b)}{\sum_i \exp(Wx_i + b)}\\\\
v &= \sum_i \alpha_i x_i
\end{align*}
$$
##### version 2
$$
\begin{align*}
h_i &= \tanh(Wx_i + b)\\\\
\alpha_i &=\frac{\exp(h_i^Tu)}{\sum_i \exp(h_i^Tu)}\\\\
v &= \sum_i \alpha_i x_i
\end{align*}
$$
$u$ is called context vector. You can have multiple context vectors to performed multi-view self-attention[7].
#### RNN
- RNN: last hidden vector
- BiRNN: concatenation of two last hidden vectors
- multi-layered RNN: concatenation of last hidden vectors in all layers
- biRNN + mean/max pooling
#### CNN
- Self-Adaptive Hierarchical Sentence Mdoel (AdaSent)
- Hierarchical ConvNet[7]: concatenation of the max pooling of each convolutional layer's feature maps
## Interaction
### Interaction of two vectors
- concatenation: [a, b]
- addition: a + b
- substraction: a - b, use |a - b| in symetric case
- element-wise multiplication: a \* b
- Element-wise Bilinear Matching [3] (discuss later)
#### Element-wise Bilinear Matching
### Interaction of two sequences

- attention score
  - dot product
  - cosine
  - concat-MLP
- attention activation
  - softmax
  - sparse attention [5]
## Meta Architecture
### Residual Connection
### Highway Connection
### Dense Connection
Input of next layer is the concatenation of outputs of all previous layers.




## Appendix
### Convolutional Neural Network
#### Motivation of convolution
Fully connected networks are computationally expensive when the number of input units is large. Locally connected networks, in which each hidden unit will connect to only a small contiguous region of pixels in the input, are feasible for high resolution images.
Natural images have the property of being ”stationary”, meaning that the statistics of one part of the image are the same as any other part. This suggests that the features that we learn at one part of the image can also be applied to other parts of the image, and we can use the same features at all locations.

#### Convolution
The dimension after convolution is
$$
n = \lfloor\frac{n_{\text{prev}} - k + 2\times \text{pad}}{\text{stride}}\rfloor + 1,
$$
where $k$ is the kernel size. The receptive field of each hidden units is 
$$
((k - 1)\times n_{\text{layer}} + 1)^2
$$
The shape of $W$ is (k, k, n_prev, n). The shape of $b$ is (1, 1, 1, n).
![Convolution_schematic.gif](resources/conv.gif)

If we apply "valid" padding, no padding is used. If we use "same" padding, the shapes before and after convolution are the same, which means left padding is $\lfloor\frac{k-1}{2}\rfloor$ and right padding is $\lfloor\frac{k}{2}\rfloor$. In 1d convolution where no future information should involve, one can set both left and right padding to $k-1$ and remove the last $k-1$ units after each convolution.

#### Motivation of pooling
* Features obtained by convolution is still computationally challenging
* Aggregating statistics of features at various locations is a natural way to describe a large image
* Pooled features are "translation invariant", which means the same pooled feature will be active even when the image undergoes (small) translations.

#### Pooling
After obtaining our convolved features as described earlier, we decide the size of the region, say $m\times n$ to pool our convolved features over. Then, we divide our convolved features into disjoint $m\times n$ regions, and take the mean (or maximum) feature activation over these regions to obtain the pooled convolved features. These pooled features can then be used for classification. In max pooling, "-inf" (instead of 0) is used as the padded value.

<img src="resources/pool.gif" width="600">

#### Architecture
A CNN consists of a number of convolutional and subsampling layers optionally followed by fully connected layers. The architecture of a CNN is designed to take advantage of the 2D structure of an input image (or other 2D input such as a speech signal). This is achieved with local connections and tied weights followed by some form of pooling which results in translation invariant features.

#### Back Propagation
- **propagate error through a pooling layer**: The weight $W$ is $\frac{1}{mn}$ in mean pooling and 1 for the maximun, 0 for others in max pooling. The bias is 0. 
- **update weights in a convolutional layer**: convolve the iuput of the convolutional layer with the incoming error.

## reference
- [1] http://deeplearning.stanford.edu/tutorial/
- [2] [Stanford CS224d Lecture 13](http://cs224d.stanford.edu/lectures/CS224d-Lecture13.pdf)
- [3] Element-wise Bilinear Sentence Matching [SEM18](http://aclweb.org/anthology/S18-2012)
- [4] A Decomposable Attention Model for Natural Language Inference
- [5] GLoMo: Unsupervisedly Learned Relational Graphs as Transferable Representations
- [6] Sematic Sentence Matching with Densely-connected Recurrent and Co-attentive Information
- [7] Supervised Learning of Universal Sentence Representations from Natural Language Inference Data
- [8] A Compare-Propagate Architecture with Alignment Factorization for Natural Language Inference
- [9] Factorized Bilinear Models for Image Recognition
