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
### neural arithmetic logic units
$$
\begin{align*}
W &= \tanh(\hat W) \odot \sigma(\hat M) \\\\
a &= Wx\\\\
m &= \exp W(\log(|x|+\epsilon))\\\\
g &= \sigma(Gx)\\\\
y &= g\odot a + (1 - g) \odot m
\end{align*}
$$
Elements of $W$ are biased to be close to −1, 0, or 1. $W$ for $a$ performs addition or substraction, while for $m$ it operates in log space and is therefore capable of learning multiplication, division and power functions. NALU functions in a way that extrapolates to numbers outside of the range observed during training [16].

### Maxout Networks
See [maxout networks](https://github.com/hitvoice/notes/blob/master/knowledge/deep%20learning/activation.md#maxout-networks).
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

When there're duplicated words in pretrained embeddings after some kind of normalization (conversion to lower case, NFD normalization, etc.):
- Use the embedding of the most frequent one
- Use the average of them [17]
- Use the one that strictly match the form after normalization

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
#### LSTM
TBD.
#### GRU
TBD.
### Convolutional Neural Networks
1D convolution. See [Appendix](#convolutional-neural-network).

### Disconnected Recurrent Neural Networks
The state at each step only depends on the previous $k-1$ words and the current word. This method incorporate the position invariance into RNN by disconnecting the information flow[13].

### Self-attention/Transformer
TBD.
### Positional Encoding
Positional encodings are useful for CNNs and attention based models, giving them a sense of position.
#### postional Embedding
Embed the obsolute position, and then concatenated or added [12] to the sequence encoding.
#### sinusoidal positional encoding
$$
\begin{align*}
E_{(p, 2i)} & = \sin(p/10000^{2i/d})\\
E_{(p, 2i+1)} & = \cos(p/10000^{2i/d})
\end{align*}
$$
where $p$ is the position and $i$ is is one of the hidden dimensions. The postional encoding is added to the sequence [19].
#### Distance-sensitive weights/bias
- multiply by hand-designed linearly decayed weight based on the distance [20]
- add learnable distance-sensitive bias, where distances greater than N share the same value [4]

## Sequence Aggregation
### Pooling
Pooling operators have no learnable parameters.
#### Max pooling
pad with -inf.
#### Mean pooling/summation
Pad with zeros, sum up and divide by actual sequence length
#### Min pooling
Mentioned in [14]. Pad with inf.
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
- Element-wise Bilinear Matching [3]
- Neural tensor network [18]
$$
s(v_1, v_2) = u^T \tanh \left(
v_1^T M^{[1:k]}v_2 + V
\begin{bmatrix}
    v_1 \\
    v_2
\end{bmatrix}
+b
\right)
$$
$M^{[1:k]}\in \mathcal R^{d\times d\times k}$ is a tensor and the bilinear tensor product $v_1^T M^{[1:k]}v_2$ results in a vector $h\in \mathcal R^k$, where each entry is computed by one slice of the tensor $h_i=v_1^TM^iv_2$.
### Interaction of two sequences
There're usually 4 steps in the interaction of 2 sequences, namely computing attention score, computing attention activation/normalization, computing the weighted average, and fusing the attended information with the original one. 

- attention score
  - inner product
  - complex-valued inner product (Hermitian product) [11]
  - cosine
  - concat-MLP (really slow and consuming huge memory, not recommended)
  - bilinear [10]
  - element-wise product + MLP [10]
  - substraction + MLP [10]
- attention activation/normalization
  - softmax
  - sparse attention (square-relu) [5]
- pooling
  - weighted average/alignment pooling
  - extractive pooling [11]: softmax(max(score, dim=1)) * a
- fusion: can use position-wise projection or sequence encoding here

If attention score is the inner product, the product should be scale by $\sqrt{1/d}$ to counteract a change in variance under the assumption that the two vectors are independent random variables. Softmax with large values will have extremely small gradients. \[19\]

The weighted sum can be scaled by $\sqrt{1/L}$ to counteract a change in variance \[12\]. Under the assumption that the attention scores are uniformly distributed (which is generally not the case but found to work well in some cases), the weighted sum can be scaled by $L\sqrt{1/L}$ [12].
## Meta Architecture
### Residual Connection
$$
x_k = h(x_{k-1}) + x_{k-1}
$$
The summation can be multiplied by $\sqrt{0.5}$ to halve the variance, assuming that both summands have the same variance which is not always true but effective in some cases [12]. 
### Highway Connection
See [highway layer](#highway-layer)
### Dense Connection
Input of next layer is the concatenation of outputs of all previous layers.
### Weighted Connection
$$
y = \gamma\sum_{l=0}^L\alpha_l h_l
$$
$\alpha$ are softmax-normalized weights and $\gamma$ allows the model to scale the entire vector [15].



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

If we apply "valid" padding, no padding is used. If we use "same" padding, the shapes before and after convolution are the same, which means left padding is $\lfloor\frac{k-1}{2}\rfloor$ and right padding is $\lfloor\frac{k}{2}\rfloor$. 

In 1d convolution where no future information should involve, one can set both left and right padding to $k-1$ and remove the last $k-1$ units after each convolution [12]. 

With kernel width $k$ and number of layers $l$, the network has a input field (perception field) of size $(k-1)\times l + 1$

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
- [10] Multiway Attention Networks for Modeling Sentence Pairs
- [11] Hermitian Co-Attention Networks for Text Matching in Asymmetrical Domains
- [12] Convolutional Sequence to Sequence Learning
- [13] [Disconnected Recurrent Neural Networks for Text Categorization](http://aclweb.org/anthology/P18-1215)
- [14] Learned in Translation: Contextual Word Vecotrs
- [15] Deep Contextualized word representations
- [16] Neural Arithmetic Logic Units
- [17] Reading Wikipedia to Answer Open-Domain Questions
- [18] [Reasoning With Neural Tensor Networks for Knowledge Base Completion](https://nlp.stanford.edu/pubs/SocherChenManningNg_NIPS2013.pdf)
- [19] Attention is All You Need
- [20] [Multi-Entity Aspect-Based Sentiment Analysis with Context, Entity and Aspect Memory](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17036/16171)
