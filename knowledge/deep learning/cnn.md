# Convolutional Neural Network

### Motivation of convolution
Fully connected networks are computationally expensive when the number of input units is large. Locally connected networks, in which each hidden unit will connect to only a small contiguous region of pixels in the input, are feasible for high resolution images.
Natural images have the property of being ”stationary”, meaning that the statistics of one part of the image are the same as any other part. This suggests that the features that we learn at one part of the image can also be applied to other parts of the image, and we can use the same features at all locations.
### Convolution
Formally, given some large $r \times c$ images $x_{large}$, we first train a sparse autoencoder on small $a \times b$ patches $x_{small}$ sampled from these images, learning $k$ features $f = \sigma(W^{(1)}x_{small} + b^{(1)})$ (where $\sigma$ is the sigmoid function), given by the weights $W^{(1)}$ and biases $b^{(1)}$ from the visible units to the hidden units. For every $a \times b$ patch $x_s$ in the large image, we compute $f_s = \sigma(W^{(1)}x_s + b^{(1)})$, giving us $f_{convolved}$, a $k \times (r - a + 1) \times (c - b + 1)$ array of convolved features.
![Convolution_schematic.gif](resources/conv.gif)

### Motivation of pooling
* Features obtained by convolution is still computationally challenging
* Aggregating statistics of features at various locations is a natural way to describe a large image
* Pooled features are "translation invariant", which means the same pooled feature will be active even when the image undergoes (small) translations.

### Pooling
After obtaining our convolved features as described earlier, we decide the size of the region, say $m\times n$ to pool our convolved features over. Then, we divide our convolved features into disjoint $m\times n$ regions, and take the mean (or maximum) feature activation over these regions to obtain the pooled convolved features. These pooled features can then be used for classification.
![Pooling_schematic.gif](resources/pool.gif)

### Architecture
A CNN consists of a number of convolutional and subsampling layers optionally followed by fully connected layers. The architecture of a CNN is designed to take advantage of the 2D structure of an input image (or other 2D input such as a speech signal). This is achieved with local connections and tied weights followed by some form of pooling which results in translation invariant features.

### Back Propagation
**propagate error through a pooling layer**
The weight $W$ is $\frac{1}{mn}$ in mean pooling and 1 for the maximun, 0 for others in max pooling. The bias is 0. 
**update weights in a convolutional layer**
convolve the iuput of the convolutional layer with the incoming error.

### reference
- http://deeplearning.stanford.edu/tutorial/
- [Stanford CS224d Lecture 13](http://cs224d.stanford.edu/lectures/CS224d-Lecture13.pdf)