# BNS Scaling

Limitations:

* for binary classification/OVR/OVO only
* using binary-BNS, thus suitable for documents that are not too long

Advantage:

* take class label into consideration, correct the inappropriate scaling by IDF
* better than TF-IDF in most benchmarks

Pratical advice:

* using lookup table can improve efficiency a lot. With 100k dimensions, a 1e-4-precision lookup table can reduce nearly 50% of the running time
* when using SVM as the classifier, using boolean data type and encode the feature weight in a kernel can be more effiencient in some SVM implementations.

```python
import numpy as np
import scipy.sparse as sp
from scipy.stats import norm

def BNS(X, y, lookup=None):
    """
    Performs BNS scaling.
    Args: 
      X:      sp.sparse with shape (nDocs, V), the count matrix of the corpus
      y:      np.array with shape (nDocs,), binary class label, with value 0 or 1
      lookup: if not None, define the precision of the lookup table
    Returns: 
      np.array with shape (nDocs, V), BNS scaled feature
    """
    assert np.array_equal(np.unique(y), np.array([0,1]))
    nrows, ncols = X.shape
    assert (nrows,) == y.shape
    X = sp.csr_matrix(X, copy=True)
    X = X.sign() 
    Xp = sp.csc_matrix(X[y == 1])
    Xn = sp.csc_matrix(X[y == 0])
    tpr = np.array(Xp.mean(axis=0))[0]
    fpr = np.array(Xn.mean(axis=0))[0]
    if lookup is not None:
        assert lookup < 1 and lookup > 0
        table = np.arange(0, 1 + lookup, lookup)
        table[0] = 5e-4
        table[-1] = 1 - 5e-4
        table = norm.ppf(table)
        def ppf(x):
            return table[np.floor(x/lookup).astype(int)]
        bns = np.abs(ppf(tpr) - ppf(fpr))
    else:
        bns = np.abs(norm.ppf(tpr) - norm.ppf(fpr))
    bns_diag = sp.spdiags(bns, diags=0, m=ncols, n=ncols, format='csc')
    return X.dot(bns_diag)
```

### Reference

* [BNS Feature Scaling: An Improved Representation over TF·IDF for SVM Text Classification](http://www.hpl.hp.com/techreports/2007/HPL-2007-32R1.pdf)