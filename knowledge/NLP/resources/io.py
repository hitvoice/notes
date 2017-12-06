import numpy as np
import scipy.sparse as sp

def save_sparse_csr(filename, array):
    assert filename.endswith('.npz')
    np.savez(filename, data=array.data, indices=array.indices,
        indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    assert filename.endswith('.npz')
    loader = np.load(filename)
    return sp.csr_matrix(
    	(loader['data'], loader['indices'], loader['indptr']),
        shape=loader['shape'])