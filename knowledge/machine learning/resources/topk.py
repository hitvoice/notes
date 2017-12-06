import numpy as np
import pandas as pd

def top_k_hit(label, predict_proba, label_names, k):
    """
    Compute whether the true label occurs in top k predictions.
    input:
        label: np.array with shape (n, 1), true class label
        predict_proba: clf.predict_proba(X)
        label_names: clf.classes_, consistent with `predict_proba`
        k: scalar
    output:
        hit: np.array with 0 & 1. Top k accuracy = np.average(hit)
    """
    y = pd.get_dummies(label).loc[:, label_names].fillna(0).as_matrix()
    c = len(label_names) - k
    assert c > 0
    convert = lambda x: np.where(x < np.partition(x, c)[c], 0, 1) # cth min
    yp = np.apply_along_axis(convert, axis=1, arr=predict_proba)
    hit = np.sum(yp * y, axis=1)
    return hit