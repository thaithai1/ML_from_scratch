import numpy as np

def vstack_one(X):
    """
    Stack a vector (1, 1, ..., 1) on top
    """
    return np.vstack((np.ones((1,X.shape[1])),X))