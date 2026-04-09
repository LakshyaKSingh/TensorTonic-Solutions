import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    x=np.array(x)
    # if x>=0:
    #     return x
    # else:
    #     return alpha*x
    return np.where(x>=0,x,alpha*x)