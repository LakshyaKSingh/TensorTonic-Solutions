import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    """
    # Converting to arrays is necessary to handle list inputs
    x = np.asarray(x)
    y = np.asarray(y)
    
    # np.linalg.norm calculates the L2 norm of the difference vector
    return float(np.linalg.norm(x - y))