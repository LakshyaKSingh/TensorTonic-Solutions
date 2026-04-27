import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # Convert input to a numpy array for vectorized operations
    x = np.array(x)
    
    # Calculate PMF: if x is 1, prob is p; if x is 0, prob is 1-p
    # Using np.where as suggested in the hints
    pmf = np.where(x == 1, p, 1 - p)
    
    # Calculate mean and variance
    mean = float(p)
    var = float(p * (1 - p))
    
    return pmf, mean, var