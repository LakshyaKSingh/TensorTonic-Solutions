import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    # Ensure x is a numpy array
    x = np.array(x)
    n = len(x)
    
    # Initialize the random number generator
    if rng is None:
        rng = np.random.default_rng()

    # 1. Generate bootstrap samples by picking random indices with replacement
    # We create a (n_bootstrap, n) matrix of indices at once for efficiency
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    resamples = x[indices]
    
    # 2. Compute the mean for each bootstrap sample
    boot_means = np.mean(resamples, axis=1)
    
    # 3. Calculate the confidence interval bounds
    alpha = 1 - ci
    lower_percentile = alpha / 2
    upper_percentile = 1 - (alpha / 2)
    
    # Use np.quantile to find the values at these percentiles
    lower, upper = np.quantile(boot_means, [lower_percentile, upper_percentile])
    
    return boot_means, lower, upper