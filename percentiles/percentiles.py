import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    # Convert inputs to NumPy arrays as per requirement
    x_arr = np.asarray(x)
    q_arr = np.asarray(q)
    
    # Compute the percentiles using the requested linear interpolation method
    return np.percentile(x_arr, q_arr, method='linear')