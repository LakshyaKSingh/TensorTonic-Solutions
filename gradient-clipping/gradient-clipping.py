import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.asarray(g)
    # Calculate the L2 norm of the entire array
    norm = np.linalg.norm(g)
    
    # If norm is 0, or within limit, OR if max_norm is non-positive, return original
    if norm <= max_norm or norm == 0 or max_norm <= 0:
        return g.copy()
    
    # Scale the gradients proportionally
    return g * (max_norm / norm)