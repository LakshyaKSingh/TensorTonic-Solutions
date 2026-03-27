import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    # Ensure inputs are numpy arrays
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)

    # 1. Determine the axes to reduce
    axis = (0, 2, 3) if x.ndim == 4 else 0
    
    # 2. Calculate mean and variance
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    
    # 3. Normalize
    x_hat = (x - mean) / np.sqrt(var + eps)
    
    # 4. Scale and shift
    if x.ndim == 4:
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
        
    out = gamma * x_hat + beta
    return out