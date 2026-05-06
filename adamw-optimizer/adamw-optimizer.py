import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    # Convert inputs to NumPy arrays to support vectorized operations
    w = np.array(w)
    m = np.array(m)
    v = np.array(v)
    grad = np.array(grad)
    
    # Step 1: Update First Moment (Exponential moving average of gradients)
    new_m = beta1 * m + (1 - beta1) * grad
    
    # Step 2: Update Second Moment (Exponential moving average of squared gradients)
    new_v = beta2 * v + (1 - beta2) * (grad ** 2)
    
    # Step 3: AdamW Parameter Update (Decoupled weight decay + adaptive gradient)
    new_w = w - lr * (weight_decay * w) - lr * (new_m / (np.sqrt(new_v) + eps))
    
    return new_w, new_m, new_v