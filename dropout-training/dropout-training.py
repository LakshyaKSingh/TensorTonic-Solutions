import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x)
    
    # Handle the edge case where p is 0 (no dropout)
    if p == 0.0:
        return x, np.ones_like(x, dtype=float)
    
    # Use the provided rng or the default numpy random module
    if rng is not None:
        random_values = rng.random(x.shape)
    else:
        random_values = np.random.random(x.shape)
        
    # Create the binary mask: 1 if we keep the neuron, 0 if we drop it
    # We keep neurons with probability (1 - p)
    mask = (random_values < (1 - p)).astype(float)
    
    # Scale the mask by 1 / (1 - p) (Inverted Dropout)
    # This ensures the expected value of the output stays the same
    scale = 1 / (1 - p)
    dropout_pattern = mask * scale
    
    # Apply the pattern to the input
    output = x * dropout_pattern
    
    return output, dropout_pattern