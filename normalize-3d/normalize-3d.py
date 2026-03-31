import numpy as np

def normalize_3d(v):
    # Ensure input is a numpy array of floats to avoid dtype casting errors
    v = np.array(v, dtype=np.float64)
    
    # Calculate magnitude (Hint 2: use keepdims=True for broadcasting)
    mag = np.linalg.norm(v, axis=-1, keepdims=True)
    
    # Hint 1: Check if mag > 1e-10 to avoid division by zero
    # We use np.divide with 'where' to handle this safely
    return np.divide(v, mag, out=np.zeros_like(v), where=mag > 1e-10)