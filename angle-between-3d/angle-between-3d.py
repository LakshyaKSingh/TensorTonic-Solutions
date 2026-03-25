import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # 1. Convert to numpy arrays for vector operations
    v = np.array(v)
    w = np.array(w)
    
    # 2. Compute norms
    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)
    
    # 3. Handle zero vectors (Hint 1)
    if norm_v < 1e-10 or norm_w < 1e-10:
        return np.nan
        
    # 4. Compute dot product and cosine
    cos_theta = np.dot(v, w) / (norm_v * norm_w)
    
    # 5. Clamp the value to [-1, 1] to avoid math errors (Hint 2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # 6. Return the angle in radians
    return np.arccos(cos_theta)