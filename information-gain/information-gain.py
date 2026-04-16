import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    # 1. Convert to numpy arrays just in case
    y = np.asarray(y)
    split_mask = np.asarray(split_mask)
    n_total = len(y)
    
    # 2. Calculate parent entropy
    h_parent = _entropy(y)
    
    # 3. Create the subsets
    y_left = y[split_mask]
    y_right = y[~split_mask]
    
    # 4. Handle edge cases (empty side)
    if y_left.size == 0 or y_right.size == 0:
        return 0.0
    
    # 5. Calculate children entropy
    h_left = _entropy(y_left)
    h_right = _entropy(y_right)
    
    # 6. Calculate weighted average of children entropy
    # (n_left / n_total) * h_left + (n_right / n_total) * h_right
    n_left = y_left.size
    n_right = y_right.size
    weighted_h_children = (n_left / n_total) * h_left + (n_right / n_total) * h_right
    
    # 7. Return Information Gain
    return float(h_parent - weighted_h_children)
