import numpy as np

def focal_loss(p, y, gamma=2.0):
    # Convert inputs to numpy arrays just in case they are passed as lists
    p = np.array(p)
    y = np.array(y)
    
    # 1. Numerical Stability (Hint 1)
    # This prevents log(0) which causes NaN errors
    p = np.clip(p, 1e-15, 1 - 1e-15)
    
    # 2. Positive term (for cases where y = 1)
    # We multiply by 'y' so this term becomes 0 when the label is 0
    pos_term = - (1 - p)**gamma * y * np.log(p)
    
    # 3. Negative term (for cases where y = 0)
    # We multiply by '(1 - y)' so this term becomes 0 when the label is 1
    neg_term = - p**gamma * (1 - y) * np.log(1 - p)
    
    # 4. Total Loss (Hint 3)
    # Sum the terms and return the average (mean) scalar value
    return np.mean(pos_term + neg_term)