import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1, +1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    # Convert inputs to numpy arrays to ensure vectorized operations
    y_true = np.asanyarray(y_true)
    y_score = np.asanyarray(y_score)
    
    # Calculate element-wise hinge loss: max(0, m - y * s)
    losses = np.maximum(0, margin - y_true * y_score)
    
    # Apply reduction
    if reduction == "mean":
        return float(np.mean(losses))
    elif reduction == "sum":
        return float(np.sum(losses))
    else:
        raise ValueError("reduction must be either 'mean' or 'sum'")