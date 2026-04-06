import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    
    Args:
        p: array-like - Predicted probabilities, shape (N,) or (H,W)
        y: array-like - Binary mask {0,1}, same shape as p
        eps: float - Smoothing epsilon for numerical stability
    """
    # Convert inputs to numpy arrays and flatten to handle both 1D and 2D
    p = np.asarray(p).flatten()
    y = np.asarray(y).flatten()
    
    # Calculate intersection: sum of element-wise product
    intersection = np.sum(p * y)
    
    # Calculate sums of both arrays for the denominator
    sum_p = np.sum(p)
    sum_y = np.sum(y)
    
    # Compute Dice Coefficient with epsilon for stability
    dice_coeff = (2. * intersection + eps) / (sum_p + sum_y + eps)
    
    # Dice Loss is 1 minus the Dice Coefficient
    return 1.0 - dice_coeff