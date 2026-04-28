import math

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    
    Args:
        predictions (list): Predicted probabilities (between 0 and 1)
        targets (list): True binary labels (0 or 1)
        alpha (float): Balancing factor
        gamma (float): Focusing parameter
        
    Returns:
        float: The mean focal loss across all samples
    """
    total_loss = 0
    n = len(predictions)
    
    for p, y in zip(predictions, targets):
        # 1. Compute p_t (probability of the true class)
        if y == 1:
            p_t = p
        else:
            p_t = 1 - p
            
        # 2. Compute the focal loss for this sample
        # Formula: FL = -alpha * (1 - p_t)^gamma * log(p_t)
        # We use a tiny epsilon to prevent log(0) if p_t is exactly 0
        p_t = max(p_t, 1e-15) 
        
        sample_loss = -alpha * ((1 - p_t) ** gamma) * math.log(p_t)
        total_loss += sample_loss
        
    # 3. Return the mean loss
    return total_loss / n