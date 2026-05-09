import math

def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    # 1. Compute the He uniform bound (L)
    # Formula: L = sqrt(6 / fan_in)
    limit = math.sqrt(6 / fan_in)
    
    # 2. Map each raw weight from [0, 1] to [-limit, limit]
    # Transformation: W_new = W_old * (2 * limit) - limit
    scaled_W = []
    for row in W:
        scaled_row = []
        for weight in row:
            scaled_weight = weight * (2 * limit) - limit
            scaled_row.append(scaled_weight)
        scaled_W.append(scaled_row)
        
    return scaled_W