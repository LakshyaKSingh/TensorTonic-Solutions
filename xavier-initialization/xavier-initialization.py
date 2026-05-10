import math

def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    # Calculate the Xavier uniform bound
    limit = math.sqrt(6 / (fan_in + fan_out))
    
    obj = []
    for row in W:
        scaled_row = []
        for val in row:
            # Map [0, 1] to [-limit, limit]
            scaled_val = val * (2 * limit) - limit
            scaled_row.append(scaled_val)
        obj.append(scaled_row)
        
    return obj