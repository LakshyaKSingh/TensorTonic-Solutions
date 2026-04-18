def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Get input dimensions
    H = len(X)
    W = len(X[0])
    
    # 1. Compute the output dimensions
    # H_out = floor((H - pool_size) / stride) + 1
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    
    # Initialize the output matrix with zeros
    # Using a list of lists as per the requirement
    res = [[0] * out_w for _ in range(out_h)]
    
    # 2. Iterate through each output position (i, j)
    for i in range(out_h):
        for j in range(out_w):
            # Calculate the starting boundaries of the window in X
            start_i = i * stride
            start_j = j * stride
            
            # Extract the window and find the maximum value
            # Initialize max_val with the first element of the window
            max_val = X[start_i][start_j]
            
            for a in range(pool_size):
                for b in range(pool_size):
                    current_val = X[start_i + a][start_j + b]
                    if current_val > max_val:
                        max_val = current_val
            
            res[i][j] = max_val
            
    return res