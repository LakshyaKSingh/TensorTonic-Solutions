def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    # 1. Get dimensions
    H = len(image)
    W = len(image[0])
    kh = len(kernel)
    kw = len(kernel[0])
    
    # 2. Apply Zero Padding
    # Create a larger grid filled with zeros
    padded_H = H + 2 * padding
    padded_W = W + 2 * padding
    padded_img = [[0.0 for _ in range(padded_W)] for _ in range(padded_H)]
    
    # Fill the center of the padded grid with the original image
    for r in range(H):
        for c in range(W):
            padded_img[r + padding][c + padding] = float(image[r][c])
            
    # 3. Calculate Output Dimensions
    out_h = ((H + 2 * padding - kh) // stride) + 1
    out_w = ((W + 2 * padding - kw) // stride) + 1
    
    # Initialize the output grid
    output = [[0.0 for _ in range(out_w)] for _ in range(out_h)]
    
    # 4. Perform Convolution (Sliding Window)
    for i in range(out_h):
        for j in range(out_w):
            # Calculate the top-left corner of the current window
            start_r = i * stride
            start_c = j * stride
            
            # Element-wise multiplication and summation
            current_sum = 0.0
            for mr in range(kh):
                for mc in range(kw):
                    img_val = padded_img[start_r + mr][start_c + mc]
                    kern_val = kernel[mr][mc]
                    current_sum += img_val * kern_val
            
            output[i][j] = current_sum
            
    return output