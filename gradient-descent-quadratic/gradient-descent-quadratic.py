def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0  # Start at the initial point
    for i in range(steps):
        # 1. Calculate the gradient (slope) at current x
        gradient = 2 * a * x + b
        
        # 2. Update x: move opposite to the gradient
        x = x - (lr * gradient)
        
    return float(x)