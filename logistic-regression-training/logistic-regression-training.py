import numpy as np

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    n_samples, n_features = X.shape
    
    # 1. Initialize weights and bias
    w = np.zeros(n_features)
    b = 0.0
    
    for _ in range(steps):
        # 2. Forward pass: Linear combination + Sigmoid
        z = np.dot(X, w) + b
        p = _sigmoid(z)
        
        # 3. Calculate gradients
        # (p - y) is the error vector
        errors = p - y
        dw = (1 / n_samples) * np.dot(X.T, errors)
        db = (1 / n_samples) * np.sum(errors)
        
        # 4. Update parameters
        w -= lr * dw
        b -= lr * db
        
    return w, b