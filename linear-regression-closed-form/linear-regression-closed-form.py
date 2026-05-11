import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Convert inputs to numpy arrays to ensure matrix operations work
    X = np.array(X)
    y = np.array(y)
    
    # Compute (X^T * X)^-1 * X^T * y
    # .T is transpose, np.linalg.inv handles the inverse
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    
    return w