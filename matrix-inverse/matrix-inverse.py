import numpy as np

def matrix_inverse(A):
    # Convert input to a numpy array first
    A = np.array(A)
    
    # 1. Validate that input is a 2D array
    if A.ndim != 2:
        return None
        
    # 2. Validate that it is a square matrix
    rows, cols = A.shape
    if rows != cols:
        return None
    
    # 3. Check if the matrix is singular (determinant is 0)
    # Using a threshold for floating point stability
    if abs(np.linalg.det(A)) < 1e-10:
        return None
        
    # 4. Compute and return the inverse
    return np.linalg.inv(A)