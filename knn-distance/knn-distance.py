import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Convert inputs to numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    # Hint 3: Handle 1D arrays by reshaping to 2D (n, d)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
        
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Hint 1: Use broadcasting to compute all pairwise distances
    # Shape becomes (n_test, n_train, d)
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    # Euclidean distance: sqrt(sum of squared differences)
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Hint 2: Use np.argsort() to get indices of sorted distances
    # This gives us indices of training points sorted by distance for each test point
    sorted_indices = np.argsort(dist, axis=1)
    
    # Take the first k neighbors
    neighbor_indices = sorted_indices[:, :k]
    
    # Requirement: Handle k larger than training set size (pad with -1)
    if k > n_train:
        padding = np.full((n_test, k - n_train), -1)
        neighbor_indices = np.hstack([neighbor_indices, padding])
        
    return neighbor_indices.astype(int)