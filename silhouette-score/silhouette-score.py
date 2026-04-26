import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    
    # 1. Compute All-Pairs Euclidean Distances
    # Using the formula: dist(a, b)^2 = ||a||^2 + ||b||^2 - 2(a . b)
    dist_sq = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
    # Ensure no negative values due to precision errors before sqrt
    dist_matrix = np.sqrt(np.maximum(dist_sq, 0))
    
    s_i = np.zeros(n_samples)
    
    for i in range(n_samples):
        current_label = labels[i]
        
        # 2. a(i): Intra-cluster distance
        # Average distance to all other points in the same cluster
        same_cluster_mask = (labels == current_label)
        same_cluster_mask[i] = False  # Exclude the point itself
        
        if np.sum(same_cluster_mask) > 0:
            a_i = np.mean(dist_matrix[i, same_cluster_mask])
        else:
            # If the cluster has only one member, s(i) is typically defined as 0
            a_i = 0
            
        # 3. b(i): Nearest inter-cluster distance
        # Minimum of average distances to points in other clusters
        other_clusters = unique_labels[unique_labels != current_label]
        b_i = np.inf
        
        for other_label in other_clusters:
            other_cluster_mask = (labels == other_label)
            avg_dist_to_other = np.mean(dist_matrix[i, other_cluster_mask])
            b_i = min(b_i, avg_dist_to_other)
            
        # 4. s(i) = (b - a) / max(a, b)
        if max(a_i, b_i) > 0:
            s_i[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i[i] = 0

    return float(np.mean(s_i))