def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    s1 = set(set_a)
    s2 = set(set_b)
    
    # Calculate the intersection and union
    intersection = s1.intersection(s2)
    union = s1.union(s2)
    
    # Handle the edge case where both sets are empty
    if not union:
        return 0.0
        
    # Return the Jaccard similarity coefficient
    return len(intersection) / len(union)
    