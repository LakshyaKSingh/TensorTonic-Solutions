import math

def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.
    """
    # Handle the edge case where the recommendation list is empty
    if not recommendations:
        return 0.0
    
    total_novelty = 0.0
    
    # Calculate novelty for each item
    for item in recommendations:
        popularity = item_counts[item] / n_users
        
        # Use base-2 logarithm for self-information
        novelty = -math.log2(popularity)
        total_novelty += novelty
        
    # Return the average novelty score
    return total_novelty / len(recommendations)