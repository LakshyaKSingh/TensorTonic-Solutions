import math

def ndcg(relevance_scores, k):
    # 1. Handle k being larger than the available scores
    k = min(k, len(relevance_scores))
    
    # 2. Define a helper function to calculate DCG
    def calculate_dcg(scores):
        dcg_sum = 0.0
        for i in range(k):
            # Use the formula: (2^rel - 1) / log2(i + 2)
            # (i + 2) because i is 0-indexed, so position 1 is i+1+1
            gain = (2**scores[i]) - 1
            discount = math.log2(i + 2)
            dcg_sum += gain / discount
        return dcg_sum

    # 3. Calculate Actual DCG
    actual_dcg = calculate_dcg(relevance_scores)
    
    # 4. Calculate Ideal DCG (IDCG)
    # Sort the available relevance scores in descending order
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = calculate_dcg(ideal_relevance)
    
    # 5. Normalize and handle the zero-case edge case
    if idcg == 0:
        return 0.0
    
    return actual_dcg / idcg