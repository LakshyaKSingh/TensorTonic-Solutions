import math

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence.
    """
    n = len(actual_tokens)
    log_probs = []
    
    for i in range(n):
        # 1. Get the probability the model gave to the real word
        p_i = prob_distributions[i][actual_tokens[i]]
        
        # 2. Take the log of that probability
        log_probs.append(math.log(p_i))
    
    # 3. Calculate Cross-Entropy (Negative Mean)
    cross_entropy = - (sum(log_probs) / n)
    
    # 4. Return the exponential (Perplexity)
    return math.exp(cross_entropy)