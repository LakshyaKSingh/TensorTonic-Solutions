import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # 1. Initialize a vector of zeros with the length of the vocabulary
    bow_vec = np.zeros(len(vocab), dtype=int)
    
    # 2. Create a mapping of word to its index in the vocabulary for fast lookup
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    # 3. Iterate through the tokens and increment the count at the correct index
    for token in tokens:
        if token in word_to_idx:
            idx = word_to_idx[token]
            bow_vec[idx] += 1
            
    return bow_vec