def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # 1. Initialize an empty dictionary
    counts = {}
    
    # 2. Iterate through each sentence in the list
    for sentence in sentences:
        # 3. Iterate through each word (token) in the sentence
        for word in sentence:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
                
    # 4. Return the completed dictionary
    return counts