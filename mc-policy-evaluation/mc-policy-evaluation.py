import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    # 1. Initialize storage
    # Sum of returns for each state
    sum_returns = np.zeros(n_states)
    # Number of times each state was the 'first visit' in an episode
    counts = np.zeros(n_states)
    
    for episode in episodes:
        # 2. Calculate returns (G) backward for efficiency
        # An episode is a list of (state, reward)
        # We need to track which states we've already seen in THIS episode
        visited_in_episode = set()
        G = 0
        
        # We iterate backward: from the last step to the first
        # This allows us to use the recursive formula: G = r + gamma * G
        for i in range(len(episode) - 1, -1, -1):
            state, reward = episode[i]
            
            # Update the return G for the current step
            # Note: The return for step 'i' includes the reward from step 'i'
            G = reward + gamma * G
            
            # 3. First-visit check
            # We only care about the return if this is the FIRST time 
            # we see this state in this specific episode.
            # Since we are moving BACKWARD, the 'first visit' is actually 
            # the LAST one we encounter in our loop. 
            # To handle this, we check if the state appears in the steps 
            # EARLIER than 'i' in the episode.
            
            # Easier way: Check if state exists in episode[:i]
            # If it's not in the earlier part, then index 'i' is the FIRST visit.
            states_in_earlier_steps = [step[0] for step in episode[:i]]
            
            if state not in states_in_earlier_steps:
                sum_returns[state] += G
                counts[state] += 1

    # 4. Calculate Final Value V(s) = Average Return
    # Avoid division by zero for unvisited states
    V = np.zeros(n_states)
    nonzero_counts = counts > 0
    V[nonzero_counts] = sum_returns[nonzero_counts] / counts[nonzero_counts]
    
    return V