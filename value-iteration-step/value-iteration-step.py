def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    new_values = []
    num_states = len(values)
    
    # Iterate over each state s
    for s in range(num_states):
        best_q = float('-inf')
        num_actions = len(transitions[s])
        
        # Iterate over each action a available in state s
        for a in range(num_actions):
            # Calculate the expected future value: sum(T(s,a,s') * V(s'))
            expected_future = sum(
                transitions[s][a][s_next] * values[s_next] 
                for s_next in range(num_states)
            )
            
            # Calculate Q(s, a) = R(s, a) + gamma * expected_future
            q_value = rewards[s][a] + gamma * expected_future
            
            # Keep track of the maximum Q-value across all actions
            if q_value > best_q:
                best_q = q_value
                
        # The new value for state s is the maximum Q-value
        new_values.append(best_q)
        
    return new_values