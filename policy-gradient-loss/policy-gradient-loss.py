def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    T = len(rewards)
    returns = [0.0] * T
    
    # 1. Compute discounted returns backward
    # G_t = r_t + gamma * G_{t+1}
    last_return = 0.0
    for t in reversed(range(T)):
        returns[t] = rewards[t] + gamma * last_return
        last_return = returns[t]
        
    # 2. Subtract the mean return as a baseline
    mean_return = sum(returns) / T
    advantages = [G - mean_return for G in returns]
    
    # 3. Compute the loss
    # L = -1/T * sum(log_prob * advantage)
    total_loss = 0.0
    for lp, adv in zip(log_probs, advantages):
        total_loss += lp * adv
        
    return -total_loss / T