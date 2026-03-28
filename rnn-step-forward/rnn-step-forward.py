import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Computes the next hidden state of an RNN cell.
    
    Returns:
    h_t: The new hidden state of shape (H,)
    """
    # 1. Calculate the contribution from the current input
    # 2. Calculate the contribution from the previous hidden state
    # 3. Add the bias and apply the tanh activation function
    
    # Using the @ operator for clean matrix/vector multiplication
    h_t = np.tanh(x_t @ Wx + h_prev @ Wh + b)
    
    return h_t