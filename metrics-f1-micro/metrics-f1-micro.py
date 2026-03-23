import numpy as np

def f1_micro(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # In micro-F1 for multi-class, TP is just the count of correct predictions
    tp = np.sum(y_true == y_pred)
    
    # Total number of samples
    n = len(y_true)
    
    # Every incorrect prediction is both a False Positive and a False Negative
    fp = n - tp
    fn = n - tp
    
    # Calculate F1
    # Use float() to ensure we don't return a numpy scalar
    f1 = float((2 * tp) / (2 * tp + fp + fn))
    
    return f1