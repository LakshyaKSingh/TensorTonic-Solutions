def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    import numpy as np
    
    # Convert inputs to numpy arrays for easier indexing
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    
    # Initialize ECE
    ece = 0.0
    
    # 1. Define bin boundaries
    # Equal-width bins over [0, 1]
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        # Define bin range
        bin_low = bin_boundaries[i]
        bin_high = bin_boundaries[i + 1]
        
        # 2. Assign predictions to bins
        # Handle the [0, 1) range and the edge case where p = 1.0
        if i == n_bins - 1:
            # Last bin includes 1.0
            bin_indices = np.where((y_pred >= bin_low) & (y_pred <= bin_high))[0]
        else:
            bin_indices = np.where((y_pred >= bin_low) & (y_pred < bin_high))[0]
            
        bin_size = len(bin_indices)
        
        # 3. Calculate Bin Accuracy and Confidence
        if bin_size > 0:
            # Accuracy: fraction of positive labels in the bin
            bin_acc = np.mean(y_true[bin_indices])
            
            # Confidence: average predicted probability in the bin
            bin_conf = np.mean(y_pred[bin_indices])
            
            # 4. Weighted Absolute Difference
            # ECE += (bin_size / total_samples) * |acc - conf|
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)
            
    return float(ece)