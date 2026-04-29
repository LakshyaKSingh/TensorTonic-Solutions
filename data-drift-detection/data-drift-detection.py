def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    # 1. Normalize reference counts
    total_ref = sum(reference_counts)
    p = [count / total_ref for count in reference_counts]
    
    # 2. Normalize production counts
    total_prod = sum(production_counts)
    q = [count / total_prod for count in production_counts]
    
    # 3. Compute Total Variation Distance (TVD)
    tvd = 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))
    
    # 4. Check if drift is strictly greater than the threshold
    drift_detected = tvd > threshold
    
    return {
        "score": float(tvd),
        "drift_detected": drift_detected
    }