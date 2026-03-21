def get_median(arr):
    n = len(arr)
    if n == 0: return 0
    mid = n // 2
    if n % 2 == 1:
        return float(arr[mid])
    else:
        return (arr[mid - 1] + arr[mid]) / 2.0

def robust_scaling(values):
    if not values:
        return []
    if len(values) == 1:
        return [0.0]
    
    # 1. Sort the data
    sorted_val = sorted(values)
    n = len(sorted_val)
    
    # 2. Find the Median
    median = get_median(sorted_val)
    
    # 3. Determine Lower and Upper halves
    # For n=5, mid is 2. Lower: [0,1], Upper: [3,4]
    # For n=4, mid is 2. Lower: [0,1], Upper: [2,3]
    mid_idx = n // 2
    if n % 2 == 1:
        lower_half = sorted_val[:mid_idx]
        upper_half = sorted_val[mid_idx+1:]
    else:
        lower_half = sorted_val[:mid_idx]
        upper_half = sorted_val[mid_idx:]
        
    # 4. Find Q1 and Q3
    q1 = get_median(lower_half)
    q3 = get_median(upper_half)
    iqr = q3 - q1
    
    # 5. Apply the scaling formula
    scaled_values = []
    for x in values:
        if iqr == 0:
            scaled_values.append(float(x - median))
        else:
            scaled_values.append(float(x - median) / iqr)
            
    return scaled_values