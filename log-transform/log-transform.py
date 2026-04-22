import math

def log_transform(values):
    """
    Apply the log1p transformation to each value.
    """
    # math.log1p(x) computes ln(1 + x) accurately
    return [math.log1p(v) for v in values]