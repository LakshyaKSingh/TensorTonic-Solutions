import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # 1. Validate input dimensions
    if x.ndim not in [3, 4]:
        raise ValueError("Input must be 3D (C, H, W) or 4D (N, C, H, W)")

    # 2. Identify spatial axes (always the last two: H and W)
    spatial_axes = (-2, -1)
    
    # 3. Compute mean across spatial dimensions
    # axis=(-2, -1) collapses H and W into their average
    return np.mean(x, axis=spatial_axes)