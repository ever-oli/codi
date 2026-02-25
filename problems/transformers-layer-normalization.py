import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.

    Args:
        x: Input array of shape (..., d_model)
        gamma: Scale parameter of shape (d_model,)
        beta: Shift parameter of shape (d_model,)
        eps: Small constant for numerical stability

    Returns:
        Normalized array of same shape as x
    """
     # Step 1: Compute mean across the feature dimension (last axis)
    # keepdims=True preserves the dimension for broadcasting
    mean = np.mean(x, axis=-1, keepdims=True)
    
    # Step 2: Compute variance across the feature dimension
    variance = np.var(x, axis=-1, keepdims=True)
    
    # Step 3: Normalize the input
    # x_normalized = (x - mean) / sqrt(variance + eps)
    x_normalized = (x - mean) / np.sqrt(variance + eps)
    
    # Step 4: Apply scale (gamma) and shift (beta)
    # gamma and beta are broadcasted across all dimensions
    output = gamma * x_normalized + beta
    
    return output
    pass