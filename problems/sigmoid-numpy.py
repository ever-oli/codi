import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    
    Args:
        x: Input scalar, list, or NumPy array.
        
    Returns:
        NumPy array of floats containing the sigmoid of x.
    """
    # Convert input to a numpy array of floats (works for scalars, lists, etc.)
    x_arr = np.asarray(x, dtype=float)
    
    # Compute sigmoid
    return 1.0 / (1.0 + np.exp(-x_arr))