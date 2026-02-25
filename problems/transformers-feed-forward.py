import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
        # Step 1: First linear transformation: x @ W1 + b1
    # Input x shape: (batch, seq_len, d_model)
    # W1 shape: (d_model, d_ff)
    # Output shape: (batch, seq_len, d_ff)
    hidden = np.matmul(x, W1) + b1
    
    # Step 2: Apply ReLU activation
    # ReLU: max(0, hidden)
    # Same shape: (batch, seq_len, d_ff)
    relu_out = np.maximum(0, hidden)
    
    # Step 3: Second linear transformation: relu_out @ W2 + b2
    # relu_out shape: (batch, seq_len, d_ff)
    # W2 shape: (d_ff, d_model)
    # Output shape: (batch, seq_len, d_model)
    output = np.matmul(relu_out, W2) + b2
    
    return output
    pass