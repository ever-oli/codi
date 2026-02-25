import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
        # Step 1: Create position indices [0, 1, 2, ..., seq_length-1]
    position = np.arange(seq_length)[:, np.newaxis]  # Shape: (seq_length, 1)
    
    # Step 2: Create dimension indices [0, 2, 4, ...] for the division term
    # We use 2i for even indices
    i = np.arange(0, d_model, 2)  # Shape: (d_model/2,)
    
    # Step 3: Calculate the division term: 10000^(2i/d_model)
    # Using exponentials for numerical stability
    div_term = np.exp(i * (-np.log(10000.0) / d_model))  # Shape: (d_model/2,)
    
    # Step 4: Initialize PE matrix
    pe = np.zeros((seq_length, d_model))
    
    # Step 5: Apply sine to even indices
    pe[:, 0::2] = np.sin(position * div_term)
    
    # Step 6: Apply cosine to odd indices
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe
    pass