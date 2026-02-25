import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    """
    batch_size = patches.shape[0]
    
    # Initialize learnable [CLS] token (would be learned during training)
    # Shape: (1, 1, embed_dim) for easy broadcasting
    cls_token = np.random.randn(1, 1, embed_dim) * 0.02
    
    # Broadcast [CLS] token to match batch size
    # Shape: (batch_size, 1, embed_dim)
    cls_token_batch = np.repeat(cls_token, batch_size, axis=0)
    
    # Concatenate [CLS] token at the beginning of the sequence (axis=1)
    output = np.concatenate([cls_token_batch, patches], axis=1)
    
    return output
    pass