import numpy as np

def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Layer normalization.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def classification_head(encoder_output: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Classification head for ViT.
    
    Extracts the [CLS] token (position 0) from the encoder output,
    applies layer normalization, then projects to class logits.
    
    Args:
        encoder_output: Output from ViT encoder of shape (batch, seq_len, embed_dim)
        num_classes: Number of output classes
    
    Returns:
        Classification logits of shape (batch, num_classes)
    """
    # Step 1: Extract [CLS] token at position 0
    # Shape: (batch, embed_dim)
    cls_token = encoder_output[:, 0, :]
    
    # Step 2: Apply layer normalization to [CLS] token
    cls_norm = layer_norm(cls_token)
    
    # Step 3: Linear projection to class logits
    # Initialize weights (would be learned in practice)
    embed_dim = cls_token.shape[-1]
    W = np.random.randn(embed_dim, num_classes) * 0.01
    b = np.zeros(num_classes)
    
    # Project to class logits
    logits = np.matmul(cls_norm, W) + b
    
    return logits