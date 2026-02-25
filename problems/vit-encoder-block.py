import numpy as np

def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Layer normalization.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU activation (approximation).
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Softmax function.
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def multi_head_self_attention(x: np.ndarray, num_heads: int, embed_dim: int) -> np.ndarray:
    """
    Multi-Head Self-Attention (simplified).
    """
    batch, seq_len, _ = x.shape
    head_dim = embed_dim // num_heads
    
    # Simplified attention weights (in practice these would be learned)
    W_q = np.random.randn(embed_dim, embed_dim) * 0.02
    W_k = np.random.randn(embed_dim, embed_dim) * 0.02
    W_v = np.random.randn(embed_dim, embed_dim) * 0.02
    W_o = np.random.randn(embed_dim, embed_dim) * 0.02
    
    # Project to Q, K, V
    Q = np.matmul(x, W_q)
    K = np.matmul(x, W_k)
    V = np.matmul(x, W_v)
    
    # Reshape for multi-head
    Q = Q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    
    # Scaled dot-product attention
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    attn_weights = softmax(scores, axis=-1)
    attn_output = np.matmul(attn_weights, V)
    
    # Reshape back
    attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
    
    # Output projection
    return np.matmul(attn_output, W_o)

def mlp(x: np.ndarray, embed_dim: int, mlp_ratio: float) -> np.ndarray:
    """
    MLP with GELU activation.
    """
    hidden_dim = int(embed_dim * mlp_ratio)
    
    # Simplified MLP weights
    W1 = np.random.randn(embed_dim, hidden_dim) * 0.02
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, embed_dim) * 0.02
    b2 = np.zeros(embed_dim)
    
    # First linear layer + GELU
    h = gelu(np.matmul(x, W1) + b1)
    
    # Second linear layer
    return np.matmul(h, W2) + b2

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> np.ndarray:
    """
    ViT Transformer encoder block.
    
    Pre-LayerNorm formulation:
    x' = x + MSA(LN(x))
    x'' = x' + MLP(LN(x'))
    
    Args:
        x: Input tensor of shape (batch, seq_len, embed_dim)
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dimension to embed_dim (default: 4.0)
    
    Returns:
        Output tensor of same shape as input
    """
    # =========== First sub-block: Multi-Head Self-Attention ===========
    # Apply LayerNorm
    x_norm1 = layer_norm(x)
    
    # Apply Multi-Head Self-Attention
    attn_output = multi_head_self_attention(x_norm1, num_heads, embed_dim)
    
    # Add residual connection
    x = x + attn_output
    
    # =========== Second sub-block: MLP ===========
    # Apply LayerNorm
    x_norm2 = layer_norm(x)
    
    # Apply MLP
    mlp_output = mlp(x_norm2, embed_dim, mlp_ratio)
    
    # Add residual connection
    x = x + mlp_output
    
    return x