import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    # Compute mean across feature dimension
    mean = np.mean(x, axis=-1, keepdims=True)
    
    # Compute variance across feature dimension
    variance = np.var(x, axis=-1, keepdims=True)
    
    # Normalize
    x_normalized = (x - mean) / np.sqrt(variance + eps)
    
    # Apply scale and shift
    output = gamma * x_normalized + beta
    
    return output
    pass

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Get dimensions
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads  # Dimension per head
    
    # 1. Linear projections
    Q_proj = np.matmul(Q, W_q)  # (batch, seq_len, d_model)
    K_proj = np.matmul(K, W_k)
    V_proj = np.matmul(V, W_v)
    
    # 2. Reshape to separate heads
    # (batch, seq_len, num_heads, d_k)
    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k)
    
    # 3. Transpose to get (batch, num_heads, seq_len, d_k)
    Q_trans = Q_heads.transpose(0, 2, 1, 3)
    K_trans = K_heads.transpose(0, 2, 1, 3)
    V_trans = V_heads.transpose(0, 2, 1, 3)
    
    # 4. Scaled dot-product attention for all heads
    # Compute attention scores: Q @ K^T
    scores = np.matmul(Q_trans, K_trans.transpose(0, 1, 3, 2))  # (batch, num_heads, seq_len, seq_len)
    
    # Scale scores
    scaled_scores = scores / np.sqrt(d_k)
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scaled_scores, axis=-1)
    
    # Apply attention to values
    head_outputs = np.matmul(attention_weights, V_trans)  # (batch, num_heads, seq_len, d_k)
    
    # 5. Concatenate heads
    # Transpose back to (batch, seq_len, num_heads, d_k)
    head_outputs_trans = head_outputs.transpose(0, 2, 1, 3)
    
    # Reshape to (batch, seq_len, d_model)
    concatenated = head_outputs_trans.reshape(batch_size, seq_len, d_model)
    
    # 6. Output projection
    output = np.matmul(concatenated, W_o)
    
    return output
    pass

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # First linear layer: expand to d_ff
    hidden = np.matmul(x, W1) + b1
    
    # ReLU activation
    relu_out = np.maximum(0, hidden)
    
    # Second linear layer: project back to d_model
    output = np.matmul(relu_out, W2) + b2
    
    return output
    pass

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
     # Part 1: Multi-Head Attention with Residual Connection and Layer Norm
    
    # Apply multi-head attention (self-attention: Q=K=V=x)
    attn_output = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    
    # Add residual connection
    x_attn_residual = x + attn_output
    
    # Apply layer normalization
    x_norm1 = layer_norm(x_attn_residual, gamma1, beta1)
    
    # Part 2: Feed-Forward Network with Residual Connection and Layer Norm
    
    # Apply feed-forward network
    ff_output = feed_forward(x_norm1, W1, b1, W2, b2)
    
    # Add residual connection
    x_ff_residual = x_norm1 + ff_output
    
    # Apply layer normalization
    output = layer_norm(x_ff_residual, gamma2, beta2)
    
    return output
    pass