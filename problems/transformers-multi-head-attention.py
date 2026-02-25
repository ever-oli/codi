import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Get dimensions
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads  # Dimension per head
    
    # 1. Linear projections for Q, K, V
    Q_proj = np.matmul(Q, W_q)  # (batch, seq_len, d_model)
    K_proj = np.matmul(K, W_k)  # (batch, seq_len, d_model)
    V_proj = np.matmul(V, W_v)  # (batch, seq_len, d_model)
    
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
    # (batch, num_heads, seq_len, seq_len)
    scores = np.matmul(Q_trans, K_trans.transpose(0, 1, 3, 2))
    
    # Scale scores
    scaled_scores = scores / np.sqrt(d_k)
    
    # Apply softmax to get attention weights
    attention_weights = softmax(scaled_scores, axis=-1)
    
    # Apply attention to values
    # (batch, num_heads, seq_len, d_k)
    head_outputs = np.matmul(attention_weights, V_trans)
    
    # 5. Concatenate heads
    # Transpose back to (batch, seq_len, num_heads, d_k)
    head_outputs_trans = head_outputs.transpose(0, 2, 1, 3)
    
    # Reshape to (batch, seq_len, d_model)
    concatenated = head_outputs_trans.reshape(batch_size, seq_len, d_model)
    
    # 6. Final output projection
    output = np.matmul(concatenated, W_o)
    
    return output
    pass