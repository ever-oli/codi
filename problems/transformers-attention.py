import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Step 1: Get dimensions
    d_k = Q.size(-1)  # Key/query dimension
    
    # Step 2: Compute attention scores
    # Q @ K.transpose(-2, -1) gives shape: (batch, seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # Step 3: Scale scores by 1/√d_k
    # This prevents dot products from becoming too large
    scaled_scores = scores / math.sqrt(d_k)
    
    # Step 4: Apply softmax to get attention weights
    # Softmax is applied along the last dimension (keys dimension)
    attention_weights = F.softmax(scaled_scores, dim=-1)
    
    # Step 5: Compute weighted sum of values
    # attention_weights @ V gives shape: (batch, seq_len_q, d_v)
    output = torch.matmul(attention_weights, V)
    
    return output
    pass