import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    # Create the embedding layer
    embedding = nn.Embedding(vocab_size, d_model)
    
    # Initialize weights with normal distribution
    # std = 1/sqrt(d_model) helps keep the scale of embeddings consistent
    nn.init.normal_(embedding.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
    
    return embedding

    pass

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    # Step 1: Look up embeddings for each token
    # This uses PyTorch's nn.Embedding which does a simple table lookup
    embedded = embedding(tokens)  # Shape: [..., d_model]
    
    # Step 2: Scale by sqrt(d_model) as per Transformer paper
    # This ensures embeddings have variance ~1, which helps with training stability
    scaled_embeddings = embedded * math.sqrt(d_model)
    
    return scaled_embeddings
    pass