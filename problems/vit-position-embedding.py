import numpy as np

def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int) -> np.ndarray:
    """
    Add learnable position embeddings to patch embeddings.
    """
    # Create learnable position embeddings (simulated with random initialization)
    # In practice, these would be learned during training
    position_embeddings = np.random.randn(1, num_patches, embed_dim) * 0.01
    
    # Add position embeddings to patch embeddings (broadcasting over batch)
    output = patches + position_embeddings
    
    return output
    pass