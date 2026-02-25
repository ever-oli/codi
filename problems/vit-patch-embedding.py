import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int) -> np.ndarray:
    """
    Convert image to patch embeddings.
    """
  # Extract dimensions
    batch, H, W, C = image.shape
    
    # Calculate number of patches
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    
    # Step 1: Reshape image into patches
    # New shape: (batch, num_patches_h, patch_size, num_patches_w, patch_size, C)
    patches = image.reshape(
        batch,
        num_patches_h, patch_size,
        num_patches_w, patch_size,
        C
    )
    
    # Step 2: Rearrange dimensions to get patches as sequences
    # First, transpose to bring patch dimensions together
    patches = patches.transpose(0, 1, 3, 2, 4, 5)  # (batch, num_patches_h, num_patches_w, patch_size, patch_size, C)
    
    # Step 3: Flatten each patch
    # (batch, num_patches_h, num_patches_w, patch_size * patch_size * C)
    patches_flat = patches.reshape(batch, num_patches_h, num_patches_w, patch_size * patch_size * C)
    
    # Step 4: Reshape to sequence form
    # (batch, num_patches, patch_size * patch_size * C)
    patches_seq = patches_flat.reshape(batch, num_patches, patch_size * patch_size * C)
    
    # Step 5: Linear projection to embedding dimension
    # Initialize projection weights (simplified - in practice these would be learned)
    # Shape: (patch_dim, embed_dim) where patch_dim = patch_size * patch_size * C
    patch_dim = patch_size * patch_size * C
    W_proj = np.random.randn(patch_dim, embed_dim) * 0.01
    
    # Apply linear projection
    embeddings = np.matmul(patches_seq, W_proj)  # (batch, num_patches, embed_dim)
    
    return embeddings
    pass