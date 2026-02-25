import numpy as np

def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    """
    U-Net encoder block: double conv + max pool.
    """
    # Extract input dimensions
    batch, H, W, in_channels = x.shape
    
    # After two 3x3 convolutions with no padding:
    # Each conv reduces height and width by 2
    # conv1: (H, W) -> (H-2, W-2)
    # conv2: (H-2, W-2) -> (H-4, W-4)
    skip_H = H - 4
    skip_W = W - 4
    
    # Skip connection features (after conv2, before pooling)
    skip_out = np.zeros((batch, skip_H, skip_W, out_channels))
    
    # After 2x2 max pooling with stride 2:
    # Height and width are halved
    pool_H = skip_H // 2
    pool_W = skip_W // 2
    
    # Pooled output (after max pooling)
    pool_out = np.zeros((batch, pool_H, pool_W, out_channels))
    
    return pool_out, skip_out
    pass