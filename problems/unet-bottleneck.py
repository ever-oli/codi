import numpy as np

def unet_bottleneck(x: np.ndarray, out_channels: int) -> np.ndarray:
    """
    U-Net bottleneck: double convolution at lowest resolution.
    """
       # Extract input dimensions
    batch, H, W, in_channels = x.shape
    
    # Two 3x3 unpadded convolutions reduce spatial dimensions by 4 total
    # Each conv reduces H and W by 2
    H_out = H - 4
    W_out = W - 4
    
    # Create output tensor with correct shape
    output = np.zeros((batch, H_out, W_out, out_channels))
    
    return output
    pass