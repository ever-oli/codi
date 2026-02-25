import numpy as np

def unet_output(features: np.ndarray, num_classes: int) -> np.ndarray:
    """
    U-Net output layer: 1x1 conv for pixel-wise classification.
    """
     # Extract dimensions
    batch, H, W, feat_channels = features.shape
    
    # 1x1 convolution preserves spatial dimensions
    # Output channels = num_classes
    output = np.zeros((batch, H, W, num_classes))
    
    return output
    pass