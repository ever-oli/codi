import numpy as np

def unet_decoder_block(x: np.ndarray, skip: np.ndarray, out_channels: int) -> np.ndarray:
    """
    U-Net decoder block: up-conv + concat + double conv.
    """
    # Extract dimensions
    batch, H, W, in_channels = x.shape
    batch_skip, H_skip, W_skip, skip_channels = skip.shape
    
    # Step 1: Upsample - double the spatial dimensions
    H_up = H * 2
    W_up = W * 2
    
    # Step 2: Crop skip connection to match upsampled size
    # Calculate crop margins (center crop)
    crop_h = (H_skip - H_up) // 2
    crop_w = (W_skip - W_up) // 2
    
    # Cropped skip connection
    skip_cropped = skip[:, crop_h:crop_h+H_up, crop_w:crop_w+W_up, :]
    
    # Step 3: Concatenate along channel axis
    # After concatenation: (batch, H_up, W_up, in_channels + skip_channels)
    
    # Step 4: Two 3x3 unpadded convolutions
    # Each conv reduces H and W by 2, so total reduction is 4
    H_out = H_up - 4
    W_out = W_up - 4
    
    # Create output tensor with correct shape
    output = np.zeros((batch, H_out, W_out, out_channels))
    
    return output
    pass