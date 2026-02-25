import numpy as np

def encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    """
    Encoder block: two 3x3 convs + max pool.
    Returns (pooled_output, skip_connection)
    """
    batch, H, W, in_channels = x.shape
    
    # After two 3x3 convs: H-4, W-4
    skip_H = H - 4
    skip_W = W - 4
    skip = np.zeros((batch, skip_H, skip_W, out_channels))
    
    # After 2x2 max pool: (H-4)//2, (W-4)//2
    pool_H = skip_H // 2
    pool_W = skip_W // 2
    pooled = np.zeros((batch, pool_H, pool_W, out_channels))
    
    return pooled, skip

def bottleneck(x: np.ndarray, out_channels: int) -> np.ndarray:
    """Bottleneck: two 3x3 convs."""
    batch, H, W, in_channels = x.shape
    # After two 3x3 convs: H-4, W-4
    return np.zeros((batch, H-4, W-4, out_channels))

def decoder_block(x: np.ndarray, skip: np.ndarray, out_channels: int) -> np.ndarray:
    """
    Decoder block: up-conv + concat + two 3x3 convs.
    """
    batch, H, W, in_channels = x.shape
    
    # Upsample: double spatial dimensions
    H_up = H * 2
    W_up = W * 2
    
    # Skip connection is larger, need to center crop
    _, H_skip, W_skip, C_skip = skip.shape
    crop_h = (H_skip - H_up) // 2
    crop_w = (W_skip - W_up) // 2
    
    # After concatenation and two 3x3 convs: H_up-4, W_up-4
    H_out = H_up - 4
    W_out = W_up - 4
    
    return np.zeros((batch, H_out, W_out, out_channels))

def output_layer(x: np.ndarray, num_classes: int) -> np.ndarray:
    """Output layer: 1x1 conv."""
    batch, H, W, channels = x.shape
    return np.zeros((batch, H, W, num_classes))
    
def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Complete U-Net for segmentation.
    """
    # =========== ENCODER PATH ===========
    # Store skip connections for decoder
    
    # Encoder 1: 64 channels
    e1_pool, e1_skip = encoder_block(x, out_channels=64)
    
    # Encoder 2: 128 channels
    e2_pool, e2_skip = encoder_block(e1_pool, out_channels=128)
    
    # Encoder 3: 256 channels
    e3_pool, e3_skip = encoder_block(e2_pool, out_channels=256)
    
    # Encoder 4: 512 channels
    e4_pool, e4_skip = encoder_block(e3_pool, out_channels=512)
    
    # =========== BOTTLENECK ===========
    # Deepest layer: 1024 channels
    bottleneck_out = bottleneck(e4_pool, out_channels=1024)
    
    # =========== DECODER PATH ===========
    # Use skips in reverse order (last encoder skip goes to first decoder)
    
    # Decoder 4: 512 channels (using e4_skip)
    d4_out = decoder_block(bottleneck_out, e4_skip, out_channels=512)
    
    # Decoder 3: 256 channels (using e3_skip)
    d3_out = decoder_block(d4_out, e3_skip, out_channels=256)
    
    # Decoder 2: 128 channels (using e2_skip)
    d2_out = decoder_block(d3_out, e2_skip, out_channels=128)
    
    # Decoder 1: 64 channels (using e1_skip)
    d1_out = decoder_block(d2_out, e1_skip, out_channels=64)
    
    # =========== OUTPUT LAYER ===========
    # Final 1x1 convolution to get class scores
    output = output_layer(d1_out, num_classes)
    
    return output

    pass