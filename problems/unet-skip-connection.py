import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Crop encoder features and concatenate with decoder features.
    """
     # Extract dimensions
    batch, H_enc, W_enc, C_enc = encoder_features.shape
    batch, H_dec, W_dec, C_dec = decoder_features.shape
    
    # Calculate crop margins for center cropping
    crop_h = (H_enc - H_dec) // 2
    crop_w = (W_enc - W_dec) // 2
    
    # Perform center crop on encoder features
    # This takes the center portion of size H_dec x W_dec
    encoder_cropped = encoder_features[
        :, 
        crop_h:crop_h + H_dec, 
        crop_w:crop_w + W_dec, 
        :
    ]
    
    # Concatenate along channel dimension (axis=-1)
    output = np.concatenate([encoder_cropped, decoder_features], axis=-1)
    
    return output
    pass