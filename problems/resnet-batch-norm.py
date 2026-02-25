import numpy as np

class BatchNorm:
    """Batch Normalization layer."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)      # Scale parameter
        self.beta = np.zeros(num_features)      # Shift parameter
        self.running_mean = np.zeros(num_features)   # Running mean for inference
        self.running_var = np.ones(num_features)     # Running variance for inference
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply batch normalization.
        
        Args:
            x: Input tensor of shape (batch, channels, ...) or (batch, features)
            training: If True, use batch statistics; if False, use running statistics
        
        Returns:
            Normalized tensor of same shape as input
        """
        # Store original shape for reshaping later
        original_shape = x.shape
        
        # Reshape to (batch, features) if input has spatial dimensions
        if len(original_shape) > 2:
            # For CNN: (batch, channels, height, width) -> (batch, channels, -1)
            batch, channels = original_shape[0], original_shape[1]
            x_reshaped = x.reshape(batch, channels, -1)  # Flatten spatial dimensions
            x_reshaped = x_reshaped.transpose(0, 2, 1).reshape(-1, channels)  # (batch*spatial, channels)
        else:
            # For MLP: (batch, features) as is
            x_reshaped = x
            channels = original_shape[-1]
        
        if training:
            # Compute batch statistics
            batch_mean = np.mean(x_reshaped, axis=0)  # Mean across batch
            batch_var = np.var(x_reshaped, axis=0)    # Variance across batch
            
            # Update running statistics (for inference)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Normalize using batch statistics
            x_norm = (x_reshaped - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            # Normalize using running statistics (inference mode)
            x_norm = (x_reshaped - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        # Apply scale and shift
        out = self.gamma * x_norm + self.beta
        
        # Reshape back to original dimensions
        if len(original_shape) > 2:
            out = out.reshape(batch, -1, channels).transpose(0, 2, 1)
            out = out.reshape(original_shape)
        else:
            out = out.reshape(original_shape)
        
        return out


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def post_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, 
                         bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Post-activation ResNet block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Uses x @ W for "convolution" (simplified as linear transform).
    
    This is the original ResNet order where activation comes after BN.
    """
    # First layer: Conv -> BN -> ReLU
    out = np.matmul(x, W1)        # Conv (simplified)
    out = bn1.forward(out)        # BN
    out = relu(out)                # ReLU
    
    # Second layer: Conv -> BN
    out = np.matmul(out, W2)       # Conv
    out = bn2.forward(out)         # BN
    
    # Add skip connection and apply ReLU
    out = out + x                   # Skip connection
    out = relu(out)                 # Final ReLU
    
    return out


def pre_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray,
                        bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Pre-activation ResNet block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    This ordering often works better for very deep networks.
    
    In this design, activation happens before convolution, improving gradient flow.
    """
    # First layer: BN -> ReLU -> Conv
    out = bn1.forward(x)           # BN first
    out = relu(out)                 # Then ReLU
    out = np.matmul(out, W1)        # Then Conv
    
    # Second layer: BN -> ReLU -> Conv
    out = bn2.forward(out)          # BN
    out = relu(out)                  # ReLU
    out = np.matmul(out, W2)         # Conv
    
    # Add skip connection (no final activation needed)
    out = out + x
    
    return out
