import numpy as np

def relu(x):
    return np.maximum(0, x)

class ConvBlock:
    """
    Convolutional Block with projection shortcut.
    Used when input/output dimensions differ.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main path weights
        self.W1 = np.random.randn(in_channels, out_channels) * 0.01
        self.W2 = np.random.randn(out_channels, out_channels) * 0.01
        
        # Shortcut projection (1x1 conv equivalent)
        self.Ws = np.random.randn(in_channels, out_channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with projection shortcut.
        """
        # Store original shape info
        original_shape = x.shape
        
        # MAIN PATH
        # First linear transformation: in_channels -> out_channels
        main = np.matmul(x, self.W1)  # Shape: (batch, out_channels, ...)
        
        # Apply ReLU
        main = relu(main)
        
        # Second linear transformation: out_channels -> out_channels
        main = np.matmul(main, self.W2)  # Shape: (batch, out_channels, ...)
        
        # SHORTCUT PATH (Projection)
        # Project input to match output dimensions
        shortcut = np.matmul(x, self.Ws)  # Shape: (batch, out_channels, ...)
        
        # COMBINE PATHS
        # Element-wise addition
        out = main + shortcut
        
        # Final ReLU
        out = relu(out)
        
        return out
        pass
