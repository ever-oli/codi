import numpy as np

def relu(x):
    return np.maximum(0, x)

class BottleneckBlock:
    """
    Bottleneck Block: 1x1 -> 3x3 -> 1x1
    Reduces computation by compressing channels.
    """
    
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int):
        self.in_ch = in_channels
        self.bn_ch = bottleneck_channels  # Compressed dimension
        self.out_ch = out_channels
        
        # 1x1 reduce
        self.W1 = np.random.randn(in_channels, bottleneck_channels) * 0.01
        # 3x3 (simplified as dense)
        self.W2 = np.random.randn(bottleneck_channels, bottleneck_channels) * 0.01
        # 1x1 expand
        self.W3 = np.random.randn(bottleneck_channels, out_channels) * 0.01
        
        # Shortcut (if dimensions differ)
        self.Ws = np.random.randn(in_channels, out_channels) * 0.01 if in_channels != out_channels else None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Bottleneck forward: compress -> process -> expand + skip
        """
        # Store original for skip connection
        identity = x
        
        # MAIN PATH - Bottleneck
        # Step 1: Compress (1x1 convolution)
        out = np.matmul(x, self.W1)  # (batch, in_ch) @ (in_ch, bn_ch) -> (batch, bn_ch)
        out = relu(out)
        
        # Step 2: Process (3x3 convolution)
        out = np.matmul(out, self.W2)  # (batch, bn_ch) @ (bn_ch, bn_ch) -> (batch, bn_ch)
        out = relu(out)
        
        # Step 3: Expand (1x1 convolution)
        out = np.matmul(out, self.W3)  # (batch, bn_ch) @ (bn_ch, out_ch) -> (batch, out_ch)
        
        # SHORTCUT PATH
        # Project identity if dimensions don't match
        if self.Ws is not None:
            identity = np.matmul(identity, self.Ws)  # (batch, in_ch) @ (in_ch, out_ch) -> (batch, out_ch)
        
        # COMBINE PATHS
        out = out + identity
        out = relu(out)
        
        return out
        pass
