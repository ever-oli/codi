import numpy as np

def relu(x):
    return np.maximum(0, x)

class BasicBlock:
    """Basic residual block (2 conv layers with skip connection)."""
    
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        self.downsample = downsample
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        # First convolutional layer (may downsample by stride 2)
        self.W1 = np.random.randn(in_ch, out_ch) * 0.01
        
        # Second convolutional layer
        self.W2 = np.random.randn(out_ch, out_ch) * 0.01
        
        # Projection shortcut if dimensions change (different channels or downsampling)
        if in_ch != out_ch or downsample:
            self.W_proj = np.random.randn(in_ch, out_ch) * 0.01
        else:
            self.W_proj = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Store identity for skip connection
        identity = x
        
        # MAIN PATH
        # First convolution
        out = np.matmul(x, self.W1)  
        out = relu(out)
        
        # Second convolution
        out = np.matmul(out, self.W2)  
        
        # SHORTCUT PATH (project if needed)
        if self.W_proj is not None:
            identity = np.matmul(identity, self.W_proj)  
        
        # COMBINE PATHS
        out = out + identity
        out = relu(out)
        
        return out


class ResNet18:
    """
    Simplified ResNet-18 architecture.
    """
    
    def __init__(self, num_classes: int = 10):
        # Initial convolution: 3 -> 64
        self.conv1 = np.random.randn(3, 64) * 0.01
        
        # Layer 1: 64 -> 64 (no downsampling)
        self.layer1 = [
            BasicBlock(64, 64, downsample=False),
            BasicBlock(64, 64, downsample=False)
        ]
        
        # Layer 2: 64 -> 128 (first block downsamples)
        self.layer2 = [
            BasicBlock(64, 128, downsample=True),
            BasicBlock(128, 128, downsample=False)
        ]
        
        # Layer 3: 128 -> 256 (first block downsamples)
        self.layer3 = [
            BasicBlock(128, 256, downsample=True),
            BasicBlock(256, 256, downsample=False)
        ]
        
        # Layer 4: 256 -> 512 (first block downsamples)
        self.layer4 = [
            BasicBlock(256, 512, downsample=True),
            BasicBlock(512, 512, downsample=False)
        ]
        
        # Final fully connected layer
        self.fc = np.random.randn(512, num_classes) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Initial convolution: 3 -> 64
        out = np.matmul(x, self.conv1)
        out = relu(out)
        
        # Layer 1: 64 -> 64 (2 blocks)
        for block in self.layer1:
            out = block.forward(out)
        
        # Layer 2: 64 -> 128 (2 blocks)
        for block in self.layer2:
            out = block.forward(out)
        
        # Layer 3: 128 -> 256 (2 blocks)
        for block in self.layer3:
            out = block.forward(out)
        
        # Layer 4: 256 -> 512 (2 blocks)
        for block in self.layer4:
            out = block.forward(out)
        
        # Fully connected layer
        logits = np.matmul(out, self.fc)
        
        return logits