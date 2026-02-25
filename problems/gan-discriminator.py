import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function with numerical stability.
    
    Args:
        x: Input array
    
    Returns:
        Output in range [0, 1]
    """
    # Clip to prevent overflow in exp(-x)
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))
    
def discriminator(x: np.ndarray) -> np.ndarray:
    """
    Classify inputs as real or fake.
    """
    # Get dimensions
    batch_size, input_dim = x.shape
    
    # Initialize weights and biases for a 2-layer discriminator
    # First layer: input_dim -> 256
    W1 = np.random.randn(input_dim, 256) * 0.02
    b1 = np.zeros(256)
    
    # Second layer: 256 -> 128
    W2 = np.random.randn(256, 128) * 0.02
    b2 = np.zeros(128)
    
    # Output layer: 128 -> 1
    W3 = np.random.randn(128, 1) * 0.02
    b3 = np.zeros(1)
    
    # Forward pass
    
    # First hidden layer: Linear + LeakyReLU (better for GANs)
    h1 = np.matmul(x, W1) + b1
    h1 = np.maximum(0.2 * h1, h1)  # LeakyReLU with slope 0.2
    
    # Second hidden layer: Linear + LeakyReLU
    h2 = np.matmul(h1, W2) + b2
    h2 = np.maximum(0.2 * h2, h2)
    
    # Output layer: Linear + Sigmoid
    logits = np.matmul(h2, W3) + b3
    probs = sigmoid(logits)
    
    return probs

    pass