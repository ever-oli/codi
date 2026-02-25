import numpy as np

def generator(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Generate fake data from noise vectors.
    """
    # Get dimensions
    batch_size, noise_dim = z.shape
    
    # Initialize weights and biases for a simple 2-layer generator
    # First layer: noise_dim -> 128
    W1 = np.random.randn(noise_dim, 128) * 0.02  # Small initialization
    b1 = np.zeros(128)
    
    # Second layer: 128 -> output_dim
    W2 = np.random.randn(128, output_dim) * 0.02
    b2 = np.zeros(output_dim)
    
    # Forward pass through the network
    
    # First layer: Linear + ReLU
    h1 = np.matmul(z, W1) + b1
    h1 = np.maximum(0, h1)  # ReLU activation
    
    # Second layer: Linear + Tanh (for bounded output)
    output = np.matmul(h1, W2) + b2
    output = np.tanh(output)  # Tanh bounds output to [-1, 1]
    
    return output
    pass