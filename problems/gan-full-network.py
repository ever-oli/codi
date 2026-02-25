import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

class GAN:
    def __init__(self, data_dim: int, noise_dim: int):
        """
        Initialize GAN with generator and discriminator.
        """
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        
        # =========== Generator weights ===========
        self.G_W1 = np.random.randn(noise_dim, 128) * 0.02
        self.G_b1 = np.zeros(128)
        self.G_W2 = np.random.randn(128, data_dim) * 0.02
        self.G_b2 = np.zeros(data_dim)
        
        # =========== Discriminator weights ===========
        self.D_W1 = np.random.randn(data_dim, 256) * 0.02
        self.D_b1 = np.zeros(256)
        self.D_W2 = np.random.randn(256, 128) * 0.02
        self.D_b2 = np.zeros(128)
        self.D_W3 = np.random.randn(128, 1) * 0.02
        self.D_b3 = np.zeros(1)
        
        # Learning rates (provided by starter code, though unused in this specific test)
        self.d_lr = 0.001
        self.g_lr = 0.001
    
    def _generator_forward(self, z: np.ndarray) -> np.ndarray:
        """Forward pass through generator."""
        h = np.maximum(0, np.matmul(z, self.G_W1) + self.G_b1)
        return np.tanh(np.matmul(h, self.G_W2) + self.G_b2)
    
    def _discriminator_forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through discriminator."""
        h1 = np.matmul(x, self.D_W1) + self.D_b1
        h1 = np.maximum(0.2 * h1, h1)
        
        h2 = np.matmul(h1, self.D_W2) + self.D_b2
        h2 = np.maximum(0.2 * h2, h2)
        
        logits = np.matmul(h2, self.D_W3) + self.D_b3
        return sigmoid(logits).flatten()
    
    def generate(self, n: int) -> np.ndarray:
        """Produce n samples from noise."""
        # Sample noise z ~ N(0, I)
        z = np.random.randn(n, self.noise_dim)
        # Apply generator transform
        return self._generator_forward(z)
    
    def discriminate(self, x: np.ndarray) -> np.ndarray:
        """Classify samples as real/fake."""
        return self._discriminator_forward(x)
    
    def train_step(self, real_data: np.ndarray) -> dict:
        """
        Perform one training iteration (forward passes & loss computation).
        """
        batch_size = real_data.shape[0]
        eps = 1e-8
        
        # 1. Generate fake data using the generate() method
        fake_data = self.generate(batch_size)
        
        # 2. Discriminate real and fake data
        real_probs = self.discriminate(real_data)
        fake_probs = self.discriminate(fake_data)
        
        # 3. Compute losses
        # Discriminator maximizes log(D(x)) + log(1 - D(G(z))), so loss is the negative
        d_loss = -np.mean(np.log(real_probs + eps) + np.log(1.0 - fake_probs + eps))
        
        # Generator maximizes log(D(G(z))), so loss is the negative
        g_loss = -np.mean(np.log(fake_probs + eps))
        
        return {
            'd_loss': float(d_loss),
            'g_loss': float(g_loss)
        }