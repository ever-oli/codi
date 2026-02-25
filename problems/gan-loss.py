import numpy as np

def discriminator_loss(real_probs: np.ndarray, fake_probs: np.ndarray) -> float:
    """
    Compute discriminator loss.
    """
    eps = 1e-8  # Small constant to avoid log(0)
    # Clip probabilities to avoid log(0) or log(1)
    real_probs = np.clip(real_probs, eps, 1 - eps)
    fake_probs = np.clip(fake_probs, eps, 1 - eps)
    
    # Loss for real data: -log(D(x))
    real_loss = -np.log(real_probs)
    
    # Loss for fake data: -log(1 - D(G(z)))
    fake_loss = -np.log(1 - fake_probs)
    
    # Average over batch and sum both components
    total_loss = np.mean(real_loss + fake_loss)
    
    return float(total_loss)
    pass

def generator_loss(fake_probs: np.ndarray) -> float:
    """
    Compute generator loss.
    """
    eps = 1e-8  # Small constant to avoid log(0)
    
    # Clip probabilities to avoid log(0)
    fake_probs = np.clip(fake_probs, eps, 1 - eps)
    
    # Generator loss: -log(D(G(z)))
    loss = -np.log(fake_probs)
    
    return float(np.mean(loss))
    pass