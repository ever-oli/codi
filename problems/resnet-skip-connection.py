import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    """
    # Start with the initial gradient vector
    grad = np.array(x, copy=True)
    
    # Backpropagate through layers (L down to 1)
    for F_grad in reversed(gradients_F):
        F_mat = np.array(F_grad)
        dim = F_mat.shape[-1] # Safely get the dimension from the Jacobian itself
        
        # Vector-Jacobian Product with skip connection: grad = grad @ (I + dF/dx)
        grad = grad @ (np.eye(dim) + F_mat)
        
    return grad

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    grad = np.array(x, copy=True)
    
    for F_grad in reversed(gradients_F):
        F_mat = np.array(F_grad)
        
        # Vector-Jacobian Product without skip: grad = grad @ dF/dx
        grad = grad @ F_mat
        
    return grad