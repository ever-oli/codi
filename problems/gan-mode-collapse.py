import numpy as np

def detect_mode_collapse(generated_samples: np.ndarray, threshold: float = 0.1) -> dict:
    """
    Detect mode collapse in generated samples.
    """
    # Compute standard deviation along the batch axis (axis=0)
    # This gives the std for each feature across all samples
    feature_stds = np.std(generated_samples, axis=0)
    
    # Average the standard deviations across all features
    diversity_score = float(np.mean(feature_stds))
    
    # Check if diversity score is below threshold
    is_collapsed = diversity_score < threshold
    
    return {
        'diversity_score': diversity_score,
        'is_collapsed': is_collapsed
    }
    pass