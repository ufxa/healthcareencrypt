"""
Inference Agent (IA) — HE-MedInfer
Executes the ML model homomorphically using CKKS operations.
Activation functions are replaced by degree-7 Chebyshev polynomial approximations.
"""
import numpy as np


def chebyshev_relu_approx(x: np.ndarray, degree: int = 7) -> np.ndarray:
    """
    Degree-7 Chebyshev polynomial approximation of ReLU over [-5, 5].
    Used as the homomorphic-friendly activation function.
    """
    coeffs = np.polynomial.chebyshev.chebfit(
        np.linspace(-5, 5, 1000),
        np.maximum(np.linspace(-5, 5, 1000), 0),
        degree,
    )
    return np.polynomial.chebyshev.chebval(x, coeffs)


def he_linear_layer(ct, weights: np.ndarray, bias: np.ndarray, he_context):
    """Perform a homomorphic linear transformation: ct = W * ct + b."""
    raise NotImplementedError("Wire up Microsoft SEAL homomorphic ops here.")


def he_inference(ct, model_params: list, he_context):
    """
    Run full encrypted MLP inference.
    model_params: list of (W, b) tuples per layer.
    """
    h = ct
    for i, (W, b) in enumerate(model_params):
        h = he_linear_layer(h, W, b, he_context)
        if i < len(model_params) - 1:
            h = chebyshev_relu_approx(h)
    return h
