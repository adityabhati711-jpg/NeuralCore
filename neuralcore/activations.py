
import numpy as np


def sigmoid(x):
    """Numerically stable sigmoid activation."""
    x = np.asarray(x, dtype=float)
    return np.exp(-np.logaddexp(0, -x))


def sigmoid_derivative(x):
    """Derivative of sigmoid with respect to its input."""
    a = sigmoid(x)
    return a * (1.0 - a)


def relu(x):
    """Rectified Linear Unit activation."""
    return np.maximum(0.0, x)


def relu_derivative(x):
    """Derivative of ReLU (defined as zero at x=0)."""
    return np.where(np.asarray(x) > 0, 1.0, 0.0)


def tanh(x):
    """Hyperbolic tangent activation."""
    return np.tanh(x)


def tanh_derivative(x):
    """Derivative of tanh with respect to its input."""
    a = tanh(x)
    return 1.0 - a**2
