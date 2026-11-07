
import numpy as np


class Dense:
    """Fully connected neural network layer."""

    def __init__(self, input_size, output_size, *, rng, init="xavier"):
        if input_size <= 0 or output_size <= 0:
            raise ValueError("Layer sizes must be positive.")

        if init == "xavier":
            scale = np.sqrt(2.0 / (input_size + output_size))
        elif init == "he":
            scale = np.sqrt(2.0 / input_size)
        else:
            raise ValueError("Initialization must be 'xavier' or 'he'.")

        self.weights = rng.normal(
            0.0, scale, size=(input_size, output_size)
        )
        self.biases = np.zeros((1, output_size))

        self._inputs = None
        self.dW = None
        self.db = None

    def forward(self, inputs):
        """Calculate output Z = XW + b."""
        inputs = np.asarray(inputs, dtype=np.float64)

        if inputs.ndim != 2 or inputs.shape[1] != self.weights.shape[0]:
            raise ValueError("Invalid input dimensions.")

        self._inputs = inputs
        return inputs @ self.weights + self.biases

    def backward(self, grad_output):
        """Backpropagate gradients using the chain rule."""
        if self._inputs is None:
            raise RuntimeError("Call forward() before backward().")

        grad_output = np.asarray(grad_output, dtype=np.float64)

        expected_shape = (
            self._inputs.shape[0], self.weights.shape[1]
        )
        if grad_output.shape != expected_shape:
            raise ValueError("Invalid gradient dimensions.")

        self.dW = self._inputs.T @ grad_output
        self.db = np.sum(grad_output, axis=0, keepdims=True)

        grad_inputs = grad_output @ self.weights.T
        return grad_inputs
