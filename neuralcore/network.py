
"""Composable feedforward neural network, built entirely with NumPy."""

import numpy as np

from neuralcore.activations import (
    sigmoid, sigmoid_derivative,
    tanh, tanh_derivative,
    relu, relu_derivative,
)
from neuralcore.layers import Dense


class Activation:
    """Activation layer with backpropagation support."""

    FUNCTIONS = {
        "sigmoid": (sigmoid, sigmoid_derivative),
        "tanh": (tanh, tanh_derivative),
        "relu": (relu, relu_derivative),
    }

    def __init__(self, name):
        if name not in self.FUNCTIONS:
            raise ValueError(f"Unknown activation: {name}")

        self.name = name
        self._forward_fn, self._derivative_fn = self.FUNCTIONS[name]
        self._inputs = None

    def forward(self, inputs):
        inputs = np.asarray(inputs, dtype=np.float64)

        if inputs.ndim != 2:
            raise ValueError("Activation expects a 2D array.")

        self._inputs = inputs
        return self._forward_fn(inputs)

    def backward(self, grad_output):
        if self._inputs is None:
            raise RuntimeError("Call forward() before backward().")

        grad_output = np.asarray(grad_output, dtype=np.float64)

        if grad_output.shape != self._inputs.shape:
            raise ValueError("Gradient shape does not match activation.")

        return grad_output * self._derivative_fn(self._inputs)


class Sequential:
    """Run layers in order and backpropagate in reverse."""

    def __init__(self, layers):
        self.layers = list(layers)

        if not self.layers:
            raise ValueError("A network needs at least one layer.")

    def forward(self, inputs):
        output = np.asarray(inputs, dtype=np.float64)

        if output.ndim != 2:
            raise ValueError("Network input must be a 2D array.")

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, grad_output):
        grad = np.asarray(grad_output, dtype=np.float64)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def parameters_and_gradients(self):
        """Yield learnable parameters and their gradients."""
        for layer in self.layers:
            if isinstance(layer, Dense):
                if layer.dW is None or layer.db is None:
                    raise RuntimeError(
                        "Call backward() before reading gradients."
                    )

                yield layer.weights, layer.dW
                yield layer.biases, layer.db
