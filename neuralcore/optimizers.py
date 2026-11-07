
"""NumPy-only gradient optimizers for NeuralCore."""

import numpy as np


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def _decay(value, name):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value < 1:
        raise ValueError(f"{name} must be in [0, 1).")
    return value


def _checked_parameters(parameters):
    pairs = list(parameters)
    if not pairs:
        raise ValueError("No parameters supplied.")

    checked = []
    seen = set()

    for parameter, gradient in pairs:
        if not isinstance(parameter, np.ndarray) or not np.issubdtype(
            parameter.dtype, np.floating
        ):
            raise ValueError(
                "Parameters must be floating-point NumPy arrays."
            )

        if id(parameter) in seen:
            raise ValueError("Duplicate parameter supplied.")
        seen.add(id(parameter))

        gradient = np.asarray(gradient, dtype=parameter.dtype)

        if gradient.shape != parameter.shape:
            raise ValueError(
                "Parameter and gradient shapes must match."
            )

        if not np.all(np.isfinite(parameter)) or not np.all(
            np.isfinite(gradient)
        ):
            raise ValueError(
                "Parameters and gradients must be finite."
            )

        checked.append((parameter, gradient.copy()))

    return checked


class SGD:
    """Standard gradient descent with mini-batch support."""

    def __init__(self, lr=0.01):
        self.lr = _positive(lr, "lr")

    def step(self, parameters):
        for parameter, gradient in _checked_parameters(parameters):
            parameter -= self.lr * gradient


class Momentum:
    """Classical momentum optimizer."""

    def __init__(self, lr=0.01, beta=0.9):
        self.lr = _positive(lr, "lr")
        self.beta = _decay(beta, "beta")
        self._slots = None

    def step(self, parameters):
        pairs = _checked_parameters(parameters)

        if self._slots is None:
            self._slots = [
                (p, np.zeros_like(p)) for p, _ in pairs
            ]
        elif len(pairs) != len(self._slots) or any(
            old is not new
            for (old, _), (new, _) in zip(self._slots, pairs)
        ):
            raise ValueError(
                "Optimizer parameters changed; create a new optimizer."
            )

        for (parameter, gradient), (_, velocity) in zip(
            pairs, self._slots
        ):
            velocity *= self.beta
            velocity += gradient
            parameter -= self.lr * velocity


class Adam:
    """Adam optimizer with bias-corrected moments."""

    def __init__(
        self,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        self.lr = _positive(lr, "lr")
        self.beta1 = _decay(beta1, "beta1")
        self.beta2 = _decay(beta2, "beta2")
        self.epsilon = _positive(epsilon, "epsilon")

        self._slots = None
        self.t = 0

    def step(self, parameters):
        pairs = _checked_parameters(parameters)

        if self._slots is None:
            self._slots = [
                (p, np.zeros_like(p), np.zeros_like(p))
                for p, _ in pairs
            ]
        elif len(pairs) != len(self._slots) or any(
            old is not new
            for (old, _, _), (new, _) in zip(self._slots, pairs)
        ):
            raise ValueError(
                "Optimizer parameters changed; create a new optimizer."
            )

        # One timestep per update, not per parameter.
        self.t += 1

        for (parameter, gradient), (_, first, second) in zip(
            pairs, self._slots
        ):
            first *= self.beta1
            first += (1.0 - self.beta1) * gradient

            second *= self.beta2
            second += (1.0 - self.beta2) * gradient**2

            corrected_first = first / (
                1.0 - self.beta1**self.t
            )
            corrected_second = second / (
                1.0 - self.beta2**self.t
            )

            parameter -= (
                self.lr
                * corrected_first
                / (np.sqrt(corrected_second) + self.epsilon)
            )
