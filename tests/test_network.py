
import numpy as np
import pytest

from neuralcore.layers import Dense
from neuralcore.network import Activation, Sequential


def build_network():
    rng = np.random.default_rng(42)

    return Sequential([
        Dense(2, 3, rng=rng),
        Activation("tanh"),
        Dense(3, 1, rng=rng),
        Activation("sigmoid"),
    ])


def test_forward_predictions():
    network = build_network()

    predictions = network.forward(
        np.array([[0., 0.], [1., 0.], [1., 1.]])
    )

    assert predictions.shape == (3, 1)
    assert np.all(np.isfinite(predictions))
    assert np.all((predictions >= 0) & (predictions <= 1))


def test_backpropagation_shapes():
    network = build_network()

    network.forward(np.ones((4, 2)))
    grad_inputs = network.backward(np.ones((4, 1)))

    assert grad_inputs.shape == (4, 2)

    parameters = list(network.parameters_and_gradients())

    assert len(parameters) == 4
    assert all(p.shape == g.shape for p, g in parameters)


def test_network_gradients_numerically():
    network = build_network()

    x = np.array([[0.2, -0.3], [0.8, 0.1]])
    upstream = np.array([[1.0], [-0.5]])

    network.forward(x)

    analytic_dx = network.backward(upstream).copy()

    analytic_parameters = [
        (p, grad.copy())
        for p, grad in network.parameters_and_gradients()
    ]

    def objective():
        return np.sum(network.forward(x) * upstream)

    def finite_difference(values, epsilon=1e-6):
        result = np.zeros_like(values)

        for index in np.ndindex(values.shape):
            original = values[index]

            values[index] = original + epsilon
            plus = objective()

            values[index] = original - epsilon
            minus = objective()

            values[index] = original
            result[index] = (plus - minus) / (2 * epsilon)

        return result

    for parameter, analytic in analytic_parameters:
        np.testing.assert_allclose(
            analytic,
            finite_difference(parameter),
            rtol=1e-5,
            atol=1e-7,
        )

    np.testing.assert_allclose(
        analytic_dx,
        finite_difference(x),
        rtol=1e-5,
        atol=1e-7,
    )


def test_invalid_architecture():
    with pytest.raises(ValueError):
        Sequential([])

    with pytest.raises(ValueError):
        Activation("unknown")
