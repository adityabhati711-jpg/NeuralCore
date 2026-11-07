
import numpy as np
import pytest

from neuralcore.activations import (
    sigmoid,
    sigmoid_derivative,
    relu,
    relu_derivative,
    tanh,
    tanh_derivative,
)


@pytest.mark.parametrize(
    "activation, derivative",
    [
        (sigmoid, sigmoid_derivative),
        (tanh, tanh_derivative),
        (relu, relu_derivative),
    ],
)
def test_derivatives_against_finite_differences(
    activation, derivative
):
    x = np.array([-2.0, -0.5, 0.5, 2.0])
    epsilon = 1e-6

    numerical = (
        activation(x + epsilon) -
        activation(x - epsilon)
    ) / (2 * epsilon)

    np.testing.assert_allclose(
        derivative(x),
        numerical,
        rtol=1e-5,
        atol=1e-7,
    )


def test_sigmoid_extreme_values():
    x = np.array([-1000.0, 0.0, 1000.0])
    result = sigmoid(x)

    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(
        result, [0.0, 0.5, 1.0]
    )


def test_relu():
    x = np.array([-3.0, 0.0, 4.0])
    np.testing.assert_array_equal(
        relu(x), [0.0, 0.0, 4.0]
    )
    np.testing.assert_array_equal(
        relu_derivative(x), [0.0, 0.0, 1.0]
    )
