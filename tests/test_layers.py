
import numpy as np
import pytest

from neuralcore.layers import Dense


def test_forward_output_shape():
    layer = Dense(2, 4, rng=np.random.default_rng(42))
    x = np.ones((5, 2))

    output = layer.forward(x)

    assert output.shape == (5, 4)


def test_reproducible_initialization():
    layer1 = Dense(2, 3, rng=np.random.default_rng(42))
    layer2 = Dense(2, 3, rng=np.random.default_rng(42))

    np.testing.assert_array_equal(
        layer1.weights, layer2.weights
    )


def test_backward_shapes():
    layer = Dense(2, 3, rng=np.random.default_rng(42))
    x = np.ones((4, 2))

    layer.forward(x)
    dx = layer.backward(np.ones((4, 3)))

    assert dx.shape == (4, 2)
    assert layer.dW.shape == (2, 3)
    assert layer.db.shape == (1, 3)


def test_gradients_numerically():
    layer = Dense(2, 2, rng=np.random.default_rng(7))

    x = np.array([[0.2, -0.4], [0.8, 0.1]])
    upstream = np.array([[1.0, -2.0], [0.5, 3.0]])

    layer.forward(x)
    analytic_dx = layer.backward(upstream)
    analytic_dw = layer.dW.copy()
    analytic_db = layer.db.copy()

    def objective():
        return np.sum(layer.forward(x) * upstream)

    def numerical_gradient(values):
        result = np.zeros_like(values)
        epsilon = 1e-6

        for index in np.ndindex(values.shape):
            original = values[index]

            values[index] = original + epsilon
            plus = objective()

            values[index] = original - epsilon
            minus = objective()

            values[index] = original
            result[index] = (plus - minus) / (2 * epsilon)

        return result

    np.testing.assert_allclose(
        analytic_dw, numerical_gradient(layer.weights),
        rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        analytic_db, numerical_gradient(layer.biases),
        rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        analytic_dx, numerical_gradient(x),
        rtol=1e-5, atol=1e-7
    )


def test_backward_requires_forward():
    layer = Dense(2, 3, rng=np.random.default_rng(42))

    with pytest.raises(RuntimeError):
        layer.backward(np.ones((2, 3)))
