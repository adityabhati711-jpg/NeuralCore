
import numpy as np
import pytest

from neuralcore.optimizers import SGD, Momentum, Adam


def test_sgd_exact_update_and_unchanged_gradient():
    parameter = np.array([1.0, -1.0])
    gradient = np.array([2.0, -2.0])

    SGD(lr=0.1).step([(parameter, gradient)])

    np.testing.assert_allclose(parameter, [0.8, -0.8])
    np.testing.assert_array_equal(gradient, [2.0, -2.0])


def test_momentum_accumulates_gradients():
    parameter = np.array([1.0])
    optimizer = Momentum(lr=0.1, beta=0.5)

    optimizer.step([(parameter, np.array([2.0]))])
    np.testing.assert_allclose(parameter, [0.8])

    optimizer.step([(parameter, np.array([1.0]))])
    np.testing.assert_allclose(parameter, [0.6])


def test_adam_corrects_bias_once_per_step():
    first = np.array([1.0, -1.0])
    second = np.array([5.0])
    optimizer = Adam(lr=0.1)

    parameters = [
        (first, np.array([2.0, -2.0])),
        (second, np.array([3.0])),
    ]

    optimizer.step(parameters)
    np.testing.assert_allclose(
        first, [0.9, -0.9], atol=1e-7
    )
    np.testing.assert_allclose(
        second, [4.9], atol=1e-7
    )

    optimizer.step(parameters)
    np.testing.assert_allclose(
        first, [0.8, -0.8], atol=1e-7
    )
    np.testing.assert_allclose(
        second, [4.8], atol=1e-7
    )

    assert optimizer.t == 2


def test_rejects_bad_hyperparameters():
    with pytest.raises(ValueError):
        SGD(lr=0)

    with pytest.raises(ValueError):
        Momentum(beta=1.0)

    with pytest.raises(ValueError):
        Adam(epsilon=0)


def test_rejects_invalid_gradients_before_any_update():
    first = np.array([1.0])
    second = np.array([2.0])
    optimizer = Adam()

    with pytest.raises(ValueError):
        optimizer.step([
            (first, np.array([1.0])),
            (second, np.array([np.nan])),
        ])

    np.testing.assert_array_equal(first, [1.0])
    np.testing.assert_array_equal(second, [2.0])
    assert optimizer.t == 0


def test_rejects_parameter_gradient_shape_mismatch():
    parameter = np.ones((2, 2))

    with pytest.raises(ValueError):
        SGD().step([
            (parameter, np.ones((2, 1)))
        ])


def test_momentum_prevents_reuse_with_new_parameters():
    optimizer = Momentum()

    optimizer.step([
        (np.array([1.0]), np.array([0.5]))
    ])

    with pytest.raises(ValueError):
        optimizer.step([
            (np.array([2.0]), np.array([0.5]))
        ])


def test_optimizer_decreases_linear_model_loss():
    """Integration: network + loss + optimizer."""

    from neuralcore.layers import Dense
    from neuralcore.network import Sequential
    from neuralcore.losses import MeanSquaredError

    x = np.array([[-1.0], [0.0], [1.0]])
    y = 2.0 * x + 1.0

    model = Sequential([
        Dense(1, 1, rng=np.random.default_rng(42))
    ])

    loss = MeanSquaredError()
    optimizer = SGD(lr=0.1)

    initial_loss = loss.forward(model.forward(x), y)

    for _ in range(150):
        loss.forward(model.forward(x), y)
        model.backward(loss.backward())
        optimizer.step(model.parameters_and_gradients())

    final_loss = loss.forward(model.forward(x), y)

    assert final_loss < initial_loss * 0.01
