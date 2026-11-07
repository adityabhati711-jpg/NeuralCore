
import numpy as np
import pytest

from neuralcore.losses import (
    MeanSquaredError,
    BinaryCrossEntropyWithLogits,
)


def numerical_gradient(loss, predictions, targets):
    epsilon = 1e-6
    gradient = np.zeros_like(predictions)

    for index in np.ndindex(predictions.shape):
        plus = predictions.copy()
        minus = predictions.copy()

        plus[index] += epsilon
        minus[index] -= epsilon

        gradient[index] = (
            loss.forward(plus, targets)
            - loss.forward(minus, targets)
        ) / (2.0 * epsilon)

    return gradient


def test_mse_values():
    loss = MeanSquaredError()

    predictions = np.array([[1.0], [3.0]])
    targets = np.array([[0.0], [1.0]])

    assert loss.forward(predictions, targets) == 2.5

    np.testing.assert_allclose(
        loss.backward(),
        [[1.0], [2.0]],
    )


def test_mse_gradient():
    loss = MeanSquaredError()

    predictions = np.array([[0.2], [-0.4], [0.8]])
    targets = np.array([[1.0], [0.0], [1.0]])

    loss.forward(predictions, targets)
    analytical = loss.backward().copy()

    numerical = numerical_gradient(
        loss, predictions, targets
    )

    np.testing.assert_allclose(
        analytical, numerical, rtol=1e-5, atol=1e-7
    )


def test_bce_values():
    loss = BinaryCrossEntropyWithLogits()

    logits = np.array([[0.0], [0.0]])
    targets = np.array([[0.0], [1.0]])

    np.testing.assert_allclose(
        loss.forward(logits, targets),
        np.log(2.0),
    )

    np.testing.assert_allclose(
        loss.backward(),
        [[0.25], [-0.25]],
    )


def test_bce_gradient():
    loss = BinaryCrossEntropyWithLogits()

    logits = np.array([[-1.2], [0.3], [2.0]])
    targets = np.array([[0.0], [1.0], [1.0]])

    loss.forward(logits, targets)
    analytical = loss.backward().copy()

    numerical = numerical_gradient(
        loss, logits, targets
    )

    np.testing.assert_allclose(
        analytical, numerical, rtol=1e-5, atol=1e-7
    )


def test_bce_extreme_logits():
    loss = BinaryCrossEntropyWithLogits()

    logits = np.array([[-1000.0], [1000.0]])
    targets = np.array([[0.0], [1.0]])

    value = loss.forward(logits, targets)
    gradients = loss.backward()

    assert np.isfinite(value)
    assert np.all(np.isfinite(gradients))
    assert value >= 0.0


def test_invalid_inputs_and_call_order():
    mse = MeanSquaredError()
    bce = BinaryCrossEntropyWithLogits()

    with pytest.raises(RuntimeError):
        mse.backward()

    with pytest.raises(RuntimeError):
        bce.backward()

    with pytest.raises(ValueError):
        mse.forward(
            np.ones((2, 1)), np.ones((2, 2))
        )

    with pytest.raises(ValueError):
        bce.forward(
            np.array([[0.0]]), np.array([[2.0]])
        )

    with pytest.raises(ValueError):
        bce.forward(
            np.array([[np.inf]]), np.array([[1.0]])
        )
