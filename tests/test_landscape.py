
import numpy as np
import pytest

from neuralcore.layers import Dense
from neuralcore.network import Sequential
from neuralcore.losses import MeanSquaredError
from neuralcore.landscape import (
    flatten_parameters,
    assign_parameters,
    orthonormal_directions,
    compute_loss_surface,
    project_trajectory,
)


def build_model():
    return Sequential([
        Dense(2, 1, rng=np.random.default_rng(42))
    ])


def test_flatten_and_restore():
    model = build_model()
    original = flatten_parameters(model)

    assign_parameters(model, original + 1.0)
    np.testing.assert_allclose(
        flatten_parameters(model), original + 1.0
    )

    assign_parameters(model, original)
    np.testing.assert_array_equal(
        flatten_parameters(model), original
    )


def test_directions_are_orthonormal():
    a, b = orthonormal_directions(100, seed=42)

    np.testing.assert_allclose(np.linalg.norm(a), 1.0)
    np.testing.assert_allclose(np.linalg.norm(b), 1.0)
    np.testing.assert_allclose(np.dot(a, b), 0.0, atol=1e-12)


def test_directions_are_reproducible():
    first = orthonormal_directions(10, seed=42)
    second = orthonormal_directions(10, seed=42)

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_surface_center_matches_actual_loss():
    model = build_model()
    loss_fn = MeanSquaredError()

    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([[1.0], [0.0]])

    center = flatten_parameters(model)
    a, b = orthonormal_directions(center.size)

    actual = loss_fn.forward(model.forward(x), y)

    _, _, surface = compute_loss_surface(
        model, loss_fn, x, y, center, a, b,
        np.array([-1.0, 0.0, 1.0])
    )

    np.testing.assert_allclose(surface[1, 1], actual)


def test_surface_restores_original_parameters():
    model = build_model()
    original = flatten_parameters(model)

    x = np.ones((2, 2))
    y = np.zeros((2, 1))

    a, b = orthonormal_directions(original.size)

    compute_loss_surface(
        model, MeanSquaredError(), x, y,
        original, a, b, np.linspace(-1, 1, 5)
    )

    np.testing.assert_array_equal(
        flatten_parameters(model), original
    )


def test_trajectory_projection():
    center = np.array([0.0, 0.0, 0.0])
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])

    trajectory = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    projected = project_trajectory(
        trajectory, center, a, b
    )

    np.testing.assert_allclose(
        projected, [[1.0, 2.0], [4.0, 5.0]]
    )


def test_invalid_parameter_vector():
    model = build_model()

    with pytest.raises(ValueError):
        assign_parameters(model, np.ones(100))


def test_invalid_surface_directions():
    model = build_model()
    center = flatten_parameters(model)
    direction = np.ones(center.size)

    with pytest.raises(ValueError):
        compute_loss_surface(
            model,
            MeanSquaredError(),
            np.ones((2, 2)),
            np.zeros((2, 1)),
            center,
            direction,
            direction,
            np.linspace(-1, 1, 5),
        )
