
"""Parameter-space loss landscape utilities for NeuralCore."""

import numpy as np


def parameters(model):
    """Return live parameter arrays from a Sequential network."""
    from neuralcore.layers import Dense

    arrays = []
    for layer in model.layers:
        if isinstance(layer, Dense):
            arrays.extend([layer.weights, layer.biases])

    if not arrays:
        raise ValueError("Model contains no trainable parameters.")

    return arrays


def flatten_parameters(model):
    """Flatten all trainable parameters into one vector."""
    return np.concatenate([
        p.ravel().copy() for p in parameters(model)
    ])


def assign_parameters(model, vector):
    """Copy a flat vector into the model's existing arrays."""
    arrays = parameters(model)
    vector = np.asarray(vector, dtype=np.float64)

    expected = sum(p.size for p in arrays)

    if vector.shape != (expected,):
        raise ValueError("Incorrect parameter vector shape.")

    if not np.all(np.isfinite(vector)):
        raise ValueError("Parameter vector must be finite.")

    offset = 0
    for p in arrays:
        size = p.size
        p[...] = vector[offset:offset + size].reshape(p.shape)
        offset += size


def orthonormal_directions(n_parameters, seed=42):
    """Generate two reproducible orthogonal unit directions."""
    if n_parameters < 2:
        raise ValueError("At least two parameters required.")

    rng = np.random.default_rng(seed)
    a = rng.normal(size=n_parameters)
    a /= np.linalg.norm(a)

    b = rng.normal(size=n_parameters)
    b -= np.dot(a, b) * a
    b /= np.linalg.norm(b)

    return a, b


def compute_loss_surface(
    model,
    loss_fn,
    x,
    y,
    center,
    direction_a,
    direction_b,
    coordinates,
):
    """Evaluate loss on a 2D slice through parameter space.

    The model is restored to its original parameters afterward.
    """
    original = flatten_parameters(model)

    center = np.asarray(center, dtype=np.float64)
    direction_a = np.asarray(direction_a, dtype=np.float64)
    direction_b = np.asarray(direction_b, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)

    n = original.size

    if any(
        vector.shape != (n,)
        for vector in (center, direction_a, direction_b)
    ):
        raise ValueError("Parameter-space dimension mismatch.")

    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError("Coordinates must be a 1D grid.")

    if not all(
        np.all(np.isfinite(arr))
        for arr in (
            center, direction_a, direction_b, coordinates
        )
    ):
        raise ValueError("All inputs must be finite.")

    if not np.isclose(np.linalg.norm(direction_a), 1.0):
        raise ValueError("First direction must have unit norm.")

    if not np.isclose(np.linalg.norm(direction_b), 1.0):
        raise ValueError("Second direction must have unit norm.")

    if not np.isclose(
        np.dot(direction_a, direction_b), 0.0, atol=1e-8
    ):
        raise ValueError("Directions must be orthogonal.")

    aa, bb = np.meshgrid(coordinates, coordinates)
    surface = np.empty_like(aa)

    try:
        for index in np.ndindex(surface.shape):
            vector = (
                center
                + aa[index] * direction_a
                + bb[index] * direction_b
            )
            assign_parameters(model, vector)
            surface[index] = loss_fn.forward(
                model.forward(x), y
            )

            if not np.isfinite(surface[index]):
                raise FloatingPointError(
                    "Non-finite loss encountered."
                )
    finally:
        assign_parameters(model, original)

    return aa, bb, surface


def project_trajectory(
    trajectory, center, direction_a, direction_b
):
    """Project weight history onto two orthonormal directions."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    direction_a = np.asarray(direction_a, dtype=np.float64)
    direction_b = np.asarray(direction_b, dtype=np.float64)

    if trajectory.ndim != 2 or trajectory.shape[1] != center.size:
        raise ValueError("Invalid trajectory dimensions.")

    if direction_a.shape != center.shape or direction_b.shape != center.shape:
        raise ValueError("Direction dimensions do not match.")

    if not all(np.all(np.isfinite(v)) for v in (
        trajectory, center, direction_a, direction_b
    )):
        raise ValueError("Non-finite trajectory input.")

    displacement = trajectory - center

    return np.column_stack([
        displacement @ direction_a,
        displacement @ direction_b,
    ])
