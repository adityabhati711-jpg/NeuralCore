
import numpy as np
import pytest

from neuralcore.datasets import (
    make_moons,
    train_val_test_split,
    StandardScaler,
)


def test_moons_shapes_and_balance():
    x, y = make_moons(n_samples=601)

    assert x.shape == (601, 2)
    assert y.shape == (601, 1)
    assert set(np.unique(y)) == {0, 1}
    assert abs(np.sum(y == 0) - np.sum(y == 1)) <= 1


def test_moons_reproducible():
    x1, y1 = make_moons(seed=123)
    x2, y2 = make_moons(seed=123)

    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_moons_different_seeds():
    x1, _ = make_moons(seed=1)
    x2, _ = make_moons(seed=2)

    assert not np.array_equal(x1, x2)


def test_split_sizes_and_class_balance():
    x, y = make_moons(n_samples=600)

    splits = train_val_test_split(x, y)

    x_train, y_train, x_val, y_val, x_test, y_test = splits

    assert len(x_train) == 420
    assert len(x_val) == 90
    assert len(x_test) == 90

    for labels in (y_train, y_val, y_test):
        assert set(np.unique(labels)) == {0, 1}


def test_split_has_no_overlap():
    x, y = make_moons(n_samples=600, noise=0.12)

    x_train, _, x_val, _, x_test, _ = train_val_test_split(
        x, y
    )

    train_rows = set(map(tuple, x_train))
    val_rows = set(map(tuple, x_val))
    test_rows = set(map(tuple, x_test))

    assert train_rows.isdisjoint(val_rows)
    assert train_rows.isdisjoint(test_rows)
    assert val_rows.isdisjoint(test_rows)


def test_split_reproducible():
    x, y = make_moons()

    first = train_val_test_split(x, y, seed=42)
    second = train_val_test_split(x, y, seed=42)

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_scaler_training_statistics():
    x_train = np.array([
        [1.0, 10.0],
        [3.0, 20.0],
        [5.0, 30.0],
    ])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(x_train)

    np.testing.assert_allclose(
        np.mean(scaled, axis=0), [0.0, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(
        np.std(scaled, axis=0), [1.0, 1.0]
    )


def test_scaler_does_not_refit_on_test():
    x_train = np.array([[0.0], [2.0]])
    x_test = np.array([[100.0]])

    scaler = StandardScaler().fit(x_train)
    result = scaler.transform(x_test)

    np.testing.assert_allclose(result, [[99.0]])
    np.testing.assert_allclose(scaler.mean_, [[1.0]])


def test_invalid_inputs():
    with pytest.raises(ValueError):
        make_moons(n_samples=2)

    with pytest.raises(ValueError):
        make_moons(noise=-1)

    with pytest.raises(ValueError):
        train_val_test_split(
            np.ones((10, 2)),
            np.zeros((10, 1)),
        )

    with pytest.raises(RuntimeError):
        StandardScaler().transform(np.ones((2, 2)))
