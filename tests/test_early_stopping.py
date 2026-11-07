
"""Tests for validation-based early stopping and checkpoint recovery."""

import numpy as np
import pytest

from neuralcore.early_stopping import (
    EarlyStopping,
    EarlyStoppingResult,
)
from neuralcore.layers import Dense
from neuralcore.network import Activation, Sequential


def make_model(seed=42):
    rng = np.random.default_rng(seed)

    return Sequential([
        Dense(2, 5, rng=rng, init="xavier"),
        Activation("tanh"),
        Dense(5, 1, rng=rng, init="xavier"),
    ])


def model_parameters(model):
    return [
        parameter
        for layer in model.layers
        if isinstance(layer, Dense)
        for parameter in (
            layer.weights,
            layer.biases,
        )
    ]


def test_initial_state():
    stopper = EarlyStopping(patience=3)

    assert stopper.best_epoch == 0
    assert stopper.epochs_observed == 0
    assert stopper.wait == 0
    assert stopper.should_stop is False


@pytest.mark.parametrize(
    "patience",
    [0, -1, 1.5, True, False, "5"],
)
def test_invalid_patience(patience):
    with pytest.raises(ValueError):
        EarlyStopping(patience=patience)


@pytest.mark.parametrize(
    "delta",
    [-0.1, np.nan, np.inf, True, "0.1"],
)
def test_invalid_min_delta(delta):
    with pytest.raises(ValueError):
        EarlyStopping(min_delta=delta)


def test_improvement_updates_best_epoch():
    model = make_model()
    stopper = EarlyStopping(patience=3)

    stopper.update(model, 0.5, 1)
    stopper.update(model, 0.4, 2)
    stopper.update(model, 0.3, 3)

    assert stopper.best_epoch == 3
    assert stopper.best_validation_loss == pytest.approx(0.3)
    assert stopper.wait == 0


def test_patience_triggers_stopping():
    model = make_model()
    stopper = EarlyStopping(patience=2)

    assert stopper.update(model, 0.3, 1) is False
    assert stopper.update(model, 0.4, 2) is False
    assert stopper.update(model, 0.5, 3) is True

    assert stopper.should_stop is True
    assert stopper.epochs_observed == 3


def test_improvement_resets_patience():
    model = make_model()
    stopper = EarlyStopping(patience=2)

    stopper.update(model, 0.5, 1)
    stopper.update(model, 0.6, 2)
    stopper.update(model, 0.4, 3)

    assert stopper.wait == 0
    assert stopper.should_stop is False


def test_min_delta_requires_sufficient_improvement():
    model = make_model()
    stopper = EarlyStopping(
        patience=3,
        min_delta=0.01,
    )

    stopper.update(model, 0.5, 1)
    stopper.update(model, 0.495, 2)

    assert stopper.best_epoch == 1
    assert stopper.wait == 1


def test_best_checkpoint_is_independent_copy():
    model = make_model()
    stopper = EarlyStopping()

    stopper.update(model, 0.3, 1)

    parameters = model_parameters(model)
    originals = [p.copy() for p in parameters]

    for parameter in parameters:
        parameter[...] += 10.0

    stopper.restore_best(model)

    for restored, original in zip(
        model_parameters(model), originals
    ):
        np.testing.assert_array_equal(restored, original)


def test_restore_best_checkpoint_not_last():
    model = make_model()
    stopper = EarlyStopping(patience=5)

    stopper.update(model, 0.3, 1)

    original = [
        p.copy() for p in model_parameters(model)
    ]

    for parameter in model_parameters(model):
        parameter[...] += 1

    stopper.update(model, 0.4, 2)
    stopper.restore_best(model)

    assert stopper.best_epoch == 1

    for parameter, saved in zip(
        model_parameters(model), original
    ):
        np.testing.assert_array_equal(parameter, saved)


def test_restore_without_checkpoint_rejected():
    with pytest.raises(RuntimeError):
        EarlyStopping().restore_best(make_model())


def test_nonconsecutive_epochs_rejected():
    model = make_model()
    stopper = EarlyStopping()

    with pytest.raises(ValueError):
        stopper.update(model, 0.5, 2)

    stopper.update(model, 0.5, 1)

    with pytest.raises(ValueError):
        stopper.update(model, 0.4, 3)


@pytest.mark.parametrize(
    "loss",
    [np.nan, np.inf, -np.inf, True, "0.5"],
)
def test_nonfinite_or_invalid_loss_rejected(loss):
    with pytest.raises(ValueError):
        EarlyStopping().update(
            make_model(), loss, 1
        )


def test_updates_rejected_after_stop():
    model = make_model()
    stopper = EarlyStopping(patience=1)

    stopper.update(model, 0.3, 1)
    stopper.update(model, 0.4, 2)

    with pytest.raises(RuntimeError):
        stopper.update(model, 0.2, 3)


def test_result_dataclass():
    model = make_model()
    stopper = EarlyStopping(patience=2)

    stopper.update(model, 0.4, 1)
    stopper.update(model, 0.5, 2)
    stopper.update(model, 0.6, 3)

    result = stopper.result()

    assert isinstance(result, EarlyStoppingResult)
    assert result.best_epoch == 1
    assert result.best_validation_loss == pytest.approx(0.4)
    assert result.epochs_observed == 3
    assert result.stopped_early is True


def test_result_before_training_rejected():
    with pytest.raises(RuntimeError):
        EarlyStopping().result()


def test_parameter_shape_change_rejected():
    model = make_model()
    stopper = EarlyStopping()

    stopper.update(model, 0.5, 1)

    dense = [
        layer for layer in model.layers
        if isinstance(layer, Dense)
    ][0]

    dense.weights = np.zeros((99, 99))

    with pytest.raises(ValueError):
        stopper.update(model, 0.4, 2)


def test_nonfinite_parameters_rejected():
    model = make_model()
    dense = [
        layer for layer in model.layers
        if isinstance(layer, Dense)
    ][0]

    dense.weights[0, 0] = np.nan

    with pytest.raises(ValueError):
        EarlyStopping().update(model, 0.5, 1)


def test_checkpoint_restoration_is_repeatable():
    model = make_model()
    stopper = EarlyStopping()

    stopper.update(model, 0.3, 1)

    originals = [
        p.copy() for p in model_parameters(model)
    ]

    for _ in range(3):
        for parameter in model_parameters(model):
            parameter[...] += 5

        stopper.restore_best(model)

        for parameter, saved in zip(
            model_parameters(model), originals
        ):
            np.testing.assert_array_equal(parameter, saved)
