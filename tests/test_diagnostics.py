
"""Tests for NeuralCore gradient and update diagnostics."""

from dataclasses import asdict

import numpy as np
import pytest

from neuralcore.diagnostics import (
    LayerDiagnostics,
    UpdateDiagnostics,
    collect_gradient_diagnostics,
    collect_update_diagnostics,
    diagnostics_to_dict,
    gradient_flow_summary,
    snapshot_parameters,
    stability_warnings,
)
from neuralcore.layers import Dense
from neuralcore.losses import BinaryCrossEntropyWithLogits
from neuralcore.network import Activation, Sequential
from neuralcore.optimizers import SGD


def make_model(seed=42):
    rng = np.random.default_rng(seed)

    return Sequential([
        Dense(2, 5, rng=rng, init="xavier"),
        Activation("tanh"),
        Dense(5, 1, rng=rng, init="xavier"),
    ])


def backward_pass(model):
    x = np.array([
        [-1.0, 0.5],
        [0.3, -0.7],
        [1.2, 0.8],
        [-0.4, -1.0],
    ])

    y = np.array([[0], [1], [1], [0]])

    criterion = BinaryCrossEntropyWithLogits()
    logits = model.forward(x)
    criterion.forward(logits, y)
    model.backward(criterion.backward())


def test_requires_backward_pass():
    model = make_model()

    with pytest.raises(RuntimeError):
        collect_gradient_diagnostics(model)


def test_layer_gradient_statistics():
    model = make_model()
    backward_pass(model)

    records = collect_gradient_diagnostics(model)

    assert len(records) == 2
    assert all(isinstance(r, LayerDiagnostics) for r in records)

    for record in records:
        assert record.gradient_norm >= 0
        assert record.gradient_mean_abs >= 0
        assert record.gradient_max_abs >= 0
        assert 0 <= record.gradient_zero_fraction <= 1
        assert record.parameter_count > 0


def test_gradient_norm_matches_manual_calculation():
    model = make_model()
    backward_pass(model)

    records = collect_gradient_diagnostics(model)
    layers = [
        layer for layer in model.layers
        if isinstance(layer, Dense)
    ]

    for record, layer in zip(records, layers):
        expected = np.linalg.norm(
            np.concatenate([
                layer.dW.ravel(),
                layer.db.ravel(),
            ])
        )

        assert record.gradient_norm == pytest.approx(expected)


def test_gradient_collection_is_reproducible():
    first = make_model(42)
    second = make_model(42)

    backward_pass(first)
    backward_pass(second)

    assert (
        collect_gradient_diagnostics(first)
        == collect_gradient_diagnostics(second)
    )


def test_snapshot_is_independent_copy():
    model = make_model()
    before = snapshot_parameters(model)

    old_value = before[0][0][0, 0]

    dense = [
        layer for layer in model.layers
        if isinstance(layer, Dense)
    ][0]

    dense.weights[0, 0] += 1

    assert before[0][0][0, 0] == old_value


def test_zero_update_ratio_before_training():
    model = make_model()
    before = snapshot_parameters(model)

    records = collect_update_diagnostics(model, before)

    assert len(records) == 2
    assert all(r.update_norm == 0 for r in records)
    assert all(r.update_to_parameter_ratio == 0 for r in records)


def test_update_statistics_after_optimizer_step():
    model = make_model()
    backward_pass(model)

    before = snapshot_parameters(model)

    optimizer = SGD(lr=0.01)
    optimizer.step(model.parameters_and_gradients())

    records = collect_update_diagnostics(model, before)

    assert len(records) == 2
    assert all(isinstance(r, UpdateDiagnostics) for r in records)
    assert any(r.update_norm > 0 for r in records)
    assert all(r.update_to_parameter_ratio >= 0 for r in records)


def test_update_norm_matches_manual_calculation():
    model = make_model()
    backward_pass(model)

    before = snapshot_parameters(model)
    SGD(lr=0.01).step(model.parameters_and_gradients())

    records = collect_update_diagnostics(model, before)

    dense_layers = [
        layer for layer in model.layers
        if isinstance(layer, Dense)
    ]

    for record, layer, (old_w, old_b) in zip(
        records, dense_layers, before
    ):
        expected = np.linalg.norm(
            np.concatenate([
                (layer.weights - old_w).ravel(),
                (layer.biases - old_b).ravel(),
            ])
        )

        assert record.update_norm == pytest.approx(expected)


def test_rejects_wrong_snapshot_count():
    model = make_model()

    with pytest.raises(ValueError):
        collect_update_diagnostics(model, [])


def test_rejects_wrong_snapshot_shape():
    model = make_model()
    before = snapshot_parameters(model)

    before[0] = (
        np.zeros((99, 99)),
        before[0][1],
    )

    with pytest.raises(ValueError):
        collect_update_diagnostics(model, before)


def test_gradient_flow_summary():
    model = make_model()
    backward_pass(model)

    records = collect_gradient_diagnostics(model)
    summary = gradient_flow_summary(records)

    assert summary["layer_count"] == 2
    assert summary["minimum_gradient_norm"] >= 0
    assert (
        summary["maximum_gradient_norm"]
        >= summary["minimum_gradient_norm"]
    )


def test_empty_gradient_flow_rejected():
    with pytest.raises(ValueError):
        gradient_flow_summary([])


def test_small_gradient_warning():
    record = LayerDiagnostics(
        layer_index=1,
        name="Dense_1",
        weight_norm=1.0,
        bias_norm=0.0,
        weight_gradient_norm=1e-12,
        bias_gradient_norm=0.0,
        gradient_norm=1e-12,
        gradient_mean_abs=1e-12,
        gradient_max_abs=1e-12,
        gradient_zero_fraction=0.0,
        parameter_count=5,
    )

    warnings = stability_warnings(
        [record],
        vanishing_threshold=1e-8,
    )

    assert any(
        warning["type"] == "small_gradient"
        for warning in warnings
    )


def test_large_gradient_warning():
    record = LayerDiagnostics(
        layer_index=1,
        name="Dense_1",
        weight_norm=1.0,
        bias_norm=0.0,
        weight_gradient_norm=200.0,
        bias_gradient_norm=0.0,
        gradient_norm=200.0,
        gradient_mean_abs=50.0,
        gradient_max_abs=200.0,
        gradient_zero_fraction=0.0,
        parameter_count=5,
    )

    warnings = stability_warnings(
        [record],
        exploding_threshold=100.0,
    )

    assert any(
        warning["type"] == "large_gradient"
        for warning in warnings
    )


def test_large_relative_update_warning():
    record = UpdateDiagnostics(
        layer_index=1,
        name="Dense_1",
        parameter_norm_before=1.0,
        update_norm=0.5,
        update_to_parameter_ratio=0.5,
    )

    warnings = stability_warnings(
        [],
        update_diagnostics=[record],
        large_update_ratio=0.1,
    )

    assert any(
        warning["type"] == "large_relative_update"
        for warning in warnings
    )


def test_undefined_relative_update_warning():
    record = UpdateDiagnostics(
        layer_index=1,
        name="Dense_1",
        parameter_norm_before=0.0,
        update_norm=1.0,
        update_to_parameter_ratio=float("inf"),
    )

    warnings = stability_warnings(
        [],
        update_diagnostics=[record],
    )

    assert any(
        warning["type"] == "undefined_relative_update"
        for warning in warnings
    )


@pytest.mark.parametrize(
    "thresholds",
    [
        {"vanishing_threshold": 0},
        {"exploding_threshold": -1},
        {"large_update_ratio": float("nan")},
    ],
)
def test_invalid_warning_thresholds(thresholds):
    with pytest.raises(ValueError):
        stability_warnings([], **thresholds)


def test_diagnostics_serialization():
    model = make_model()
    backward_pass(model)

    records = collect_gradient_diagnostics(model)
    serialized = diagnostics_to_dict(records)

    assert isinstance(serialized, list)
    assert isinstance(serialized[0], dict)
    assert serialized[0]["name"] == "Dense_1"
    assert serialized[0] == asdict(records[0])
