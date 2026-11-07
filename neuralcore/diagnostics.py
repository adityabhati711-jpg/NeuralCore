
"""Gradient and parameter diagnostics for NumPy neural networks.

Provides layer-wise gradient statistics, gradient-flow measurements,
update-to-parameter ratios, and configurable stability warnings.

Diagnostic thresholds are heuristics, not universal failure criteria.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from neuralcore.layers import Dense


@dataclass(frozen=True)
class LayerDiagnostics:
    """Measurements for one Dense layer."""

    layer_index: int
    name: str
    weight_norm: float
    bias_norm: float
    weight_gradient_norm: float
    bias_gradient_norm: float
    gradient_norm: float
    gradient_mean_abs: float
    gradient_max_abs: float
    gradient_zero_fraction: float
    parameter_count: int


@dataclass(frozen=True)
class UpdateDiagnostics:
    """Measurements of one optimizer update."""

    layer_index: int
    name: str
    parameter_norm_before: float
    update_norm: float
    update_to_parameter_ratio: float


def _validated_array(value, name):
    array = np.asarray(value, dtype=np.float64)

    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")

    return array


def _norm(value):
    """Stable Euclidean norm."""
    array = _validated_array(value, "array")
    return float(np.linalg.norm(array.ravel()))


def _dense_layers(model):
    if not hasattr(model, "layers"):
        raise TypeError("Expected a Sequential-like model.")

    layers = [
        layer for layer in model.layers
        if isinstance(layer, Dense)
    ]

    if not layers:
        raise ValueError("Model has no Dense layers.")

    return layers


def collect_gradient_diagnostics(model):
    """Collect layer-wise statistics after backpropagation.

    Call model.backward(...) before invoking this function.
    """

    results = []

    for index, layer in enumerate(_dense_layers(model), start=1):
        if layer.dW is None or layer.db is None:
            raise RuntimeError(
                "Run backward() before collecting gradients."
            )

        weights = _validated_array(layer.weights, "weights")
        biases = _validated_array(layer.biases, "biases")
        dw = _validated_array(layer.dW, "weight gradients")
        db = _validated_array(layer.db, "bias gradients")

        if weights.shape != dw.shape:
            raise ValueError("Weight gradient shape mismatch.")

        if biases.shape != db.shape:
            raise ValueError("Bias gradient shape mismatch.")

        gradients = np.concatenate([
            dw.ravel(),
            db.ravel(),
        ])

        results.append(
            LayerDiagnostics(
                layer_index=index,
                name=f"Dense_{index}",
                weight_norm=_norm(weights),
                bias_norm=_norm(biases),
                weight_gradient_norm=_norm(dw),
                bias_gradient_norm=_norm(db),
                gradient_norm=_norm(gradients),
                gradient_mean_abs=float(
                    np.mean(np.abs(gradients))
                ),
                gradient_max_abs=float(
                    np.max(np.abs(gradients))
                ),
                gradient_zero_fraction=float(
                    np.mean(gradients == 0)
                ),
                parameter_count=int(
                    weights.size + biases.size
                ),
            )
        )

    return results


def snapshot_parameters(model):
    """Copy parameters before an optimizer update."""

    snapshots = []

    for layer in _dense_layers(model):
        snapshots.append((
            _validated_array(layer.weights, "weights").copy(),
            _validated_array(layer.biases, "biases").copy(),
        ))

    return snapshots


def collect_update_diagnostics(model, before):
    """Measure parameter changes after optimizer.step(...).

    The `before` argument must come from snapshot_parameters(model)
    immediately before the optimizer update.
    """

    layers = _dense_layers(model)

    if len(before) != len(layers):
        raise ValueError("Snapshot layer count mismatch.")

    results = []

    for index, (layer, snapshot) in enumerate(
        zip(layers, before), start=1
    ):
        if len(snapshot) != 2:
            raise ValueError("Invalid parameter snapshot.")

        old_w = _validated_array(snapshot[0], "snapshot weights")
        old_b = _validated_array(snapshot[1], "snapshot biases")
        new_w = _validated_array(layer.weights, "weights")
        new_b = _validated_array(layer.biases, "biases")

        if old_w.shape != new_w.shape:
            raise ValueError("Snapshot weight shape mismatch.")

        if old_b.shape != new_b.shape:
            raise ValueError("Snapshot bias shape mismatch.")

        old_vector = np.concatenate([
            old_w.ravel(),
            old_b.ravel(),
        ])

        update_vector = np.concatenate([
            (new_w - old_w).ravel(),
            (new_b - old_b).ravel(),
        ])

        parameter_norm = _norm(old_vector)
        update_norm = _norm(update_vector)

        # A zero-norm parameter vector has no meaningful
        # relative update ratio.
        ratio = (
            update_norm / parameter_norm
            if parameter_norm > 0
            else (0.0 if update_norm == 0 else math.inf)
        )

        results.append(
            UpdateDiagnostics(
                layer_index=index,
                name=f"Dense_{index}",
                parameter_norm_before=parameter_norm,
                update_norm=update_norm,
                update_to_parameter_ratio=float(ratio),
            )
        )

    return results


def gradient_flow_summary(layer_diagnostics):
    """Summarize gradient magnitudes across Dense layers."""

    if not layer_diagnostics:
        raise ValueError("No layer diagnostics supplied.")

    norms = np.array([
        item.gradient_norm
        for item in layer_diagnostics
    ], dtype=np.float64)

    if not np.all(np.isfinite(norms)):
        raise ValueError("Non-finite gradient norm.")

    positive = norms[norms > 0]

    return {
        "layer_count": len(norms),
        "minimum_gradient_norm": float(np.min(norms)),
        "maximum_gradient_norm": float(np.max(norms)),
        "mean_gradient_norm": float(np.mean(norms)),
        "max_to_min_positive_ratio": (
            float(np.max(positive) / np.min(positive))
            if len(positive) else None
        ),
        "all_gradients_zero": bool(np.all(norms == 0)),
    }


def stability_warnings(
    layer_diagnostics,
    update_diagnostics=None,
    vanishing_threshold=1e-8,
    exploding_threshold=100.0,
    large_update_ratio=0.1,
):
    """Return heuristic warnings about potentially unstable training.

    Thresholds depend on model scale, loss reduction, and batch size.
    A warning is a diagnostic signal, not proof of a training failure.
    """

    thresholds = (
        vanishing_threshold,
        exploding_threshold,
        large_update_ratio,
    )

    if any(
        not np.isfinite(value) or value <= 0
        for value in thresholds
    ):
        raise ValueError(
            "Diagnostic thresholds must be finite and positive."
        )

    warnings = []

    for item in layer_diagnostics:
        if item.gradient_norm < vanishing_threshold:
            warnings.append({
                "layer": item.name,
                "type": "small_gradient",
                "value": item.gradient_norm,
                "message": (
                    "Gradient norm is below the configured "
                    "small-gradient threshold."
                ),
            })

        if item.gradient_norm > exploding_threshold:
            warnings.append({
                "layer": item.name,
                "type": "large_gradient",
                "value": item.gradient_norm,
                "message": (
                    "Gradient norm exceeds the configured "
                    "large-gradient threshold."
                ),
            })

    if update_diagnostics is not None:
        for item in update_diagnostics:
            ratio = item.update_to_parameter_ratio

            if not np.isfinite(ratio):
                warnings.append({
                    "layer": item.name,
                    "type": "undefined_relative_update",
                    "value": None,
                    "message": (
                        "Relative update is undefined because "
                        "the original parameter norm is zero."
                    ),
                })

            elif ratio > large_update_ratio:
                warnings.append({
                    "layer": item.name,
                    "type": "large_relative_update",
                    "value": ratio,
                    "message": (
                        "Parameter update is large relative "
                        "to the previous parameter magnitude."
                    ),
                })

    return warnings


def diagnostics_to_dict(items):
    """Convert diagnostic records into JSON-compatible dictionaries."""
    return [asdict(item) for item in items]
