
# NeuralCore — Reproducibility Guide

## 1. Purpose

This document describes how to recreate the NeuralCore Python environment, execute its automated tests, rerun its experiments, and inspect the resulting research artifacts.

It also distinguishes verified reproducibility checks from validation work that remains outstanding.

## 2. Recorded Development Environment

| Component | Recorded version |
|---|---|
| Operating system | Windows |
| Python | 3.13.5 |
| NumPy | 2.5.4 |
| Matplotlib | 3.11.2 |
| Pytest | 9.2.0 |
| Pillow | 13.0.0 |

The full recorded Python package environment is stored in `requirements.txt`.

The requirements file was generated using `pip freeze`, so it includes direct dependencies and transitive dependencies present in the development environment.

Exact Python package versions improve repeatability but do not guarantee identical numerical results across operating systems, hardware, or numerical-library builds.

## 3. Environment Setup

Open PowerShell in the NeuralCore project root.

Create a virtual environment:

```powershell
py -3.13 -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the recorded dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify the Python interpreter:

```powershell
python --version
```

Verify key dependencies:

```powershell
python -c "import numpy, matplotlib, pytest; print('NumPy:', numpy.__version__); print('Matplotlib:', matplotlib.__version__); print('Pytest:', pytest.__version__)"
```

Run commands from the repository root so that the `neuralcore` and `experiments` modules can be imported correctly.

## 4. Automated Test Suite

Run all tests:

```powershell
python -m pytest -q
```

At the documented development checkpoint:

**216 tests passed in 9.03 seconds.**

This is a recorded result, not a guarantee that every future environment or code revision will produce the same test count.

## 5. Experiments

### 5.1 XOR Training

```powershell
python -m experiments.xor_training
```

### 5.2 Two-Moons Training

```powershell
python -m experiments.moons_training
```

### 5.3 Optimizer Trajectories

```powershell
python -m experiments.optimizer_trajectories
```

### 5.4 Loss Landscape

```powershell
python -m experiments.loss_landscape
```

### 5.5 Optimizer Benchmark

```powershell
python -m experiments.optimizer_benchmark
```

### 5.6 Hyperparameter Search

```powershell
python -m experiments.hyperparameter_search
```

### 5.7 Gradient Diagnostics

```powershell
python -m experiments.gradient_flow
```

### 5.8 Early Stopping Comparison

```powershell
python -m experiments.early_stopping_comparison
```

The early-stopping experiment uses 20 seeds, a maximum of 500 epochs, and 10,000 bootstrap resamples.

Its results are stored in:

`outputs/early_stopping_comparison/`

The experiment deliberately executes the full training budget for both checkpoint comparisons.

Consequently, reported early-stopping epoch savings are potential savings, not measured runtime reductions.

## 6. Randomness and Experimental Controls

NeuralCore uses explicit random seeds for supported dataset-generation, splitting, and model-initialization operations.

The optimizer comparisons use repeated seeds to evaluate run-to-run variability.

The hyperparameter experiment separates learning-rate selection from its final evaluation seed group.

The early-stopping experiment uses seeds 1000 through 1019, with paired checkpoints from the same training trajectory for each seed.

Seed control is important for reproducibility, but identical seeds alone do not establish identical results across different computing environments.

## 7. Data Leakage Controls

The recorded data-preparation workflow separates training, validation, and test partitions.

Feature scaling is fitted on training data and applied to the other partitions.

Validation data is used for model selection and early-stopping decisions.

Held-out test data is used for final evaluation rather than checkpoint selection.

These controls reduce common forms of data leakage. They do not replace a complete source-code review of every experiment.

## 8. Reproducibility Audit

Run:

```powershell
python -m experiments.reproducibility_audit
```

The audit writes:

`outputs/reproducibility_audit/audit_report.json`

At the documented development checkpoint:

**17/17 audit checks passed.**

The audit checks:

- Deterministic synthetic-data generation.
- Deterministic train/validation/test splitting.
- Deterministic model initialization.
- Value-based overlap between data partitions.
- Training-only feature-scaling statistics.
- Selected optimizer-benchmark artifact consistency.
- Early-stopping paired-result consistency.
- Selected figure-file availability.

### Audit Limitations

The audit does not:

- Independently rerun every full training experiment.
- Prove cross-platform numerical reproducibility.
- Establish that all experiment configurations are scientifically optimal.
- Demonstrate real-world generalization.
- Independently verify every statistical assumption.

These limitations should be considered when interpreting the audit result.

## 9. Research Artifacts

Generated outputs are stored in the `outputs/` directory.

The recorded project includes:

| Directory | Purpose |
|---|---|
| `outputs/moons/` | Two-moons experiment outputs |
| `outputs/landscape/` | Loss-landscape visualizations |
| `outputs/optimizer_comparison/` | Optimizer trajectory visualizations |
| `outputs/optimizer_benchmark/` | Multi-seed optimizer benchmark |
| `outputs/hyperparameter_search/` | Hyperparameter search and evaluation |
| `outputs/gradient_diagnostics/` | Gradient-flow diagnostics |
| `outputs/early_stopping_comparison/` | Paired early-stopping results |
| `outputs/reproducibility_audit/` | Audit report |

Depending on the experiment, artifacts include CSV data, JSON summaries, and PNG visualizations.

## 10. Checking Results

After rerunning an experiment, compare the generated numerical outputs with the recorded reference results.

For exact reproducibility checks, compare the relevant CSV and JSON numerical fields rather than relying only on visual similarity between plots.

Where numerical agreement is approximate, record the tolerance and the environment used.

Do not treat differences in image metadata or PNG encoding as proof that the underlying numerical experiment changed.

## 11. Known Reproducibility Limitations

The following verification work remains outstanding:

1. Independent full-training reruns in separate clean environments compared against saved numerical reference values.
2. Reproducibility verification across separate clean environments.
3. Cross-platform testing.
4. Direct runtime measurements for early stopping.
5. Additional experiments on external datasets.

## 12. Recommended Verification Sequence

For a fresh environment:

1. Install Python 3.13.
2. Create and activate a virtual environment.
3. Install `requirements.txt`.
4. Run the automated test suite.
5. Execute the principal experiments.
6. Run the reproducibility audit.
7. Inspect saved CSV and JSON artifacts.
8. Compare numerical results against the recorded reference values.
9. Document any discrepancies, including the Python and package versions used.

## 13. Research Integrity

NeuralCore distinguishes between:

- **Implementation verification:** Automated tests of code behaviour.
- **Artifact consistency:** Checks that saved reports agree with selected underlying outputs.
- **Statistical evaluation:** Multi-seed comparisons and bootstrap confidence intervals.
- **Full experimental reproducibility:** Independently rerunning complete experiments and comparing their numerical results.

The first three have been addressed to varying degrees in the current project.

Independent full-training repeatability has been verified for three seeds within one environment. Broader cross-platform reproducibility remains an open objective.

## 14. Conclusion

NeuralCore provides a documented environment, automated tests, controlled experiment configurations, and a reproducibility audit.

These measures support repeatability and transparent research reporting.

Further independent training reruns and clean-environment verification are necessary before making stronger reproducibility claims.


## Independent Full-Training Reproducibility

NeuralCore independently trained two models for each of three
random seeds (42, 43, and 44), using Adam with learning rate
0.03 for 300 epochs.

All three independent seed pairs passed seven exact numerical
comparisons covering training loss histories, validation loss
histories, final parameters, test logits, test labels, test loss,
and classification metrics.

**Result: 3/3 seed pairs passed; 21/21 exact checks passed.**

Different-seed sensitivity also passed.

The experiment was conducted using Python 3.13.5 and
NumPy 2.5.4 on Windows 11.

These findings establish exact repeatability for this tested
configuration within the recorded environment, not across
different platforms or architectures.

Run the experiment:

``powershell
python -m experiments.full_training_reproducibility
``

Results are stored in:
`outputs/full_training_reproducibility/full_training_report.json`



