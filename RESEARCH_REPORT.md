
# NeuralCore — Research Report

## Neural Networks from Scratch: Optimization, Training Dynamics, and Generalization

**Project:** NeuralCore  
**Implementation:** Python and NumPy  
**Research approach:** Controlled computational experiments  
**Development environment:** Python 3.13.5

---

## 1. Abstract

NeuralCore investigates neural-network optimization and generalization through a neural-network framework implemented from scratch using NumPy.

The framework implements dense layers, activation functions, forward propagation, backpropagation, binary cross-entropy loss, and three optimization algorithms: stochastic gradient descent (SGD), Momentum, and Adam.

The research evaluates optimizer performance, hyperparameter sensitivity, gradient behaviour, and early stopping using controlled experiments on synthetic classification datasets.

An initial optimizer benchmark showed substantial differences under fixed learning rates. Following learning-rate tuning and evaluation on separate seeds, these differences narrowed. A paired early-stopping experiment across 20 random seeds found improvements in mean test accuracy and binary cross-entropy loss relative to training for a fixed 500 epochs.

The implementation is supported by 216 passing automated tests and an initial reproducibility audit with 17 passing checks.

The findings demonstrate the importance of controlled experimental design, validation-based model selection, and transparent statistical reporting.

---

## 2. Research Questions

This project investigates four primary questions:

**RQ1:** How accurately can fundamental neural-network training operations be implemented and verified without automatic differentiation?

**RQ2:** How do SGD, Momentum, and Adam compare under fixed hyperparameters, and how does learning-rate tuning affect their relative performance?

**RQ3:** What can gradient-flow diagnostics reveal about neural-network training behaviour?

**RQ4:** Does validation-based early stopping improve generalization compared with using the final checkpoint after a fixed training budget?

---

## 3. Neural Network Implementation

### 3.1 Core Architecture

NeuralCore uses a sequential architecture composed of fully connected dense layers and activation functions.

For a dense layer:

\[
Z = XW + b
\]

where:

- \(X\) represents the input activations.
- \(W\) represents the weight matrix.
- \(b\) represents the bias vector.
- \(Z\) represents the pre-activation output.

Nonlinear activation functions transform these values to enable learning nonlinear decision boundaries.

### 3.2 Backpropagation

The network computes gradients using the chain rule.

For a dense layer, given the upstream gradient with respect to its output:

\[
\frac{\partial L}{\partial W}
=
X^T \frac{\partial L}{\partial Z}
\]

\[
\frac{\partial L}{\partial b}
=
\sum_i \frac{\partial L}{\partial Z_i}
\]

\[
\frac{\partial L}{\partial X}
=
\frac{\partial L}{\partial Z} W^T
\]

These gradients propagate backward through the network and are used by the optimization algorithms.

The implementation is accompanied by automated tests, including numerical gradient verification.

### 3.3 Binary Classification Loss

Binary classification experiments use binary cross-entropy with logits.

For a target \(y\) and logit \(z\), the numerically stable per-example expression is:

\[
L(z,y)
=
\max(z,0)-zy+\log(1+e^{-|z|})
\]

This formulation avoids explicitly calculating logarithms of probabilities close to zero or one.

---

## 4. Experimental Data and Validation

### 4.1 Synthetic Datasets

The project includes XOR and two-moons classification problems.

XOR demonstrates the need for nonlinear representations, while the two-moons dataset provides a nonlinear decision-boundary problem suitable for comparing optimizers and training strategies.

### 4.2 Data Partitioning

For the audited dataset configuration, 600 examples are divided into:

| Partition | Examples | Proportion |
|---|---:|---:|
| Training | 420 | 70% |
| Validation | 90 | 15% |
| Test | 90 | 15% |

The feature scaler is fitted using the training partition and then applied to the validation and test partitions.

This avoids incorporating validation or test feature statistics into the training preprocessing step.

### 4.3 Random Seeds

Seed-controlled dataset generation, data splitting, and model initialization support repeatable experimental configurations.

Multiple random seeds are used to examine variation across runs rather than relying exclusively on a single result.

---

## 5. Optimization Algorithms

### 5.1 Stochastic Gradient Descent

The SGD update is:

\[
\theta_{t+1}
=
\theta_t-\eta g_t
\]

where \(\eta\) is the learning rate and \(g_t\) is the gradient.

### 5.2 Momentum

Momentum maintains a running update direction based on current and previous gradients.

Its purpose is to incorporate information from preceding optimization steps rather than treating each gradient independently.

### 5.3 Adam

Adam maintains moving estimates of first- and second-order gradient moments.

It adapts parameter updates using these estimates and bias correction.

All three algorithms are implemented in the NeuralCore optimization module.

---

## 6. Experiment A — Initial Optimizer Benchmark

### 6.1 Objective

The initial benchmark evaluates SGD, Momentum, and Adam using predetermined learning rates.

### 6.2 Configuration

The recorded benchmark used:

| Optimizer | Learning rate |
|---|---:|
| SGD | 0.05 |
| Momentum | 0.01 |
| Adam | 0.01 |

The benchmark was evaluated across 10 random seeds.

### 6.3 Results

| Optimizer | Mean test accuracy |
|---|---:|
| SGD | 87.44% |
| Momentum | 90.89% |
| Adam | 97.78% |

### 6.4 Interpretation

Adam achieved the highest mean accuracy under these specific learning rates.

However, this result alone does not establish that Adam is inherently superior.

The algorithms were not necessarily operating at equally suitable learning rates.

This motivated a separate hyperparameter-tuning experiment.

---

## 7. Experiment B — Hyperparameter Selection and Evaluation

### 7.1 Objective

This experiment investigates whether the apparent optimizer ranking changes when learning rates are selected through a controlled tuning process.

### 7.2 Method

Learning-rate selection and final evaluation use separate seed groups.

This separates hyperparameter selection from the final reported evaluation.

### 7.3 Selected Learning Rates

| Optimizer | Selected learning rate |
|---|---:|
| SGD | 0.20 |
| Momentum | 0.10 |
| Adam | 0.03 |

### 7.4 Final Evaluation

The recorded final evaluation used 20 seeds.

| Optimizer | Mean test accuracy |
|---|---:|
| SGD | 97.00% |
| Momentum | 97.61% |
| Adam | 97.44% |

### 7.5 Interpretation

After tuning, the performance gap between optimizers became much smaller.

Momentum had the highest recorded mean accuracy, but the differences were narrow.

The paired comparisons did not establish a clear universal winner.

**Research finding:** Optimizer comparisons can be strongly influenced by hyperparameter selection. A fixed-learning-rate comparison should not be treated as definitive evidence of algorithm superiority.

---

## 8. Experiment C — Gradient-Flow Diagnostics

### 8.1 Objective

This experiment examines the behaviour of gradients and parameter updates during training.

### 8.2 Measurements

The diagnostic system records:

- Layer-wise gradient norms.
- Parameter update-to-weight ratios.
- Training and validation losses.
- Diagnostic warnings for selected abnormal training conditions.

### 8.3 Recorded Configuration

The recorded diagnostic experiment used Adam with learning rate 0.01 for 500 epochs, with seed 42.

### 8.4 Results

| Measurement | Recorded value |
|---|---:|
| Final training BCE | 0.042921 |
| Final validation BCE | 0.071591 |
| Best validation BCE | 0.059044 |
| Best validation epoch | 256 |

The recorded final layer-wise gradient norms were approximately:

| Dense layer | Gradient norm |
|---|---:|
| Layer 1 | 0.01374 |
| Layer 2 | 0.00451 |
| Layer 3 | 0.00179 |

The experiment recorded no diagnostic warning counts under its configured warning rules.

### 8.5 Interpretation

Validation loss reached its minimum before the final training epoch.

This indicates that the final checkpoint was not the best checkpoint according to validation BCE.

The result motivated the subsequent investigation of validation-based early stopping.

Gradient norms varied between layers, but these observations alone do not prove the presence or absence of all optimization pathologies.

---

## 9. Experiment D — Early Stopping

### 9.1 Research Question

Does selecting a checkpoint using validation loss improve held-out test performance relative to taking the final checkpoint after a fixed training budget?

### 9.2 Experimental Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.03 |
| Maximum epochs | 500 |
| Patience | 50 |
| Minimum improvement | 0.0 |
| Random seeds | 20 |
| Bootstrap resamples | 10,000 |

The experiment used seeds 1000 through 1019.

### 9.3 Paired Comparison Design

For each seed, one training trajectory was generated for the full 500 epochs.

Two checkpoints from that trajectory were compared:

**Fixed:** The model parameters at epoch 500.

**Early stopping:** The best validation-selected checkpoint recorded before the first patience-exhaustion event.

The held-out test partition was used to evaluate these checkpoints after training, not to choose the checkpoint.

Using checkpoints from the same trajectory helps control for differences in initialization and training data within each seed.

### 9.4 Mean Results

| Metric | Fixed | Early stopping |
|---|---:|---:|
| Test accuracy | 0.96944 | 0.97611 |
| F1 score | 0.96934 | 0.97610 |
| Test BCE | 0.12666 | 0.07327 |
| Simulated epochs run | 500.0 | 231.9 |
| Selected checkpoint epoch | 500.0 | 181.9 |

### 9.5 Paired Statistical Results

The paired difference is defined as:

**Early stopping minus fixed training.**

| Metric | Mean difference | 95% bootstrap CI |
|---|---:|---|
| Accuracy | +0.00667 | [+0.00167, +0.01222] |
| F1 | +0.00675 | [+0.00185, +0.01238] |
| Test BCE | −0.05339 | [−0.08259, −0.02687] |
| Simulated epochs | −268.1 | [−316.90, −212.85] |

The accuracy improvement corresponds to approximately **+0.67 percentage points**.

Mean test BCE decreased by approximately **42.2%**, calculated relative to the fixed-training mean.

### 9.6 Interpretation

The paired bootstrap intervals for accuracy, F1, and test BCE exclude zero.

This provides evidence of improved held-out performance for early stopping in the evaluated configuration.

However, the experiment is limited to 20 seeds and the chosen synthetic classification task.

### 9.7 Computational Limitation

The experiment intentionally continued training to epoch 500 even when the early-stopping criterion would have triggered earlier.

Therefore, the mean reduction of 268.1 epochs represents **potential training epochs avoided**, not directly measured computational savings.

A separate experiment that actually terminates training at the stopping event would be required to measure runtime savings.

---

## 10. Reproducibility and Software Verification

### 10.1 Automated Tests

The project includes tests covering neural-network operations, numerical correctness, optimization utilities, early stopping, experiment logic, and reproducibility-audit functionality.

At the recorded development checkpoint:

**216 tests passed in 6.84 seconds.**

### 10.2 Reproducibility Audit

An initial audit evaluated:

- Deterministic synthetic-data generation.
- Deterministic dataset splitting.
- Model-initialization reproducibility.
- Partition-value overlap checks.
- Training-only scaling statistics.
- Selected saved-artifact consistency.
- Early-stopping paired-result arithmetic.
- Expected visualization availability.

The recorded result was:

**17 of 17 checks passed.**

### 10.3 Scope of Verification

Passing tests and audit checks supports confidence in the examined implementation behaviours.

It does not constitute a mathematical proof that every component is correct.

The audit also does not independently repeat every full training experiment.

---

## 11. Research Limitations

Several limitations affect interpretation:

1. **Synthetic data:** The experiments do not establish generalization to real-world datasets.

2. **Limited architectures:** The findings are specific to the investigated network configurations.

3. **Hyperparameter dependence:** Optimizer results depend on the learning-rate search procedure and other training settings.

4. **Finite seed samples:** Bootstrap confidence intervals reflect the evaluated paired-seed distribution and resampling procedure.

5. **Early-stopping runtime:** The comparison estimates potential epoch savings rather than measuring actual execution-time savings.

6. **Reproducibility scope:** Independent full-training repeatability has been verified for three seeds within the recorded environment. Cross-platform numerical agreement remains unverified.

7. **Diagnostic scope:** Gradient measurements provide useful indicators but do not establish complete training stability.

---

## 12. Future Work

Potential extensions include:

- Cross-platform and clean-environment reproducibility verification.
- Real-world classification datasets.
- Additional network architectures and activation comparisons.
- Direct measurement of early-stopping runtime savings.
- Wider hyperparameter-search configurations.
- Additional numerical stability and edge-case testing.
- More extensive experiment metadata and environment tracking.

---

## 13. Conclusion

NeuralCore demonstrates the implementation and experimental evaluation of neural networks without relying on automatic differentiation frameworks.

The research highlights three findings:

**First**, optimizer performance depends strongly on hyperparameter selection. Apparent differences under fixed learning rates narrowed after controlled tuning.

**Second**, gradient-flow and validation-loss diagnostics can identify training behaviours that are not apparent from training accuracy alone.

**Third**, validation-based early stopping improved mean held-out accuracy and test BCE in a 20-seed paired experiment, with bootstrap confidence intervals supporting the observed differences.

Together, the implementation, experiments, automated tests, and reproducibility audit form a foundation for further research into neural-network optimization and generalization.

---

## 14. Reproduction

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run automated tests:

```powershell
python -m pytest -q
```

Run the principal experiments:

```powershell
python -m experiments.optimizer_benchmark
python -m experiments.hyperparameter_search
python -m experiments.gradient_flow
python -m experiments.early_stopping_comparison
```

Run the audit:

```powershell
python -m experiments.reproducibility_audit
```

Detailed environment and reproduction instructions are maintained in `REPRODUCIBILITY.md`.


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




