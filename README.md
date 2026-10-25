# ML Observability

A production-oriented ML observability service for detecting numerical data drift, evaluating detector behavior, and supporting evidence-based incident triage.

The project emphasizes reproducible evaluation, explicit limitations, automated testing, and a clean separation between drift detection and API delivery.

## Features

- Numerical drift detection using the two-sample Kolmogorov–Smirnov (KS) test.
- Benjamini–Hochberg false-discovery-rate correction across evaluated numerical features.
- Input validation and explicit handling of insufficient samples.
- FastAPI endpoints for drift analysis and health checks.
- Synthetic evaluations with known labels and multiple distribution-change scenarios.
- Multi-seed evaluation with 95% Wilson confidence intervals for selected metrics.
- Automated tests, linting, and formatting checks.

## Architecture

The service separates the detector from its API layer:

1. **Drift detection:** compares reference and current numerical feature distributions.
2. **Statistical correction:** adjusts feature-level significance decisions for multiple testing.
3. **API layer:** exposes detection results through typed HTTP endpoints.
4. **Evaluation layer:** measures detector behavior against synthetic scenarios with known ground truth.
5. **Tests and quality checks:** validate expected behavior, input handling, reproducibility, and regression safety.

The detector does not automatically retrain, modify, or deploy models. Drift alerts are signals for further investigation, not proof of model degradation or root cause.

## Technology stack

- Python 3.13
- FastAPI and Uvicorn
- NumPy and pandas
- SciPy
- Pydantic
- pytest
- Ruff

## Getting started

### 1. Create and activate a virtual environment

PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

Install the project and development dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 3. Run the API

```powershell
uvicorn ml_observability.api:app --reload
```

The interactive API documentation is available at:

- http://127.0.0.1:8000/docs

### 4. Run automated tests and quality checks

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

## Synthetic evaluation

The repository contains multiple evaluations. Each measures a different aspect of detector behavior; their results should not be treated as interchangeable.

### Baseline drift evaluation

Run:

```powershell
python scripts/evaluate_drift.py
```

The recorded baseline experiment reported:

| Metric | Result |
|---|---:|
| Shifted-feature detection rate | 1.00 |
| Unchanged-feature false-positive rate | 0.01 |
| False positives | 9 of 900 unchanged-feature tests |

These are results from one synthetic configuration, not production performance guarantees.

### Labeled scenario evaluation

Run:

```powershell
python scripts/evaluate_scenarios.py
```

This evaluation compares detector output against known synthetic labels and reports a confusion matrix and classification metrics.

The recorded experiment used 100 trials, 200 samples, five numerical features, a location shift of `0.8`, and significance level `0.05`.

| Metric | Observed result |
|---|---:|
| True positives | 100 |
| False positives | 0 |
| True negatives | 400 |
| False negatives | 0 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1 | 1.00 |
| False-positive rate | 0.00 |

These perfect scores apply only to this particular synthetic experiment. They do not establish generalization to real-world data.

### Distribution-change scenario evaluation

Run the multi-seed evaluation:

```powershell
python scripts/evaluate_drift_scenarios.py
```

By default, the evaluator uses seeds `42`, `123`, and `2026`, with 100 trials per seed, 200 samples per feature, five numerical features, and significance level `0.05`.

To specify the parameters explicitly:

```powershell
python scripts/evaluate_drift_scenarios.py --seeds 42 123 2026 --trials 100 --sample-size 200
```

The evaluation includes five scenarios:

- **No drift:** all features follow the reference distribution.
- **Location shift:** the target feature's mean is shifted.
- **Variance increase:** the target feature's variance is increased.
- **Heavy-tailed:** the target feature follows a variance-normalized Student's t distribution.
- **Skewed:** the target feature follows a centered, right-skewed exponential distribution.

In each positive scenario, only `feature_0` is deliberately changed. The other four features act as unchanged controls.

#### Recorded results

The following results were obtained with three seeds and 100 trials per seed, giving 300 trials per scenario.

| Scenario | Target detections | Recall | 95% recall CI | Feature-level FPR |
|---|---:|---:|---:|---:|
| No drift | Not applicable | Not applicable | Not applicable | 0.53% |
| Location shift | 300/300 | 100.0% | 98.7%–100.0% | 1.50% |
| Variance increase | 56/300 | 18.7% | 14.7%–23.5% | 0.83% |
| Heavy-tailed | 69/300 | 23.0% | 18.6%–28.1% | 0.75% |
| Skewed | 234/300 | 78.0% | 73.0%–82.3% | 0.92% |

The reported recall interval is the 95% Wilson score interval for target-feature detection across trials. Recall is not defined for the no-drift scenario because it has no positive ground-truth examples.

In the no-drift scenario, there were eight false-positive feature predictions among 1,500 unchanged-feature evaluations. Eight of 300 trials had at least one false-positive control feature, resulting in a **2.67% trial-level family-wise false-positive rate** (95% Wilson CI: 1.36%–5.17%).

#### Interpretation

The detector was highly sensitive to the tested location shift at this sample size. Detection recall was substantially lower for variance increases and heavy-tailed data, while the tested skewed distribution had intermediate-to-high recall.

These results demonstrate that detection power depends on the nature of the distribution change, not only on sample size and significance level. A detector that is sensitive to location shifts may be less sensitive to other distributional changes.

#### Metric definitions

- **Precision:** true positives divided by all positive predictions.
- **Recall:** true positives divided by all positive ground-truth cases.
- **F1:** harmonic mean of precision and recall.
- **Feature-level false-positive rate:** false-positive feature predictions divided by all unchanged-feature predictions.
- **Family-wise false-positive rate:** trials with at least one false-positive control feature divided by all no-change control trials.
- **95% Wilson confidence interval:** an interval for a binomial proportion that behaves better than the basic normal approximation when observed proportions are near zero or one.

Feature-level FPR and family-wise FPR use different denominators and should not be conflated. Feature outcomes within a trial may be dependent because the detector applies multiple-testing correction.

#### Limitations of the experiment

- Results come from seeded synthetic distributions, not production datasets.
- Each positive scenario changes only one feature; simultaneous changes across many features may behave differently.
- The experiment covers a single sample size, significance level, and set of distribution parameters.
- Wilson intervals quantify binomial sampling uncertainty under the assumed trial model. They do not account for scenario selection, model assumptions, or mismatch with production data.
- Distribution drift does not necessarily imply degraded predictive performance.
- These evaluations do not measure downstream model quality, business impact, alerting effectiveness, or root-cause accuracy.

Treat these results as a reproducible baseline, not a production performance guarantee.

### Drift evaluation matrix

Run the full matrix:

```powershell
python scripts/evaluate_drift_matrix.py
```

The matrix evaluates location-shift detection over combinations of shift magnitude, sample size, and random seed. It reports detection metrics for each configuration to make the relationship between effect size and sample size explicit.

The recorded matrix covered:

- Shifts: `0`, `0.2`, `0.5`, `0.8`, and `1.2`
- Sample sizes: `50`, `100`, `200`, and `500`
- Seeds: `42`, `123`, and `2026`
- Five numerical features
- Significance level: `0.05`
- 100 trials per configuration

The matrix contains 60 configurations and 6,000 trials overall.

Selected observed recall ranges across the three seeds:

| Location shift | Sample size | Observed recall range |
|---|---:|---:|
| 0.2 | 50 | 1%–3% |
| 0.2 | 100 | 7%–11% |
| 0.2 | 200 | 15%–19% |
| 0.2 | 500 | 49%–60% |
| 0.5 | 50 | 16%–26% |
| 0.5 | 100 | 56%–63% |
| 0.5 | 200 | 92%–98% |
| 0.5 | 500 | 100% |
| 0.8 | 50 | 70%–73% |
| 0.8 | 100 | 99%–100% |
| 0.8 | 200 | 100% |
| 0.8 | 500 | 100% |
| 1.2 | 50 | 98%–100% |
| 1.2 | 100 or more | 100% |

Across zero-shift controls, the observed false-positive rate ranged from `0.000` to `0.004` at the individual-seed configuration level. Positive-shift configurations had zero observed false positives in this run.

These measurements are finite-sample observations from synthetic normal distributions with location shifts. They do not establish performance on non-normal distributions, variance changes, real production data, or downstream model degradation. Three seeds provide only a limited view of variability; observed rates are not guarantees.

## Statistical and operational limitations

- Statistical significance of a distribution change does not establish that model performance has degraded.
- Detection power depends on effect size, sample size, distribution characteristics, and threshold settings.
- Multiple features require appropriate multiple-testing considerations.
- Synthetic evaluation cannot substitute for validation on representative data.
- Incident recommendations are investigation steps, not confirmed root causes.
- The service does not automatically retrain, modify, or deploy models.
- Before production use, calibrate thresholds on representative data and measure latency, resource usage, alert quality, and failure behavior.
- Confidence intervals describe uncertainty under their statistical assumptions; they do not guarantee production outcomes.

## Development principles

- Keep the detector independent from the API layer.
- Make evaluation parameters and random seeds explicit.
- Test negative cases, failure paths, and invalid inputs.
- Avoid logging raw sensitive data.
- Treat benchmarks as evidence, not guarantees.
- Validate changes with automated tests and quality checks before merging.

## Roadmap

- Extend evaluation to additional variance and non-normal distribution changes.
- Measure runtime and memory consumption across sample sizes and feature counts.
- Expand testing of multiple-testing behavior and correlated features.
- Add operational observability and incident-triage capabilities.
- Validate against representative, appropriately governed real-world data.
- Evaluate alert quality and downstream model-performance impact.

## License

No license has been specified yet. Add a `LICENSE` file before presenting the repository as open source.
