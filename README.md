# ML Observability

A production-oriented ML monitoring service for detecting data-quality failures and numerical feature drift, generating evidence-backed incidents, and recommending investigation steps.

The project emphasizes reproducible evaluation, explicit failure handling, typed API contracts, automated tests, and operational limitations. All evaluation data is synthetic; no proprietary data or cloud credentials are required.

## Features

- **Data validation:** detect missing values, unexpected columns, and invalid inputs.
- **Numerical drift detection:** compare reference and current feature distributions using the two-sample Kolmogorov–Smirnov test.
- **Incident generation:** produce structured incidents with severity, supporting evidence, and recommended investigation steps.
- **REST API:** expose monitoring functionality through FastAPI.
- **Reproducible evaluation:** measure detection quality on controlled, labeled scenarios and across multiple drift magnitudes, sample sizes, and random seeds.
- **Automated quality checks:** validate behavior with pytest and enforce code quality with Ruff.

## Architecture

The monitoring pipeline separates responsibilities into distinct components:

1. Input validation checks dataset structure and supported inputs.
2. The drift detector evaluates feature distributions.
3. The evidence builder captures the results needed to explain detections.
4. The triage layer generates structured incidents and investigation recommendations.
5. The FastAPI layer exposes the service through validated request and response contracts.
6. The evaluation suite measures detection behavior against synthetic ground truth.

Detection logic remains independent of the API layer so it can be tested and reused directly.

See [`docs/architecture.md`](docs/architecture.md) for the architecture and design constraints.

## Technology stack

- Python 3.13
- FastAPI and Pydantic
- pandas, NumPy, and SciPy
- pytest
- Ruff
- GitHub Actions

## Getting started

### 1. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
```

### 2. Install dependencies

Install the project and its development dependencies using the repository's configured package metadata:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 3. Run the API

```powershell
uvicorn ml_observability.api:app --reload
```

Use the application's `/docs` endpoint to inspect the available API operations and request schemas.

### 4. Run automated tests and quality checks

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

## Synthetic evaluation

The evaluation suite measures drift detection under controlled conditions. Results are specific to the stated data-generating process, sample sizes, significance threshold, and random seeds.

### Baseline drift evaluation

Run:

```powershell
python scripts/evaluate_drift.py
```

The baseline experiment uses 100 trials, 200 samples per dataset, five numerical features, a location shift of `0.8`, random seed `42`, and significance level `0.05`.

Observed baseline results:

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

This evaluation compares detector output against known synthetic labels and reports the confusion matrix and classification metrics.

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

### Drift evaluation matrix

Run the full matrix:

```powershell
python scripts/evaluate_drift_matrix.py
```

The evaluator measures detection performance across five location-shift magnitudes, four sample sizes, and three random seeds.

| Parameter | Values |
|---|---|
| Location shift | `0.0`, `0.2`, `0.5`, `0.8`, `1.2` |
| Samples per dataset | `50`, `100`, `200`, `500` |
| Random seeds | `42`, `123`, `2026` |
| Trials per configuration | `100` |
| Numerical features | `5` |
| Significance level | `0.05` |
| Configurations | `60` |
| Total trials | `6,000` |

The zero-shift configuration is the negative control. For positive shifts, `feature_0` is the target feature and the other features are unchanged. Metrics are calculated against these synthetic labels.

**Observed findings from the current run:**

- **Detection sensitivity increases with sample size and shift magnitude.** For shift `0.5`, recall ranged from `0.16–0.26` at 50 samples, `0.56–0.63` at 100 samples, `0.92–0.98` at 200 samples, and `1.00` at 500 samples across the three seeds.
- **Small shifts are harder to detect.** For shift `0.2`, recall ranged from `0.01–0.03` at 50 samples to `0.49–0.60` at 500 samples.
- **Larger shifts were consistently detected in this run.** For shift `0.8`, recall was `0.70–0.73` at 50 samples and at least `0.99` at 100 samples.
- **Observed false-positive rates were low in these trials.** Across zero-shift controls, the false-positive rate ranged from `0.000–0.004` at the individual-seed configuration level. Positive-shift configurations had zero observed false positives in this run.

Precision is `null` when there are no positive predictions. In a zero-shift control, precision is `0.0` if false positives occur and `null` otherwise. Recall and F1 are `null` for the zero-shift control because there are no positive ground-truth examples. False-positive rate remains meaningful for this control.

**Limitations:** These results describe seeded synthetic normal distributions with a location shift. They do not establish performance on non-normal distributions, changes in variance, real production data, or downstream model degradation. The three seeds provide a limited view of variability; observed rates are not guarantees.

## Statistical and operational limitations

- A statistically significant distribution change does not necessarily imply degraded model performance.
- Detection power depends on effect size, sample size, distribution assumptions, and significance thresholds.
- Evaluating multiple features introduces multiple-testing considerations.
- Synthetic evaluation cannot substitute for validation against representative production data.
- Incident recommendations identify investigation steps; they do not establish root cause.
- The service does not automatically retrain, modify, or deploy models.

Before production use, calibrate thresholds against representative data, measure latency and resource use, establish alerting policies, and validate operational failure handling.

## Development principles

- Keep detector logic independent of transport and API concerns.
- Make evaluation runs reproducible through explicit seeds and documented parameters.
- Test negative controls, failure paths, and invalid inputs.
- Avoid logging raw records or unnecessary sensitive values.
- Treat benchmarks as measured evidence, not universal guarantees.
- Prefer explicit contracts and reviewable changes over implicit behavior.

## Roadmap

- Expand evaluation to variance shifts and non-normal distributions.
- Quantify runtime and memory behavior as sample size and feature count increase.
- Evaluate multiple-testing corrections.
- Improve operational observability and incident lifecycle handling.
- Validate detection thresholds against representative datasets.

## License

No license has been specified yet. Check the repository for an applicable license before reusing or redistributing the project.
