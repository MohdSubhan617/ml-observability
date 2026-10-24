# ML Observability

A production-oriented ML monitoring service for dataset validation, numerical feature drift detection, and evidence-based incident triage.

## Features

- **Schema validation:** Detect missing and unexpected columns.
- **Data-quality checks:** Count null values by column.
- **Numerical drift detection:** Compare reference and current distributions using the two-sample Kolmogorov–Smirnov (KS) test.
- **Multiple-testing correction:** Apply Benjamini–Hochberg false discovery rate (FDR) correction across evaluated features.
- **Explicit evaluation status:** Mark features as not evaluated when sample counts are insufficient or non-finite values are present.
- **REST API:** Expose monitoring operations through FastAPI.
- **Automated quality checks:** Run formatting, linting, and tests through GitHub Actions.
- **Reproducible evaluation:** Measure drift detection and false-positive rates on seeded synthetic data.

## Architecture

```text
Reference and current datasets
            |
            v
      FastAPI service
        /       \
       v         v
 Schema checks   KS drift tests
       |         |
       v         v
 Data-quality    BH-adjusted
 findings        drift results
        \       /
         v     v
       JSON responses
```

## Requirements

- Python 3.13
- pip
- Git

## Setup

Clone the repository and enter its directory:

```powershell
git clone https://github.com/MohdSubhan617/ml-observability.git
cd ml-observability
```

Create and activate a virtual environment on Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project and development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

## Run the API

```powershell
python -m uvicorn ml_observability.app:app --reload
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Interactive API documentation:

http://127.0.0.1:8000/docs

## API examples

### Validate a dataset

```powershell
$body = @{
    reference = @(
        @{ age = 25; score = 0.4 }
        @{ age = 30; score = 0.8 }
    )
    current = @(
        @{ age = 26; score = 0.5 }
        @{ age = 31; score = 0.7 }
    )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/validate `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

### Detect numerical drift

```powershell
$reference = @(1..100 | ForEach-Object { @{ value = [double]$_ } })
$current = @(1..100 | ForEach-Object { @{ value = [double]($_ + 1000) } })

$body = @{
    reference = $reference
    current = $current
    min_samples = 20
    alpha = 0.05
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
    -Uri http://127.0.0.1:8000/drift `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

## Tests and code quality

Run the test suite:

```powershell
python -m pytest -q
```

Check formatting:

```powershell
python -m ruff format --check .
```

Run linting:

```powershell
python -m ruff check .
```

## Synthetic evaluation

Run the deterministic drift evaluation:

```powershell
python scripts/evaluate_drift.py
```

The current seeded configuration uses 100 trials, 200 samples per dataset, five numerical features, and a location shift of 0.8 for the target feature.

Observed result from this configuration:

| Metric | Result |
|---|---:|
| Shifted-feature detection rate | 100% |
| Unchanged-feature false-positive rate | 1% |
| Trials | 100 |
| Unchanged-feature tests | 900 |

These are results from one synthetic Gaussian scenario, not a guarantee of production performance. Broader evaluation should cover multiple distributions, sample sizes, shift magnitudes, missingness patterns, and repeated random seeds.

## Statistical and operational limitations

- The KS test detects differences between empirical distributions; it does not establish that a shift harms model performance.
- The detector currently evaluates shared numerical features only.
- Benjamini–Hochberg correction controls the false discovery rate under its applicable assumptions; it does not eliminate false positives.
- Schema and null checks do not yet cover every data-quality failure mode.
- Synthetic evaluation does not substitute for validation against representative production data.
- Incident triage recommendations and production integrations remain future extensions.

## Development status

This repository is an evolving portfolio project focused on reproducibility, testability, and production-oriented ML monitoring design.
