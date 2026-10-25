"""Integration coverage for the synthetic monitoring-to-investigation demo."""

import json
from pathlib import Path

import pytest
from scripts.run_demo import main, render_report, run_demo


def test_demo_is_deterministic_and_traces_the_injected_findings() -> None:
    first = run_demo(seed=42)
    second = run_demo(seed=42)

    assert first == second
    assert first["schema_validation"]["unexpected_columns"] == ["campaign_source"]
    assert first["schema_validation"]["missing_values"] == {"balance": 12}
    drift = {item["feature"]: item for item in first["drift_results"]}
    assert drift["account_age_days"]["drift_detected"] is True
    assert drift["balance"]["drift_detected"] is False
    assert len(first["incidents"]) == 1
    investigation = first["investigation"]
    assert not investigation["abstained"]
    assert {item["category"] for item in investigation["hypotheses"]} == {
        "feature_distribution_shift",
        "schema_contract_violation",
        "missingness",
    }
    supplied_ids = {"demo-schema-001", "demo-drift-account_age_days", "demo-drift-balance"}
    referenced_ids = {
        evidence["evidence_id"]
        for hypothesis in investigation["hypotheses"]
        for evidence in (*hypothesis["supporting_evidence"], *hypothesis["contradictory_evidence"])
    }
    assert referenced_ids <= supplied_ids


def test_demo_report_explains_signals_and_synthetic_limitations() -> None:
    report = render_report(run_demo())

    assert "# Synthetic ML observability demo" in report
    assert "account_age_days" in report
    assert "campaign_source" in report
    assert "does not demonstrate production diagnostic accuracy" in report


def test_demo_cli_can_save_machine_readable_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "demo-report.json"
    monkeypatch.setattr("sys.argv", ["run_demo.py", "--json-output", str(output_path)])

    main()

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["seed"] == 42
    assert "# Synthetic ML observability demo" in capsys.readouterr().out
