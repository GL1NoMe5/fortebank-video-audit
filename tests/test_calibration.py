"""Synthetic calibration replay checks; no footage or private reports are read."""

from copy import deepcopy

import pytest

from scripts.calibrate import calibrate_reports


@pytest.fixture
def calibration_case(tmp_path):
    records = [
        {"id": "synthetic-positive-a", "path": "synthetic/positive-a.mp4", "label": 1, "annotated_sec": 4.0},
        {"id": "synthetic-positive-b", "path": "synthetic/positive-b.mp4", "label": 1, "annotated_sec": 7.0},
        {"id": "synthetic-negative", "path": "synthetic/negative.mp4", "label": 0, "annotated_sec": None},
    ]
    return {"root": tmp_path, "records": records, "expected_profile": "a" * 64}


def report_for(case, index, *, score=None, evidence_valid=True, profile=None, error=None):
    record = case["records"][index]
    report = {
        "video_path": str((case["root"] / record["path"]).resolve()),
        "analysis_profile": deepcopy(profile if profile is not None else case["expected_profile"]),
        "stages": [{"stage": "overview", "result": {"view_usable": True, "timeline_complete": True, "candidates": []}}],
    }
    if error:
        report["error_type"] = error
        return report
    if score is not None:
        report["stages"].append({
            "stage": "verification",
            "acceptance_schema": 1,
            "evidence_valid_before_threshold": evidence_valid,
            "candidate": {"rule_id": "synthetic-rule", "timecode_sec": 4.0},
            "result": {
                "confirmed": True,
                "sequence_visible": True,
                "rule_id": "synthetic-rule",
                "confidence": score,
                "timecode_sec": 4.0,
                "description": "Synthetic observed sequence",
            },
        })
    positive = score is not None and evidence_valid is True
    report["result"] = {
        "verdict": "violation" if positive else "no_violation",
        "violations": [{"timecode_sec": 4.0, "description": "Synthetic event"}] if positive else [],
    }
    return report


def complete_reports(case):
    return [report_for(case, 0, score=0.8), report_for(case, 1, score=0.9), report_for(case, 2)]


def at_threshold(summary, threshold=0.65):
    return next(row for row in summary["grid"] if row["threshold"] == threshold)


def test_complete_single_profile_can_recommend(calibration_case):
    summary = calibrate_reports(reports=complete_reports(calibration_case), **calibration_case)
    assert summary["expected_videos"] == summary["videos"] == summary["completed"] == 3
    assert summary["failed"] == summary["missing"] == summary["invalid_reports"] == 0
    assert summary["coverage_complete"] is True
    assert summary["profile_matches_current"] is True
    assert summary["recommended"] is not None
    assert summary["recall_one_reached"] is True
    row = at_threshold(summary)
    assert (row["tp"], row["fp"], row["tn"], row["fn"]) == (2, 0, 1, 0)


def test_latest_failure_replaces_older_success(calibration_case):
    reports = complete_reports(calibration_case) + [report_for(calibration_case, 0, error="TimeoutError")]
    summary = calibrate_reports(reports=reports, **calibration_case)
    assert summary["completed"] == 2
    assert summary["failed"] == 1
    assert summary["missing"] == 0
    assert at_threshold(summary)["recall"] == 0.5
    assert at_threshold(summary)["fn"] == 1
    assert summary["recommended"] is None
    assert summary["recall_one_reached"] is False


def test_missing_positive_remains_in_recall_denominator(calibration_case):
    reports = [report_for(calibration_case, 0, score=0.8), report_for(calibration_case, 2)]
    summary = calibrate_reports(reports=reports, **calibration_case)
    assert summary["expected_videos"] == 3
    assert summary["completed"] == 2
    assert summary["missing"] == 1
    assert summary["coverage_complete"] is False
    assert at_threshold(summary)["recall"] == 0.5
    assert at_threshold(summary)["fn"] == 1
    assert summary["recommended"] is None
    assert summary["recall_one_reached"] is False


@pytest.mark.parametrize("missing", [True, False])
def test_uncompleted_negative_is_not_a_true_negative(calibration_case, missing):
    reports = complete_reports(calibration_case)[:2]
    if not missing:
        reports.append(report_for(calibration_case, 2, error="ValueError"))
    summary = calibrate_reports(reports=reports, **calibration_case)
    assert at_threshold(summary)["tn"] == 0
    assert summary["recommended"] is None
    assert summary["recall_one_reached"] is False


@pytest.mark.parametrize("mutation", ["legacy", "unsupported_schema", "false_schema", "missing_evidence", "non_boolean_evidence"])
def test_unreplayable_verification_does_not_count_as_completed(calibration_case, mutation):
    reports = complete_reports(calibration_case)
    stage = reports[0]["stages"][-1]
    if mutation == "legacy":
        stage.pop("acceptance_schema")
        stage.pop("evidence_valid_before_threshold")
    elif mutation == "unsupported_schema":
        stage["acceptance_schema"] = 2
    elif mutation == "false_schema":
        stage["acceptance_schema"] = False
    elif mutation == "missing_evidence":
        stage.pop("evidence_valid_before_threshold")
    else:
        stage["evidence_valid_before_threshold"] = "true"
    summary = calibrate_reports(reports=reports, **calibration_case)
    assert summary["completed"] == 2
    assert summary["invalid_reports"] >= 1
    assert at_threshold(summary)["fn"] == 1
    assert summary["recommended"] is None


def test_false_acceptance_flag_overrides_raw_high_confidence(calibration_case):
    reports = complete_reports(calibration_case)
    reports[0] = report_for(calibration_case, 0, score=0.99, evidence_valid=False)
    summary = calibrate_reports(reports=reports, **calibration_case)
    assert summary["completed"] == 3
    assert summary["invalid_reports"] == 0
    for row in summary["grid"]:
        assert row["tp"] <= 1
        assert row["fn"] >= 1
    assert summary["recall_one_reached"] is False


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.01, 1.01, 10**400, True, "0.9"])
def test_invalid_accepted_confidence_makes_report_unreplayable(calibration_case, score):
    reports = complete_reports(calibration_case)
    reports[0]["stages"][-1]["result"]["confidence"] = score
    summary = calibrate_reports(reports=reports, **calibration_case)
    assert summary["completed"] == 2
    assert summary["invalid_reports"] >= 1
    assert at_threshold(summary)["fn"] == 1
    assert summary["recommended"] is None


@pytest.mark.parametrize("mutation", ["missing_overview", "duplicate_overview", "incomplete_timeline", "missing_timeline", "invalid_final_result"])
def test_incomplete_report_cannot_become_a_completed_negative(calibration_case, mutation):
    reports = complete_reports(calibration_case)
    negative = reports[2]
    if mutation == "missing_overview":
        negative["stages"] = []
    elif mutation == "duplicate_overview":
        negative["stages"].append(deepcopy(negative["stages"][0]))
    elif mutation == "incomplete_timeline":
        negative["stages"][0]["result"]["timeline_complete"] = False
    elif mutation == "missing_timeline":
        negative["stages"][0]["result"].pop("timeline_complete")
    else:
        negative["result"]["violations"] = [{"timecode_sec": 4.0}]
    summary = calibrate_reports(reports=reports, **calibration_case)
    assert summary["completed"] == 2
    assert summary["invalid_reports"] >= 1
    assert at_threshold(summary)["tn"] == 0
    assert summary["recommended"] is None


def test_profiles_cannot_be_combined_to_complete_coverage(calibration_case):
    old_profile = "b" * 64
    reports = [
        report_for(calibration_case, 0, score=0.8),
        report_for(calibration_case, 1, score=0.9, profile=old_profile),
        report_for(calibration_case, 2),
    ]
    summary = calibrate_reports(reports=reports, **calibration_case)
    assert summary["profile_matches_current"] is True
    assert summary["completed"] == 2
    assert summary["missing"] == 1
    assert at_threshold(summary)["recall"] == 0.5
    assert summary["recommended"] is None


def test_latest_train_report_from_old_profile_blocks_recommendation(calibration_case):
    reports = complete_reports(calibration_case)
    reports.append(report_for(calibration_case, 0, score=0.95, profile="b" * 64))
    summary = calibrate_reports(reports=reports, **calibration_case)
    assert summary["profile_matches_current"] is False
    assert summary["recommended"] is None
    assert summary["recall_one_reached"] is False


def test_holdout_paths_do_not_affect_profile_selection_or_metrics(calibration_case):
    reports = complete_reports(calibration_case)
    holdout = report_for(calibration_case, 0, score=0.99, profile="c" * 64)
    holdout["video_path"] = str(calibration_case["root"] / "synthetic" / "holdout.mp4")
    expected = calibrate_reports(reports=reports, **calibration_case)
    actual = calibrate_reports(reports=reports + [holdout], **calibration_case)
    for field in ("videos", "completed", "failed", "missing", "invalid_reports", "grid", "recommended", "profile_matches_current"):
        assert actual[field] == expected[field]


def test_latest_legacy_attempt_cannot_resurrect_previous_success(calibration_case):
    reports = complete_reports(calibration_case)
    latest = report_for(calibration_case, 0, score=0.99)
    latest["stages"][-1].pop("acceptance_schema")
    reports.append(latest)
    summary = calibrate_reports(reports=reports, **calibration_case)
    assert summary["completed"] == 2
    assert at_threshold(summary)["recall"] == 0.5
    assert summary["recommended"] is None
