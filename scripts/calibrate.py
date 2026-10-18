"""Replay validated, train-only evidence within one current analysis profile.

This reports threshold estimates; it never changes production settings. Missing,
failed or legacy attempts cannot become clean predictions or disappear from recall.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate import splits

THRESHOLDS = (.3, .4, .5, .55, .6, .62, .65, .7, .75, .8, .85, .9)


def _path_key(value: str, root: Path) -> str:
    path = Path(value)
    return str((path if path.is_absolute() else root / path).resolve()).casefold()


def _validated_scores(report: dict) -> list[float]:
    """Use the producer's threshold-independent acceptance gate, never raw votes."""
    result, stages = report.get("result"), report.get("stages")
    if not isinstance(result, dict) or not isinstance(stages, list) or any(not isinstance(s, dict) for s in stages):
        raise ValueError("Incomplete analysis report")
    verdict, violations = result.get("verdict"), result.get("violations")
    if verdict not in ("violation", "no_violation") or not isinstance(violations, list):
        raise ValueError("Invalid completed result")
    if (verdict == "violation") != bool(violations):
        raise ValueError("Inconsistent completed result")
    overview = [stage.get("result") for stage in stages if stage.get("stage") == "overview"]
    if len(overview) != 1 or not isinstance(overview[0], dict):
        raise ValueError("Missing completed overview")
    if overview[0].get("view_usable") is not True or overview[0].get("timeline_complete") is not True or not isinstance(overview[0].get("candidates"), list):
        raise ValueError("Incomplete overview")
    scores = []
    for stage in stages:
        if stage.get("stage") != "verification":
            continue
        if type(stage.get("acceptance_schema")) is not int or stage["acceptance_schema"] != 1 or not isinstance(stage.get("evidence_valid_before_threshold"), bool):
            raise ValueError("Missing current evidence acceptance metadata")
        if stage["evidence_valid_before_threshold"] is False:
            continue
        evidence = stage.get("result")
        score = evidence.get("confidence") if isinstance(evidence, dict) else None
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1 or not math.isfinite(score):
            raise ValueError("Invalid accepted evidence confidence")
        scores.append(float(score))
    return scores


def calibrate_reports(records: list[dict], reports: list[dict], expected_profile: str,
                      *, root: Path = ROOT, unreadable_reports: int = 0) -> dict:
    """Reports are ordered oldest to newest; input records are TRAIN only."""
    if not isinstance(expected_profile, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_profile):
        raise ValueError("Current analysis profile must be a SHA-256 fingerprint")
    by_path = {_path_key(record["path"], root): record for record in records}
    if len(by_path) != len(records) or len({record["id"] for record in records}) != len(records):
        raise ValueError("Training records contain duplicate paths or IDs")
    if any(type(record.get("label")) is not int or record["label"] not in (0, 1) for record in records):
        raise ValueError("Training labels must be binary integers")
    eligible = []
    for report in reports:
        if not isinstance(report, dict) or not isinstance(report.get("video_path"), str) or not report["video_path"]:
            continue
        key = _path_key(report["video_path"], root)
        if key in by_path:
            eligible.append((key, report))
    latest_profile = eligible[-1][1].get("analysis_profile") if eligible else None
    profile_matches = latest_profile == expected_profile
    # Never backfill a newer foreign/legacy cohort with older current results.
    latest = {}
    if profile_matches:
        for key, report in eligible:
            if report.get("analysis_profile") == expected_profile:
                latest[key] = report
    attempts = {}
    invalid_reports = int(unreadable_reports)
    for key, report in latest.items():
        if report.get("error_type"):
            attempts[key] = {"failed": True, "scores": []}
            continue
        try:
            attempts[key] = {"failed": False, "scores": _validated_scores(report)}
        except ValueError:
            invalid_reports += 1
            attempts[key] = {"failed": True, "scores": []}
    failed = sum(attempt["failed"] for attempt in attempts.values())
    missing = len(records) - len(attempts)
    completed = len(attempts) - failed
    coverage_complete = bool(records) and missing == 0
    grid = []
    for threshold in THRESHOLDS:
        tp = fp = tn = fn = 0
        for key, record in by_path.items():
            attempt = attempts.get(key)
            success = attempt is not None and not attempt["failed"]
            positive = bool(success and any(score >= threshold for score in attempt["scores"]))
            tp += int(record["label"] == 1 and positive)
            fp += int(record["label"] == 0 and positive)
            tn += int(record["label"] == 0 and success and not positive)
            fn += int(record["label"] == 1 and not positive)
        recall = tp / (tp + fn) if tp + fn else None
        precision = tp / (tp + fp) if tp + fp else None
        r, p = recall or 0.0, precision or 0.0
        grid.append({"threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                     "recall": recall, "precision": precision,
                     "accuracy": (tp + tn) / len(records) if records else None,
                     "f2": 5 * p * r / (4 * p + r) if p + r else 0.0})
    ready = profile_matches and coverage_complete and failed == 0 and invalid_reports == 0
    best = max(grid, key=lambda row: (row["recall"] or 0, row["precision"] or 0, row["threshold"])) if ready else None
    reasons = []
    if not profile_matches:
        reasons.append("latest_analysis_profile_is_not_current")
    if missing:
        reasons.append("training_coverage_incomplete")
    if failed:
        reasons.append("latest_attempts_failed_or_invalid")
    if invalid_reports:
        reasons.append("unreadable_or_legacy_reports")
    return {"split": "train", "analysis_profile": expected_profile,
            "latest_profile": latest_profile if isinstance(latest_profile, str) and re.fullmatch(r"[0-9a-f]{64}", latest_profile) else None,
            "profile_matches_current": profile_matches,
            "videos": len(attempts), "expected_videos": len(records), "completed": completed,
            "failed": failed, "missing": missing, "invalid_reports": invalid_reports,
            "coverage_complete": coverage_complete,
            "failed_positive": sum(record["label"] == 1 and key in attempts and attempts[key]["failed"] for key, record in by_path.items()),
            "missing_positive": sum(record["label"] == 1 and key not in attempts for key, record in by_path.items()),
            "grid": grid, "recommended": best, "recommendation_blocked_reasons": reasons,
            "recall_one_reached": bool(best and best["recall"] == 1),
            "note": "Train-only replay estimates. Recommendation requires complete current-profile coverage without failures; a chosen threshold still requires fresh full-pipeline validation. No setting is changed."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, default=ROOT / "runtime/private")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/dataset_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/calibration.json")
    args = parser.parse_args()
    from forte.analysis.config import AnalysisConfig
    from forte.analysis.pipeline import analysis_profile

    records = splits(json.loads(args.manifest.read_text(encoding="utf-8-sig"))["records"])["train"]
    profile = analysis_profile(AnalysisConfig.from_env())
    reports, unreadable = [], 0
    for path in sorted(args.reports.glob("*.json"), key=lambda item: (item.stat().st_mtime_ns, item.name)):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(report, dict):
                reports.append(report)
            else:
                unreadable += 1
        except (OSError, ValueError):
            unreadable += 1
    summary = calibrate_reports(records, reports, profile, unreadable_reports=unreadable)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
