"""Evaluate locally; stdout is limited to aggregate metrics, never video content."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
import random
import statistics
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _canonical(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _digest(value) -> str:
    encoded = json.dumps(_canonical(value), sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_fingerprint(path: Path) -> dict:
    # Hash actual local artifacts once before warmup, not just the model's name
    # or mutable download metadata. Hidden downloader caches are not model inputs.
    files = [item for item in path.rglob("*") if item.is_file()
             and not any(part.startswith(".") for part in item.relative_to(path).parts)]
    if not (path / "config.json").is_file():
        raise ValueError("Model artifacts are incomplete; config.json is missing.")
    entries = [{"path": item.relative_to(path).as_posix(), "bytes": item.stat().st_size,
                "sha256": _file_digest(item)} for item in sorted(files)]
    return {"path": str(path.resolve()), "files": entries, "sha256": _digest(entries)}


def build_run_provenance(*, config, manifest: dict, split: str,
                         selected: list[dict], root: Path = ROOT) -> dict:
    """Bind a run to inputs and implementation without inspecting video pixels."""
    code_paths = sorted((root / "forte" / "analysis").glob("*.py"))
    code_paths.append(root / "scripts" / "evaluate.py")
    code = {path.relative_to(root).as_posix(): _file_digest(path) for path in code_paths}
    examples = config.private_dir / "fewshot_examples.json"
    packages = {}
    for name in ("torch", "transformers", "bitsandbytes", "av", "numpy", "Pillow",
                 "opencv-python", "opencv-python-headless", "accelerate", "safetensors"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    identity = {
        "split": split,
        "config": _canonical(asdict(config)),
        "model": _model_fingerprint(config.model_path),
        "grounding_model": _model_fingerprint(config.grounding_model_path)
            if config.grounding_enabled and (config.grounding_model_path / "config.json").is_file() else None,
        "rules_sha256": _file_digest(config.rules_path),
        "code": code,
        "manifest_sha256": _digest(manifest),
        "examples_sha256": _file_digest(examples) if examples.is_file() else None,
        "environment": {"python": sys.version.split()[0], "packages": packages},
    }
    selected_records = [{"id": row["id"], "label": row["label"],
                         "annotated_sec": row.get("annotated_sec"),
                         "record_sha256": _digest(row)} for row in selected]
    selected_ids = [row["id"] for row in selected_records]
    if any(not isinstance(item, str) or not item for item in selected_ids):
        raise ValueError("Selected records must have nonempty string IDs.")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Selected records contain duplicate IDs.")
    return {"schema_version": 1, "identity": identity, "identity_sha256": _digest(identity),
            "selected_ids": selected_ids, "selected_records": selected_records}


def validate_resume(payload: dict, current: dict) -> list[dict]:
    """Reject mixed runs. A larger limit may only extend the same ordered split."""
    if not isinstance(payload, dict) or not isinstance(current, dict):
        raise ValueError("Evaluation output and provenance must be JSON objects.")
    previous = payload.get("provenance")
    if not isinstance(previous, dict) or previous.get("schema_version") != 1:
        raise ValueError("Existing output lacks versioned provenance; use a new output file.")
    for provenance in (previous, current):
        identity = provenance.get("identity")
        if not isinstance(identity, dict) or provenance.get("identity_sha256") != _digest(identity):
            raise ValueError("Evaluation provenance fingerprint is invalid.")
    if previous["identity_sha256"] != current["identity_sha256"]:
        raise ValueError("Run identity changed (split, model, config, rules, code, manifest or examples); use a new output file.")
    if payload.get("split") != current["identity"]["split"]:
        raise ValueError("Stored split does not match run provenance.")
    if payload.get("model") != Path(current["identity"]["config"]["model_path"]).name:
        raise ValueError("Stored model does not match run provenance.")
    old_ids, new_ids = previous.get("selected_ids"), current.get("selected_ids")
    old_records, new_records = previous.get("selected_records"), current.get("selected_records")
    if not isinstance(old_ids, list) or not isinstance(old_records, list) or \
            any(not isinstance(item, str) or not item for item in old_ids) or \
            any(not isinstance(row, dict) for row in old_records):
        raise ValueError("Existing output has invalid selected-record provenance.")
    if len(set(old_ids)) != len(old_ids) or old_ids != [row.get("id") for row in old_records]:
        raise ValueError("Existing output has duplicate or inconsistent selected IDs.")
    if old_ids != new_ids[:len(old_ids)] or old_records != new_records[:len(old_records)]:
        raise ValueError("Resume may only preserve or increase the limit within the same ordered split.")
    rows = payload.get("records")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Existing evaluation records are invalid.")
    ids = [row.get("id") for row in rows]
    if ids != old_ids[:len(rows)] or len(rows) > len(old_ids):
        raise ValueError("Existing results are not a unique prefix of the selected records.")
    for row, expected in zip(rows, old_records):
        if row.get("label") != expected["label"] or row.get("annotated_sec") != expected["annotated_sec"]:
            raise ValueError("Existing result labels or annotations do not match selected records.")
        if row.get("predicted") not in (None, 0, 1):
            raise ValueError("Existing result contains an invalid prediction.")
        if not row.get("error_type") and row.get("predicted") is None:
            raise ValueError("Completed result is missing a prediction.")
        seconds = row.get("seconds")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Existing result has an invalid elapsed time.")
    return rows


def _save_payload(path: Path, payload: dict) -> None:
    # Atomic replacement prevents an interruption from destroying resumability.
    # Revalidate immediately before replacement so another run is never relabelled.
    validate_resume(payload, payload["provenance"])
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        prior_rows = validate_resume(prior, payload["provenance"])
        if prior_rows != payload["records"][:len(prior_rows)]:
            raise ValueError("Stored progress cannot be removed or replaced; use a new output file.")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def splits(records: list[dict]) -> dict[str, list[dict]]:
    rng = random.Random(26)
    train, holdout = [], []
    for label in (0, 1):
        group = sorted([r for r in records if r["label"] == label], key=lambda r: r["id"])
        rng.shuffle(group)
        holdout.extend(group[:4])
        train.extend(group[4:])
    def interleave(rows):
        groups = [[r for r in rows if r["label"] == label] for label in (0, 1)]
        return [group[i] for i in range(max(map(len, groups))) for group in groups if i < len(group)]
    train, holdout = interleave(train), interleave(holdout)
    return {"train": train, "holdout": holdout, "all": train + holdout}


def summarize(rows: list[dict]) -> dict:
    good = [r for r in rows if not r.get("error_type")]
    tp = sum(r["label"] == 1 and r["predicted"] == 1 for r in good)
    fp = sum(r["label"] == 0 and r["predicted"] == 1 for r in good)
    tn = sum(r["label"] == 0 and r["predicted"] == 0 for r in good)
    # A failed positive video is a missed detection for end-to-end recall. Keep
    # its prediction null in the record rather than inventing a clean verdict.
    fn = sum(r["label"] == 1 and (bool(r.get("error_type")) or r.get("predicted") != 1) for r in rows)
    times = [r["seconds"] for r in rows]
    errors = [r["localization_error_sec"] for r in good if r.get("localization_error_sec") is not None]
    return {
        "videos": len(rows), "completed": len(good), "failed": len(rows) - len(good),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "accuracy": round((tp + tn) / len(rows), 4) if rows else None,
        "completed_accuracy": round((tp + tn) / len(good), 4) if good else None,
        "failed_positive": sum(r["label"] == 1 and bool(r.get("error_type")) for r in rows),
        "mean_seconds": round(statistics.mean(times), 3) if times else None,
        "max_seconds": round(max(times), 3) if times else None,
        "median_localization_error_sec": round(statistics.median(errors), 3) if errors else None,
        "localized_positive_count": len(errors),
        "localized_within_3_sec": sum(error <= 3 for error in errors),
        "over_300_seconds": sum(t >= 300 for t in times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "dataset_manifest.json")
    parser.add_argument("--split", choices=["train", "holdout", "all"], default="holdout")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime" / "evaluation.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.output.exists() and not args.resume:
        parser.error("Output already exists; use --resume for the identical run or choose a new --output.")
    from forte.analysis.config import AnalysisConfig
    from forte.analysis.pipeline import VideoAnalyzer

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    selected = splits(manifest["records"])[args.split]
    if args.limit is not None:
        selected = selected[:args.limit]
    config = AnalysisConfig.from_env()
    print(json.dumps({"event": "fingerprinting_run"}), flush=True)
    provenance = build_run_provenance(config=config, manifest=manifest, split=args.split, selected=selected)
    rows = []
    if args.resume and args.output.exists():
        try:
            rows = validate_resume(json.loads(args.output.read_text(encoding="utf-8")), provenance)
        except ValueError as exc:
            parser.error(str(exc))
    done = {r["id"] for r in rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"split": args.split, "model": config.model_path.name, "provenance": provenance,
               "records": rows, "summary": summarize(rows)}
    if not args.output.exists():
        # Exclusive creation also protects against two fresh runs choosing the
        # same path between the initial check and provenance calculation.
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    else:
        _save_payload(args.output, payload)
    if len(done) == len(selected):
        print(json.dumps({"event": "already_complete", "summary": summarize(rows)}), flush=True)
        return
    analyzer = VideoAnalyzer(config)
    analyzer.warmup()
    print(json.dumps({"event": "model_ready", "load_seconds": analyzer.vlm.load_seconds}), flush=True)
    for index, record in enumerate(selected):
        if record["id"] in done:
            continue
        source = Path(record["path"])
        if not source.is_absolute():
            source = ROOT / source
        start = time.monotonic()
        row = {"id": record["id"], "label": record["label"], "annotated_sec": record.get("annotated_sec"), "predicted": None}
        try:
            result = analyzer.analyze(source)
            predicted_times = [event["timecode_sec"] for event in result["violations"]]
            row["predicted"] = int(result["verdict"] == "violation")
            row["predicted_times"] = predicted_times
            if record.get("annotated_sec") is not None and predicted_times:
                row["localization_error_sec"] = min(abs(t - record["annotated_sec"]) for t in predicted_times)
        except Exception as exc:
            row["error_type"] = type(exc).__name__
        row["seconds"] = round(time.monotonic() - start, 3)
        row["metrics"] = analyzer.last_metrics
        rows.append(row)
        payload = {"split": args.split, "model": config.model_path.name, "provenance": provenance,
                   "records": rows, "summary": summarize(rows)}
        _save_payload(args.output, payload)
        print(json.dumps({"event": "progress", "completed": len(rows), "total": len(selected), "last_seconds": row["seconds"], "last_error_type": row.get("error_type"), "summary": payload["summary"]}), flush=True)


if __name__ == "__main__":
    main()

