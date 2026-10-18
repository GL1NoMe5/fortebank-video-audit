"""Synthetic timestamp/image binding probe using the production LocalVLM API.

CPU-only contract check (never loads Torch or model weights):
    python scripts/validate_multiframe.py --self-test
Run ONE model at a time, when the GPU is free:
    python scripts/validate_multiframe.py --model-path models/Qwen3-VL-4B-Instruct
    python scripts/validate_multiframe.py --model-path models/Qwen3-VL-8B-Instruct

Every image and response in this probe is synthetic and may be inspected publicly.
No dataset, normative catalog, private report or real video is read. A model pass
is recorded only after comparison with generated ground truth. Core deadlines are
cooperative; an unresponsive GPU kernel still needs an external process watchdog.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLORS = {"red": (225, 35, 40), "blue": (35, 80, 225),
          "yellow": (245, 190, 20), "green": (20, 175, 65)}
SHAPES = {"circle", "square", "triangle", "star"}
POSITIONS = {"left", "right", "center"}
CASE_NAMES = ("sequence", "permuted", "needle32")


@dataclass
class ProbeCase:
    name: str
    frames: list
    expected: dict


def _object(color, shape, position):
    return {"color": color, "shape": shape, "position": position}


def _image(objects):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (464, 848), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    for item in objects:
        cx = {"left": 116, "right": 348, "center": 232}[item["position"]]
        cy, radius = 424, 75
        box = (cx - radius, cy - radius, cx + radius, cy + radius)
        color, shape = COLORS[item["color"]], item["shape"]
        if shape == "circle":
            draw.ellipse(box, fill=color)
        elif shape == "square":
            draw.rectangle(box, fill=color)
        elif shape == "triangle":
            draw.polygon([(cx, cy-radius), (cx+radius, cy+radius), (cx-radius, cy+radius)], fill=color)
        elif shape == "star":
            points = []
            for index in range(10):
                angle = -math.pi / 2 + index * math.pi / 5
                r = radius if index % 2 == 0 else radius * .42
                points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
            draw.polygon(points, fill=color)
        else:
            raise ValueError("Unknown synthetic shape")
    return image


def build_cases() -> list[ProbeCase]:
    from forte.analysis.video import VideoFrame

    sequence = [
        (0., [_object("red", "circle", "left"), _object("blue", "square", "right")]),
        (5., [_object("red", "circle", "right"), _object("blue", "square", "left")]),
        (10., [_object("yellow", "triangle", "left"), _object("blue", "square", "right")]),
        (15., [_object("yellow", "triangle", "right"), _object("red", "circle", "left")]),
    ]
    frames = [VideoFrame(timestamp, _image(objects)) for timestamp, objects in sequence]
    expected = {"frames": [{"timestamp_sec": timestamp, "objects": deepcopy(objects)} for timestamp, objects in sequence]}
    needle = [(float(index), [_object("green", "star", "center") if index == 17
                             else _object("red", "circle", "center")]) for index in range(32)]
    return [ProbeCase("sequence", frames, expected),
            ProbeCase("permuted", [frames[index] for index in (2, 0, 3, 1)], deepcopy(expected)),
            ProbeCase("needle32", [VideoFrame(timestamp, _image(objects)) for timestamp, objects in needle],
                      {"frames": [{"timestamp_sec": timestamp, "objects": objects} for timestamp, objects in needle]})]


def prompt_for(case: ProbeCase) -> str:
    # No expected object identity, position pattern or needle timestamp is given.
    return f"""Inspect EVERY one of the {len(case.frames)} supplied images independently.
Each image contains one or two colored geometric shapes on a plain background.
The text attached to each image gives its actual timestamp in seconds. Image order
may differ from timestamp order: preserve the timestamp attached to that image,
never use its presentation index as the time. Do not assume frames are identical.
List ALL visible colored shapes for EACH supplied timestamp exactly once.
Use only these lowercase labels:
color: red, blue, yellow, green
shape: circle, square, triangle, star
position: left, right, center (horizontal location in the full image)
Return only this JSON schema, without markdown or extra keys:
{{"frames":[{{"timestamp_sec":123.5,"objects":[{{"color":"blue","shape":"star","position":"right"}}]}}]}}
The schema example demonstrates structure only; it is not an answer. Entries may
be in any order. Include all {len(case.frames)} timestamps even if shapes repeat.
"""


def _timestamp(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    try:
        return float(value) if math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def _objects(value):
    if not isinstance(value, list) or not value:
        return None
    result = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"color", "shape", "position"}:
            return None
        if not all(isinstance(item[key], str) for key in item):
            return None
        color, shape, position = (item[key].strip().lower() for key in ("color", "shape", "position"))
        if color not in COLORS or shape not in SHAPES or position not in POSITIONS:
            return None
        result.append((color, shape, position))
    return tuple(sorted(result))


def score_response(case: ProbeCase, response) -> dict:
    """Exact per-time object MULTISET matching; no nearest-time/fuzzy matching."""
    gold = {float(row["timestamp_sec"]): _objects(row["objects"]) for row in case.expected["frames"]}
    rows = response.get("frames") if isinstance(response, dict) else None
    schema_valid = isinstance(response, dict) and set(response) == {"frames"} and isinstance(rows, list)
    parsed, counts = {}, Counter()
    invalid_rows = 0
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or set(row) != {"timestamp_sec", "objects"}:
            invalid_rows += 1
            continue
        timestamp, objects = _timestamp(row["timestamp_sec"]), _objects(row["objects"])
        if timestamp is None or objects is None:
            invalid_rows += 1
            continue
        counts[timestamp] += 1
        parsed[timestamp] = objects
    schema_valid = bool(schema_valid and invalid_rows == 0)
    duplicates = sum(count - 1 for count in counts.values())
    correct = sum(counts[timestamp] == 1 and parsed.get(timestamp) == objects for timestamp, objects in gold.items())
    coverage = sum(counts[timestamp] == 1 for timestamp in gold)
    exact_times = set(parsed) == set(gold) and duplicates == 0
    inventory_expected = Counter(item for objects in gold.values() for item in objects)
    inventory_observed = Counter(item for timestamp, objects in parsed.items() if counts[timestamp] == 1 for item in objects)
    metrics = {
        "schema_valid": schema_valid,
        "expected_frames": len(gold), "returned_frames": len(rows) if isinstance(rows, list) else 0,
        "correct_frames": correct, "temporal_binding_accuracy": correct / len(gold),
        "exact_temporal_binding": bool(schema_valid and exact_times and correct == len(gold)),
        "timestamp_coverage": coverage / len(gold), "exact_timestamp_set": exact_times,
        "duplicate_timestamp_count": duplicates, "invalid_row_count": invalid_rows,
        "unexpected_timestamp_count": len(set(parsed) - set(gold)),
        "inventory_exact_ignoring_time": bool(schema_valid and duplicates == 0 and inventory_expected == inventory_observed),
    }
    if case.name == "needle32":
        observed = sorted(timestamp for timestamp, objects in parsed.items()
                          if counts[timestamp] == 1 and ("green", "star", "center") in objects)
        metrics.update(needle_expected_timestamp_sec=17., needle_observed_timestamps_sec=observed,
                       needle_timestamp_exact=observed == [17.])
    return metrics


def self_test() -> dict:
    """Test generator/scorer on synthetic oracle/corrupt answers, not the model."""
    cases = build_cases()
    for case in cases:
        assert score_response(case, case.expected)["exact_temporal_binding"]
        reverse = {"frames": list(reversed(deepcopy(case.expected["frames"])))}
        assert score_response(case, reverse)["exact_temporal_binding"]
        assert not score_response(case, {"frames": []})["exact_temporal_binding"]
    original = {frame.timestamp_sec: hashlib.sha256(frame.image.tobytes()).hexdigest() for frame in cases[0].frames}
    permuted = {frame.timestamp_sec: hashlib.sha256(frame.image.tobytes()).hexdigest() for frame in cases[1].frames}
    assert original == permuted
    assert [frame.timestamp_sec for frame in cases[1].frames] == [10., 0., 15., 5.]
    wrong = deepcopy(cases[0].expected)
    wrong["frames"][0]["objects"], wrong["frames"][1]["objects"] = wrong["frames"][1]["objects"], wrong["frames"][0]["objects"]
    scored = score_response(cases[0], wrong)
    assert scored["temporal_binding_accuracy"] == .5 and scored["inventory_exact_ignoring_time"]
    duplicate = deepcopy(cases[0].expected)
    duplicate["frames"].append(deepcopy(duplicate["frames"][0]))
    assert not score_response(cases[0], duplicate)["exact_temporal_binding"]
    wrong = deepcopy(cases[2].expected)
    wrong["frames"][17]["objects"], wrong["frames"][18]["objects"] = wrong["frames"][18]["objects"], wrong["frames"][17]["objects"]
    scored = score_response(cases[2], wrong)
    assert not scored["needle_timestamp_exact"] and scored["needle_observed_timestamps_sec"] == [18.]
    assert _timestamp(float("nan")) is None and _timestamp(10**400) is None
    return {"cpu_self_test_passed": True, "model_inference_run": False,
            "cases": list(CASE_NAMES), "synthetic_frames": 36}


def _rss_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1024**2, 2)
    except (ImportError, OSError):
        return None


def _gpu_memory(torch):
    try:
        return {"gpu_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
                "gpu_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
                "gpu_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2)}
    except Exception:
        return {"gpu_memory_available": False}


def run(args) -> tuple[dict, Path]:
    from forte.analysis.config import AnalysisConfig
    from forte.analysis.vlm import LocalVLM

    config = AnalysisConfig.from_env()
    overrides = {"grounding_enabled": False}
    if args.model_path:
        overrides["model_path"] = args.model_path.resolve()
    if args.quantization:
        overrides["quantization"] = args.quantization
    config = replace(config, **overrides)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "runtime" / f"synthetic_multiframe_{config.model_path.name}_{stamp}.json"
    if output.exists():
        raise ValueError("Output already exists; select a new path.")
    pixels = args.max_pixels or config.overview_pixels
    cases = [case for case in build_cases() if case.name in args.cases]
    report = {"schema_version": 1, "synthetic_only": True, "model_load_attempted": True, "model_inference_run": False,
              "model": config.model_path.name, "model_path": str(config.model_path),
              "quantization": config.quantization, "created_at": datetime.now(timezone.utc).isoformat(),
              "max_pixels": pixels, "max_tokens": args.max_tokens, "timeout_sec_per_case": args.timeout_sec,
              "deadline_policy": "cooperative core deadline; no external kernel watchdog",
              "core_vlm_sha256": hashlib.sha256((ROOT / "forte/analysis/vlm.py").read_bytes()).hexdigest(),
              "cases": [], "all_exact_temporal_binding": False}
    output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)

    runner = LocalVLM(config)
    started = time.monotonic()
    report["cpu_rss_before_load_mb"] = _rss_mb()
    print(json.dumps({"event": "model_load_start", "model": config.model_path.name}), flush=True)
    try:
        runner.load()
    except Exception as exc:
        report.update(load_error_type=type(exc).__name__, load_wall_seconds=round(time.monotonic()-started, 3))
        persist()
        return report, output
    import torch
    report.update(load_wall_seconds=round(time.monotonic()-started, 3),
                  model_load_seconds=runner.load_seconds, cpu_rss_after_load_mb=_rss_mb(),
                  gpu_after_load=_gpu_memory(torch))
    persist()
    print(json.dumps({"event": "model_ready", "model": config.model_path.name,
                      "load_wall_seconds": report["load_wall_seconds"]}), flush=True)
    for case in cases:
        row = {"name": case.name, "input_timestamps_sec": [frame.timestamp_sec for frame in case.frames],
               "expected": case.expected, "cpu_rss_before_mb": _rss_mb()}
        report["active_case"] = case.name
        persist()
        started = time.monotonic()
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            report["model_inference_run"] = True
            response, stats, raw = runner.generate(case.frames, prompt_for(case), max_pixels=pixels,
                max_tokens=args.max_tokens, deadline=time.monotonic()+args.timeout_sec)
            row.update(response=response, raw_response=raw, core_stats=stats, metrics=score_response(case, response))
        except Exception as exc:
            row.update(error_type=type(exc).__name__, metrics=score_response(case, None))
        row.update(wall_seconds=round(time.monotonic()-started, 3), cpu_rss_after_mb=_rss_mb(), **_gpu_memory(torch))
        report["cases"].append(row)
        report["active_case"] = None
        persist()
        print(json.dumps({"event": "case_complete", "case": case.name,
                          "wall_seconds": row["wall_seconds"], "error_type": row.get("error_type"),
                          "metrics": row["metrics"], "gpu_peak_allocated_mb": row.get("gpu_peak_allocated_mb")}), flush=True)
    by_name = {row["name"]: row for row in report["cases"]}
    if "sequence" in by_name and "permuted" in by_name:
        pair = [by_name[name] for name in ("sequence", "permuted")]
        complete = all(row["metrics"]["schema_valid"] and row["metrics"]["exact_timestamp_set"] for row in pair)
        maps = [{float(item["timestamp_sec"]): _objects(item["objects"]) for item in row.get("response", {}).get("frames", [])}
                for row in pair] if complete else []
        report["paired_permutation_consistent"] = maps[0] == maps[1] if complete else None
    report["all_exact_temporal_binding"] = bool(report["cases"]) and all(
        not row.get("error_type") and row["metrics"]["exact_temporal_binding"] for row in report["cases"])
    persist()
    return report, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true", help="CPU-only generator/scoring checks; no model/GPU access.")
    parser.add_argument("--model-path", type=Path, help="Local 4B or 8B checkpoint directory; defaults to FORTE_MODEL_PATH/core config.")
    parser.add_argument("--quantization", choices=("nf4", "none"))
    parser.add_argument("--cases", nargs="+", choices=CASE_NAMES, default=list(CASE_NAMES))
    parser.add_argument("--max-pixels", type=int, help="Defaults to current production overview pixel budget.")
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--timeout-sec", type=float, default=180.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test()))
        return 0
    if not math.isfinite(args.timeout_sec) or not 1 <= args.timeout_sec <= 600:
        parser.error("--timeout-sec must be between 1 and 600")
    if not 128 <= args.max_tokens <= 8192 or (args.max_pixels is not None and not 4096 <= args.max_pixels <= 1_000_000):
        parser.error("Invalid token or pixel budget")
    report, output = run(args)
    print(json.dumps({"event": "probe_complete", "model": report["model"], "output": str(output),
                      "all_exact_temporal_binding": report["all_exact_temporal_binding"],
                      "load_error_type": report.get("load_error_type")}))
    return 0 if report["all_exact_temporal_binding"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
