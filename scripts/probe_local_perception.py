"""Privacy-safe diagnostic probe for local VLM perception and subset ablation.

All raw model outputs stay in config.private_dir. Stdout prints ONLY numeric
and technical aggregates (counts, rule ID histograms, token/time/GPU stats).
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import av
from PIL import Image, ImageDraw

from forte.analysis.config import AnalysisConfig
from forte.analysis.pipeline import (
    VideoAnalyzer,
    active_rules,
    overview_chunks,
    validate_overview_payload,
)
from forte.analysis.prompts import chunk_subwindows, overview_prompt
from forte.analysis.video import overview_frames, sample_video


def build_synthetic_probe_video(path: Path, duration_sec: int = 12, fps: int = 2) -> Path:
    """Create a deterministic synthetic cashier-counter interaction clip."""
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = 640
    stream.height = 360
    stream.pix_fmt = "yuv420p"
    total_frames = duration_sec * fps
    for idx in range(total_frames):
        t = idx / fps
        img = Image.new("RGB", (640, 360), (215, 220, 225))
        draw = ImageDraw.Draw(img)
        # Counter / desk
        draw.rectangle([100, 210, 540, 330], fill=(120, 85, 55), outline=(60, 40, 25), width=3)
        # Partition glass
        draw.rectangle([315, 50, 325, 210], fill=(180, 210, 230))
        # Client silhouette on left (0..9s)
        if t < 9.0:
            draw.ellipse([160, 90, 215, 145], fill=(60, 80, 130))
            draw.rectangle([150, 145, 225, 240], fill=(50, 70, 120))
        # Cashier silhouette on right (leaves desk after 8.5s)
        cx = 420 if t < 8.5 else min(580, int(420 + (t - 8.5) * 55))
        draw.ellipse([cx, 95, cx + 55, 150], fill=(45, 110, 80))
        draw.rectangle([cx - 10, 150, cx + 65, 245], fill=(35, 95, 70))
        # Banknotes / bag bundle moving across counter (2..11s)
        if 1.5 <= t <= 11.0:
            bx = int(230 + min(1.0, max(0.0, (t - 1.5) / 3.0)) * 130)
            draw.rectangle([bx, 185, bx + 65, 218], fill=(40, 150, 60), outline=(20, 80, 30), width=2)
            draw.rectangle([bx + 10, 192, bx + 55, 210], fill=(230, 240, 210))
        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def run_synthetic_probe(config: AnalysisConfig, run_full_pipeline: bool = False) -> dict[str, Any]:
    config.private_dir.mkdir(parents=True, exist_ok=True)
    probe_tmp = config.private_dir / "probe_tmp"
    probe_tmp.mkdir(parents=True, exist_ok=True)
    rules = active_rules(config.rules_path)
    rules_by_id = {r["id"]: r for r in rules}
    analyzer = VideoAnalyzer(config)
    start = time.monotonic()
    deadline = start + config.timeout_sec

    try:
        video_path = build_synthetic_probe_video(probe_tmp / "synthetic_probe.mp4")
        if run_full_pipeline:
            result = analyzer.analyze(video_path)
            return {
                "mode": "synthetic_full",
                "verdict_is_violation": result.get("verdict") == "violation",
                "violations_count": len(result.get("violations", [])),
                "metrics": analyzer.last_metrics,
            }

        video = sample_video(video_path, fps=config.scan_fps)
        scan = overview_frames(video, min(16, config.overview_frames))
        chunks = overview_chunks(scan)
        windows_total = 0
        non_idle_windows = 0
        candidate_hist: dict[str, int] = {}
        input_tokens = 0
        output_tokens = 0
        gpu_peak_mb = 0.0
        private_records: list[dict[str, Any]] = []
        probe_log = config.private_dir / "synthetic_probe_private.json"

        try:
            for idx, chunk in enumerate(chunks):
                sub = chunk_subwindows(chunk, video.duration_sec)
                overview, stats, raw = analyzer.vlm.generate(
                    chunk,
                    overview_prompt(rules, video.duration_sec, len(chunk), windows=sub),
                    max_pixels=config.overview_pixels,
                    max_tokens=min(650, config.generation_tokens),
                    deadline=deadline,
                )
                record: dict[str, Any] = {
                    "chunk_index": idx,
                    "subwindows": sub,
                    "stats": stats,
                    "overview": overview,
                    "raw": raw,
                }
                private_records.append(record)
                input_tokens += int(stats.get("input_tokens", 0))
                output_tokens += int(stats.get("output_tokens", 0))
                gpu_peak_mb = max(gpu_peak_mb, float(stats.get("gpu_peak_mb", 0.0)))
                try:
                    chunk_explicit, chunk_win_candidates = validate_overview_payload(
                        overview,
                        sub,
                        chunk,
                        rules_by_id,
                        video.duration_sec,
                        chunk_index=idx,
                    )
                except Exception as exc:
                    record["validation_error_type"] = type(exc).__name__
                    record["validation_error"] = str(exc)
                    raise
                wins = overview.get("windows", [])
                windows_total += len(wins)
                for w in wins:
                    if any(c != "idle" for c in w.get("cues", [])):
                        non_idle_windows += 1
                for cand in chunk_win_candidates:
                    rid = cand["rule_id"]
                    candidate_hist[rid] = candidate_hist.get(rid, 0) + 1
                for cand in chunk_explicit:
                    rid = cand["rule_id"]
                    candidate_hist[rid] = candidate_hist.get(rid, 0) + 1
        finally:
            if private_records:
                probe_log.write_text(json.dumps(private_records, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "mode": "synthetic_perception",
            "chunks_run": len(chunks),
            "windows_returned": windows_total,
            "non_idle_windows": non_idle_windows,
            "candidates_retrieved": sum(candidate_hist.values()),
            "candidate_rule_counts": candidate_hist,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "gpu_peak_mb": round(gpu_peak_mb, 1),
            "seconds": round(time.monotonic() - start, 3),
        }
    finally:
        shutil.rmtree(probe_tmp, ignore_errors=True)


def run_diagnostic_videos(config: AnalysisConfig, video_paths: list[Path]) -> dict[str, Any]:
    """Unlabeled diagnostic stage counter for explicit video paths; makes no accuracy/recall claim."""
    config.private_dir.mkdir(parents=True, exist_ok=True)
    analyzer = VideoAnalyzer(config)
    analyzer.warmup()

    overview_stages_total = 0
    windows_total = 0
    non_idle_windows_total = 0
    candidates_total = 0
    videos_with_candidates = 0
    verification_stages_total = 0
    evidence_valid_total = 0
    rule_candidate_hist: dict[str, int] = {}
    rule_confirmed_hist: dict[str, int] = {}
    errors: dict[str, int] = {}
    timings: list[float] = []
    in_tokens: list[int] = []
    out_tokens: list[int] = []
    gpu_peak_max = 0.0

    for path in video_paths:
        before_reports = set(config.private_dir.glob("*.json"))
        try:
            analyzer.analyze(path)
        except Exception as exc:
            err_name = type(exc).__name__
            errors[err_name] = errors.get(err_name, 0) + 1

        metrics = analyzer.last_metrics
        if metrics:
            timings.append(float(metrics.get("seconds", 0.0)))
            in_tokens.append(int(metrics.get("input_tokens", 0)))
            out_tokens.append(int(metrics.get("output_tokens", 0)))
            gpu_peak_max = max(gpu_peak_max, float(metrics.get("gpu_peak_mb", 0.0)))

        new_reports = [p for p in config.private_dir.glob("*.json") if p not in before_reports and p.name != "fewshot_examples.json"]
        for rep_path in new_reports:
            try:
                rep = json.loads(rep_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            vid_cands = 0
            for stage in rep.get("stages", []):
                if stage.get("stage") == "overview":
                    overview_stages_total += 1
                    ov = stage.get("result", {})
                    wins = ov.get("windows", [])
                    if isinstance(wins, list):
                        windows_total += len(wins)
                        for w in wins:
                            cues = w.get("cues", []) if isinstance(w, dict) else []
                            if any(c != "idle" for c in cues if isinstance(c, str)):
                                non_idle_windows_total += 1
                elif stage.get("stage") == "verification":
                    verification_stages_total += 1
                    vid_cands += 1
                    candidates_total += 1
                    cand = stage.get("candidate", {})
                    rid = str(cand.get("rule_id", "unknown"))
                    rule_candidate_hist[rid] = rule_candidate_hist.get(rid, 0) + 1
                    if stage.get("evidence_valid_before_threshold") is True:
                        evidence_valid_total += 1
                        ver = stage.get("result", {})
                        if isinstance(ver.get("confidence"), (int, float)) and ver["confidence"] >= config.confidence_threshold:
                            rule_confirmed_hist[rid] = rule_confirmed_hist.get(rid, 0) + 1
            if vid_cands > 0:
                videos_with_candidates += 1

    return {
        "mode": "unlabeled_diagnostic_no_accuracy_claim",
        "videos_evaluated": len(video_paths),
        "overview_stages_total": overview_stages_total,
        "windows_total": windows_total,
        "non_idle_windows_total": non_idle_windows_total,
        "candidates_total": candidates_total,
        "videos_with_candidates": videos_with_candidates,
        "verification_stages_total": verification_stages_total,
        "evidence_valid_before_threshold_total": evidence_valid_total,
        "rule_candidate_histogram": rule_candidate_hist,
        "rule_confirmed_histogram": rule_confirmed_hist,
        "timing_sec": {
            "mean": round(statistics.mean(timings), 3) if timings else 0.0,
            "max": round(max(timings), 3) if timings else 0.0,
            "min": round(min(timings), 3) if timings else 0.0,
        },
        "tokens": {
            "input_mean": round(statistics.mean(in_tokens), 1) if in_tokens else 0.0,
            "output_mean": round(statistics.mean(out_tokens), 1) if out_tokens else 0.0,
            "output_max": max(out_tokens) if out_tokens else 0,
        },
        "gpu_peak_mb_max": round(gpu_peak_max, 1),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Privacy-safe local perception probe. For labeled train evaluation use scripts.evaluate --split train."
    )
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic perception probe (default when no --video is passed).")
    parser.add_argument("--full", action="store_true", help="Run full pipeline in synthetic mode.")
    parser.add_argument("--video", action="append", default=[], help="Explicit video path for unlabeled stage diagnostics.")
    args = parser.parse_args()

    config = AnalysisConfig.from_env()
    if args.video and not args.synthetic:
        summary = run_diagnostic_videos(config, [Path(v) for v in args.video])
        print(json.dumps(summary, ensure_ascii=False))
        return

    summary = run_synthetic_probe(config, run_full_pipeline=args.full)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
