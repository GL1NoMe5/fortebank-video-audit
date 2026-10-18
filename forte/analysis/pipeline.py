"""Two-stage local video review: chunked action perception and independent normative verification."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .config import AnalysisConfig
from .grounding import SceneGrounder, detail_crops
from .prompts import ACTION_CUES, chunk_subwindows, format_retry_prompt, overview_prompt, verification_prompt
from .video import VideoFrame, evidence_frames, overview_frames, sample_video, window_frames
from .vlm import LocalVLM


MAX_OVERVIEW_CHUNK_FRAMES = 9
OVERVIEW_CHUNK_OVERLAP = 1
TIMESTAMP_TOLERANCE_SEC = 5e-3
MAX_STAGE_ATTEMPTS = 2
NON_RETRYABLE_REASONS = frozenset({"unusable_view", "incomplete_timeline", "timeout", "operational_error"})

ERROR_REASON_PATTERNS: tuple[tuple[str, str], ...] = (
    ("incomplete JSON object", "incomplete_json"),
    ("did not return a complete JSON object", "no_json_object"),
    ("Video view is not usable", "unusable_view"),
    ("did not cover the complete timeline", "incomplete_timeline"),
    ("subwindow count mismatch", "subwindow_count_mismatch"),
    ("Window bounds do not match", "window_bounds_mismatch"),
    ("not one of the supplied subwindow frame timestamps", "window_timecode_mismatch"),
    ("Window timestamp lies outside", "window_timestamp_out_of_bounds"),
    ("Window observation text", "missing_window_observation"),
    ("Contradictory window cues", "contradictory_idle_cues"),
    ("Contradictory idle window", "contradictory_idle_cues"),
    ("Unknown window cue", "unknown_window_cue"),
    ("Duplicate window cue", "duplicate_window_cue"),
    ("Malformed or empty window cues", "malformed_window_cues"),
    ("Malformed cue value", "malformed_window_cues"),
    ("window rule_id", "invalid_window_rule_id"),
    ("Malformed candidate-scan windows", "malformed_windows"),
    ("Malformed candidate-scan window entry", "malformed_windows"),
    ("Unknown or malformed candidate rule", "invalid_candidate_rule"),
    ("Candidate timestamp lies outside", "candidate_timestamp_out_of_bounds"),
    ("Candidate start_sec lies outside", "candidate_timestamp_out_of_bounds"),
    ("Candidate end_sec lies outside", "candidate_timestamp_out_of_bounds"),
    ("Candidate confidence", "invalid_candidate_confidence"),
    ("Duplicate candidate", "duplicate_candidate"),
    ("Malformed candidate-scan response", "malformed_overview_schema"),
    ("Contradictory event confirmation", "contradictory_confirmation"),
    ("Malformed confirmation confidence", "malformed_confirmation_confidence"),
    ("Malformed confirmed event evidence", "malformed_confirmation_evidence"),
    ("Malformed confirmed event timestamps", "malformed_confirmation_timestamps"),
    ("Malformed verification response", "malformed_verification_schema"),
    ("numeric field", "invalid_numeric_field"),
)


ALLOWED_ERROR_REASONS: frozenset[str] = frozenset({
    "incomplete_json",
    "no_json_object",
    "unusable_view",
    "incomplete_timeline",
    "subwindow_count_mismatch",
    "window_bounds_mismatch",
    "window_timecode_mismatch",
    "window_timestamp_out_of_bounds",
    "missing_window_observation",
    "contradictory_idle_cues",
    "unknown_window_cue",
    "duplicate_window_cue",
    "malformed_window_cues",
    "invalid_window_rule_id",
    "malformed_windows",
    "invalid_candidate_rule",
    "candidate_timestamp_out_of_bounds",
    "invalid_candidate_confidence",
    "duplicate_candidate",
    "malformed_overview_schema",
    "contradictory_confirmation",
    "malformed_confirmation_confidence",
    "malformed_confirmation_evidence",
    "malformed_confirmation_timestamps",
    "malformed_verification_schema",
    "invalid_numeric_field",
    "validation_value_error",
    "timeout",
    "operational_error",
})


def _classify_error(exc: BaseException, error_stage: str) -> dict[str, Any]:
    """Return privacy-safe error classification and source line from a closed reason allowlist."""
    raw_reason = getattr(exc, "error_reason", None)
    if isinstance(raw_reason, str) and raw_reason in ALLOWED_ERROR_REASONS:
        reason = raw_reason
    elif isinstance(exc, TimeoutError):
        reason = "timeout"
    elif isinstance(exc, ValueError):
        msg = str(exc)
        reason = "validation_value_error"
        for pattern, code in ERROR_REASON_PATTERNS:
            if pattern in msg and code in ALLOWED_ERROR_REASONS:
                reason = code
                break
    else:
        reason = "operational_error"
    func_name = "unknown"
    line_no = 0
    tb = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
    for frame in reversed(tb):
        if Path(frame.filename).parent.name == "analysis":
            func_name = frame.name
            line_no = int(frame.lineno or 0)
            break
    if func_name == "unknown" and tb:
        func_name = tb[-1].name
        line_no = int(tb[-1].lineno or 0)
    return {
        "error_stage": error_stage,
        "error_type": type(exc).__name__,
        "error_reason": reason,
        "error_function": func_name,
        "error_line": line_no,
    }

CUE_TO_RULES: dict[str, list[tuple[str, float]]] = {
    "cash_receipt": [("cash_recount_on_receipt", 0.82)],
    "cash_delivery": [("cash_recount_on_delivery", 0.82)],
    "cash_on_desk": [("cash_separation_by_transaction", 0.80)],
    "cashier_leaves": [("cashier_leaves_during_service", 0.88)],
    "bag_handling": [("collection_bag_seal_check", 0.85)],
    "bag_open_or_damaged": [("damaged_collection_bag_not_recounted", 0.88), ("collection_bag_seal_check", 0.78)],
    "device_in_cash_area": [("camera_capable_device_in_cash_area", 0.84)],
    "desk_interaction": [
        ("cash_recount_on_receipt", 0.72),
        ("cash_recount_on_delivery", 0.70),
    ],
}

NEGATED_IDLE_CLAUSES = re.compile(
    r"(?:"
    r"\bне\s+(?:встает|встал|встала|уходит|ушел|ушла|выходит|вышел|вышла|покидает|покинул|покинула|отходит|отошел|отошла|использует|держит|снимает)\b"
    r"|\b(?:передач|денег|купюр|банкнот|наличных|сумок|сумки|мешков|мешка|пакетов|телефонов|телефона|смартфона|камер|устройств|действий|движений|клиентов|посетителей)\s+нет\b"
    r"|\bнет\s+(?:передач|денег|купюр|банкнот|наличных|сумок|сумки|мешков|пакетов|телефонов|телефона|смартфона|камер|устройств|действий|движений|клиентов|посетителей)\b"
    r"|\bбез\s+(?:денег|купюр|сумок|сумки|телефона|смартфона|камеры|передач|движений)\b"
    r")",
    flags=re.I,
)


def finite_number(value, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (ValueError, TypeError):
        return default


def active_rules(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rules = payload["rules"] if isinstance(payload, dict) else payload
    result = [r for r in rules if r.get("supported") is True and r.get("id") and r.get("source", {}).get("url")]
    if not result:
        raise RuntimeError("No verified normative rules are available")
    return result


def reviewed_examples(path: Path, rules_by_id: dict) -> list[dict]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("training_only") is not True:
        return []
    return [
        {"rule_id": e["rule_id"], "observed_example": str(e["observed_example"])[:600]}
        for e in payload.get("examples", [])
        if e.get("human_validated") is True and e.get("rule_id") in rules_by_id and e.get("observed_example")
    ][:8]


def clean_observation(value) -> str:
    """Legal citations and links are server-owned, never copied from model text."""
    if not isinstance(value, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "", value, flags=re.I)
    text = re.split(
        r"(?:нормативный источник|согласно\s+(?:пункту|правилам|статье)|в соответствии с\s+(?:пунктом|правилами|законом)|\bп\.\s*\d+)",
        text,
        maxsplit=1,
        flags=re.I,
    )[0]
    return text.strip(" \n\r\t.;")[:1400]


def analysis_profile(config: AnalysisConfig) -> str:
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
        if key not in {"private_dir", "confidence_threshold"}
    }
    files = list(Path(__file__).parent.glob("*.py")) + [
        config.rules_path,
        config.model_path / "download_provenance.json",
        config.grounding_model_path / "download_provenance.json",
        config.private_dir / "fewshot_examples.json",
    ]
    hashes = [(path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in sorted(files) if path.is_file()]
    return hashlib.sha256(json.dumps({"config": settings, "files": hashes}, sort_keys=True).encode()).hexdigest()


def validate_confirmation(verified: dict, rule_id: str, begin: float, end: float) -> dict | None:
    if not isinstance(verified.get("confirmed"), bool) or not isinstance(verified.get("sequence_visible"), bool):
        raise ValueError("Malformed verification response")
    if verified["confirmed"] is False:
        return None
    timestamp = finite_number(verified.get("timecode_sec"), -1)
    confidence = verified.get("confidence")
    description = clean_observation(verified.get("description"))
    evidence = verified.get("evidence_timestamps")
    if verified.get("rule_id") != rule_id or verified["sequence_visible"] is not True:
        raise ValueError("Contradictory event confirmation")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Malformed confirmation confidence")
    if isinstance(verified.get("timecode_sec"), bool) or not begin <= timestamp <= end or len(description) < 8:
        raise ValueError("Malformed confirmed event evidence")
    if not isinstance(evidence, list) or not evidence or any(isinstance(t, bool) or not begin <= finite_number(t, -1) <= end for t in evidence):
        raise ValueError("Malformed confirmed event timestamps")
    return {"timecode_sec": timestamp, "confidence": confidence, "description": description, "evidence_timestamps": evidence}


def overview_chunks(frames: list[VideoFrame], max_chunk_frames: int = MAX_OVERVIEW_CHUNK_FRAMES, overlap: int = OVERVIEW_CHUNK_OVERLAP) -> list[list[VideoFrame]]:
    """Split overview frames into bounded overlapping temporal chunks."""
    if not frames:
        return []
    if len(frames) <= max_chunk_frames + 1:
        return [list(frames)]
    step = max(1, max_chunk_frames - max(0, overlap))
    chunks: list[list[VideoFrame]] = []
    index = 0
    while index < len(frames):
        chunk = list(frames[index : index + max_chunk_frames])
        if len(chunk) == 1 and chunks:
            chunks[-1].append(chunk[0])
            break
        chunks.append(chunk)
        if index + max_chunk_frames >= len(frames):
            break
        index += step
    return chunks


def _require_finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Malformed numeric field: {field_name}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric field: {field_name}")
    return number


def _rules_from_observed_text(observed: str) -> list[tuple[str, float]]:
    cleaned = NEGATED_IDLE_CLAUSES.sub(" ", observed.lower())
    matches: list[tuple[str, float]] = []

    if re.search(r"(?:вскры|распак|поврежд|порван|разрез|откры\w*\s+(?:сумк|меш|пакет|пломб)|unsealed|damaged\s+bag|unpack)", cleaned):
        matches.append(("damaged_collection_bag_not_recounted", 0.86))
        matches.append(("collection_bag_seal_check", 0.78))
    elif re.search(r"(?:сумк|мешок|мешк|инкассат|пломб|баул|сверт|спецпакет|\bbag\b|\bseal\b|\bpouch\b)", cleaned):
        matches.append(("collection_bag_seal_check", 0.84))

    if re.search(
        r"(?:встает|встал|встала|уходит|ушел|ушла|выходит|вышел|вышла|покидает|покинул|покинула|"
        r"отошел|отошла|отходит\s+от\s+стол|пуст\w*\s+(?:кресл|стул)|нет\s+на\s+месте|отсутствует\s+за\s+стол|"
        r"направил\w*\s+к\s+(?:двер|выход)|stands\s+up|walks\s+away|leaves\s+(?:desk|work))",
        cleaned,
    ):
        matches.append(("cashier_leaves_during_service", 0.86))

    if re.search(
        r"(?:телефон|смартфон|айфон|камер\w*\s+(?:телефон|устройств|навед|съемк)|снимает\s+на\s+(?:телефон|камер)|"
        r"фотограф|видеосъемк|мобильн\w*\s+(?:телефон|устройств)|держит\s+(?:телефон|смартфон)|"
        r"\bphone\b|\bsmartphone\b|camera\s+device)",
        cleaned,
    ):
        matches.append(("camera_capable_device_in_cash_area", 0.84))

    has_receipt = bool(
        re.search(
            r"(?:принимает|принял|приняла|получает|получил|получила|берет\s+(?:деньг|купюр|пачк|наличн)|"
            r"взял\w*\s+(?:деньг|купюр|пачк)|(?:клиент|посетитель|человек)\s+(?:передает|дает|протягивает)|"
            r"передает\s+кассир|протягивает\s+кассир|receives\s+cash|hands\s+banknotes\s+to\s+cashier)",
            cleaned,
        )
    )
    has_delivery = bool(
        re.search(
            r"(?:выдает|выдал|выдала|отдает|отдал|отдала|вручает|передает\s+(?:клиент|посетител|человек)|"
            r"протягивает\s+(?:клиент|посетител|человек)|delivers\s+cash|hands\s+banknotes\s+to\s+(?:client|person|customer))",
            cleaned,
        )
    )
    has_desk_cash = bool(
        re.search(
            r"(?:(?:деньг|купюр|банкнот|наличн|пачк)\w*\s+(?:лежат|лежит|остал\w*|на\s+стол|на\s+прилавк)|"
            r"на\s+стол\w*\s+(?:лежат|лежит|остал\w*)\s+(?:деньг|купюр|банкнот|наличн|пачк)|"
            r"не\s+убра\w*\s+(?:деньг|купюр|со\s+стол)|cash\s+(?:on|left\s+on)\s+(?:desk|table|counter))",
            cleaned,
        )
    )
    has_general_cash = bool(
        re.search(r"(?:деньг|денег|купюр|банкнот|наличн|пересчет|пересчитыв|счетн\w*\s+машин|\bcash\b|\bbanknote)", cleaned)
    )

    if has_receipt:
        matches.append(("cash_recount_on_receipt", 0.82))
    if has_delivery:
        matches.append(("cash_recount_on_delivery", 0.82))
    if has_desk_cash:
        matches.append(("cash_separation_by_transaction", 0.80))
    if has_general_cash and not (has_receipt or has_delivery or has_desk_cash):
        matches.append(("cash_recount_on_receipt", 0.76))
        matches.append(("cash_recount_on_delivery", 0.74))

    return matches


def extract_window_candidates(
    windows: Any,
    rules_by_id: dict[str, dict],
    duration: float,
    expected_windows: list[dict[str, Any]] | None = None,
    chunk_index: int = 0,
) -> list[dict[str, Any]]:
    """Strictly validate perceived sub-windows against requested subwindows and map actions to candidate rules."""
    if not isinstance(windows, list) or not windows:
        raise ValueError("Malformed candidate-scan windows")
    if expected_windows is not None and len(windows) != len(expected_windows):
        raise ValueError("Malformed candidate-scan windows: subwindow count mismatch")
    allowed_cues = set(ACTION_CUES)
    derived: list[dict[str, Any]] = []

    for win_idx, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError("Malformed candidate-scan window entry")
        start_sec = _require_finite_number(window.get("start_sec"), "window.start_sec")
        end_sec = _require_finite_number(window.get("end_sec"), "window.end_sec")
        timecode = _require_finite_number(window.get("timecode_sec"), "window.timecode_sec")
        if (
            start_sec < -TIMESTAMP_TOLERANCE_SEC
            or end_sec > duration + TIMESTAMP_TOLERANCE_SEC
            or start_sec > end_sec + TIMESTAMP_TOLERANCE_SEC
            or not (start_sec - TIMESTAMP_TOLERANCE_SEC <= timecode <= end_sec + TIMESTAMP_TOLERANCE_SEC)
        ):
            raise ValueError("Window timestamp lies outside the subwindow or video bounds")

        if expected_windows is not None:
            exp = expected_windows[win_idx]
            exp_start = float(exp["start_sec"])
            exp_end = float(exp["end_sec"])
            if abs(start_sec - exp_start) > TIMESTAMP_TOLERANCE_SEC or abs(end_sec - exp_end) > TIMESTAMP_TOLERANCE_SEC:
                raise ValueError("Window bounds do not match requested subwindow order or timestamps")
            stamps = exp.get("timestamps") or []
            if stamps and not any(abs(timecode - float(ts)) <= TIMESTAMP_TOLERANCE_SEC for ts in stamps):
                raise ValueError("Window timecode_sec is not one of the supplied subwindow frame timestamps")

        raw_observed = window.get("observed")
        if not isinstance(raw_observed, str):
            raise ValueError("Window observation text must be a string")
        observed = clean_observation(raw_observed)
        if len(observed) < 3:
            raise ValueError("Window observation text is missing")

        if "cues" not in window or not isinstance(window.get("cues"), list) or not window["cues"]:
            raise ValueError("Malformed or empty window cues")
        cues = window["cues"]
        seen_cues: set[str] = set()
        for cue in cues:
            if not isinstance(cue, str):
                raise ValueError("Malformed cue value: must be string")
            if cue not in allowed_cues:
                raise ValueError(f"Unknown window cue: {cue}")
            if cue in seen_cues:
                raise ValueError(f"Duplicate window cue: {cue}")
            seen_cues.add(cue)
        if "idle" in seen_cues and len(seen_cues) > 1:
            raise ValueError("Contradictory window cues: idle combined with active cue")

        if "rule_ids" not in window or not isinstance(window.get("rule_ids"), list):
            raise ValueError("Malformed window rule_ids")
        rule_ids = window["rule_ids"]
        seen_rids: set[str] = set()
        for rid in rule_ids:
            if not isinstance(rid, str) or rid not in rules_by_id:
                raise ValueError("Unknown or non-string window rule_id")
            if rid in seen_rids:
                raise ValueError("Duplicate window rule_id")
            seen_rids.add(rid)
        if "idle" in seen_cues and seen_rids:
            raise ValueError("Contradictory idle window with non-empty rule_ids")

        scored_rules: dict[str, float] = {}
        for rid in rule_ids:
            scored_rules[rid] = max(scored_rules.get(rid, 0.0), 0.85)

        for cue in cues:
            if cue in CUE_TO_RULES:
                for mapped_rule, conf in CUE_TO_RULES[cue]:
                    if mapped_rule in rules_by_id:
                        scored_rules[mapped_rule] = max(scored_rules.get(mapped_rule, 0.0), conf)

        if "idle" not in seen_cues:
            for mapped_rule, conf in _rules_from_observed_text(observed):
                if mapped_rule in rules_by_id:
                    scored_rules[mapped_rule] = max(scored_rules.get(mapped_rule, 0.0), conf)

        for rid, conf in scored_rules.items():
            derived.append({
                "rule_id": rid,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "timecode_sec": round(timecode, 3),
                "confidence": round(min(1.0, max(0.0, conf)), 3),
                "observed": observed[:160],
                "uncertainty": "Проверить последовательность действий на плотных кадрах",
                "source": "window",
                "chunk_index": chunk_index,
                "window_index": win_idx,
            })
    return derived


def validate_overview_payload(
    overview: Any,
    subwindows: list[dict[str, Any]],
    chunk: list[VideoFrame],
    rules_by_id: dict[str, dict],
    duration: float,
    chunk_index: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate overview stage structure, subwindow coverage, and explicit candidates."""
    if (
        not isinstance(overview, dict)
        or not isinstance(overview.get("candidates"), list)
        or not isinstance(overview.get("view_usable"), bool)
        or not isinstance(overview.get("timeline_complete"), bool)
    ):
        raise ValueError("Malformed candidate-scan response")
    if overview.get("view_usable") is False:
        raise ValueError("Video view is not usable for visual review")
    if overview["timeline_complete"] is False:
        raise ValueError("Candidate scan did not cover the complete timeline")

    chunk_win_candidates = extract_window_candidates(
        overview.get("windows"),
        rules_by_id,
        duration,
        expected_windows=subwindows,
        chunk_index=chunk_index,
    )

    full_frames = [f for f in chunk if getattr(f, "view", "full frame") == "full frame"] or list(chunk)
    chunk_stamps = [round(float(f.timestamp_sec), 3) for f in full_frames]
    chunk_start = chunk_stamps[0] if chunk_stamps else 0.0
    chunk_end = chunk_stamps[-1] if chunk_stamps else duration

    chunk_explicit: list[dict[str, Any]] = []
    seen_explicit: set[tuple[str, float]] = set()
    for candidate in overview.get("candidates", []):
        if not isinstance(candidate, dict):
            raise ValueError("Unknown or malformed candidate rule")
        rid = candidate.get("rule_id")
        if not isinstance(rid, str) or rid not in rules_by_id:
            raise ValueError("Unknown or malformed candidate rule")
        timestamp = _require_finite_number(candidate.get("timecode_sec"), "candidate.timecode_sec")
        if (
            not (0.0 <= timestamp <= duration)
            or not (chunk_start - TIMESTAMP_TOLERANCE_SEC <= timestamp <= chunk_end + TIMESTAMP_TOLERANCE_SEC)
            or (chunk_stamps and not any(abs(timestamp - ts) <= TIMESTAMP_TOLERANCE_SEC for ts in chunk_stamps))
        ):
            raise ValueError("Candidate timestamp lies outside the chunk or supplied frames")

        if "start_sec" in candidate and candidate["start_sec"] is not None:
            c_start = _require_finite_number(candidate["start_sec"], "candidate.start_sec")
            if not (chunk_start - TIMESTAMP_TOLERANCE_SEC <= c_start <= timestamp + TIMESTAMP_TOLERANCE_SEC):
                raise ValueError("Candidate start_sec lies outside the chunk")
        else:
            c_start = timestamp

        if "end_sec" in candidate and candidate["end_sec"] is not None:
            c_end = _require_finite_number(candidate["end_sec"], "candidate.end_sec")
            if not (timestamp - TIMESTAMP_TOLERANCE_SEC <= c_end <= chunk_end + TIMESTAMP_TOLERANCE_SEC):
                raise ValueError("Candidate end_sec lies outside the chunk")
        else:
            c_end = timestamp

        if "confidence" in candidate and candidate["confidence"] is not None:
            conf = _require_finite_number(candidate["confidence"], "candidate.confidence")
            if not (0.0 <= conf <= 1.0):
                raise ValueError("Candidate confidence must be between 0 and 1")
        else:
            conf = 0.75

        key = (rid, round(timestamp, 3))
        if key in seen_explicit:
            raise ValueError("Duplicate candidate in chunk")
        seen_explicit.add(key)

        cand_copy = dict(candidate)
        cand_copy["rule_id"] = rid
        cand_copy["start_sec"] = round(c_start, 3)
        cand_copy["end_sec"] = round(c_end, 3)
        cand_copy["timecode_sec"] = round(timestamp, 3)
        cand_copy["confidence"] = round(conf, 3)
        cand_copy["source"] = "explicit"
        cand_copy["chunk_index"] = chunk_index
        chunk_explicit.append(cand_copy)

    return chunk_explicit, chunk_win_candidates


def _same_episode(existing: dict[str, Any], incoming: dict[str, Any]) -> bool:
    """Determine if two candidates of the same rule refer to the exact same observed episode."""
    if existing["rule_id"] != incoming["rule_id"]:
        return False
    if abs(existing["timecode_sec"] - incoming["timecode_sec"]) < 1.0:
        return True
    e_start = finite_number(existing.get("start_sec"), existing["timecode_sec"])
    e_end = finite_number(existing.get("end_sec"), existing["timecode_sec"])
    i_start = finite_number(incoming.get("start_sec"), incoming["timecode_sec"])
    i_end = finite_number(incoming.get("end_sec"), incoming["timecode_sec"])
    intervals_overlap = max(e_start, i_start) <= min(e_end, i_end) + TIMESTAMP_TOLERANCE_SEC
    if not intervals_overlap:
        return False
    # If one is explicit and one is a window candidate from the same chunk covering the same span,
    # or two window candidates overlap across an adjacent chunk boundary frame, merge them.
    if existing.get("source") != incoming.get("source") and existing.get("chunk_index") == incoming.get("chunk_index"):
        return True
    if existing.get("source") == "window" and incoming.get("source") == "window" and existing.get("chunk_index") != incoming.get("chunk_index"):
        return True
    return False


class LegacyTwoStageAnalyzer:
    def __init__(self, config: AnalysisConfig | None = None):
        self.config = config or AnalysisConfig.from_env()
        self.vlm = LocalVLM(self.config)
        self.grounder = SceneGrounder(self.config)
        self.last_metrics: dict = {}

    def warmup(self) -> None:
        self.vlm.load()
        self.grounder.load()

    def analyze(self, video_path: Path, progress: Callable[[str, float], None] | None = None) -> dict:
        start = time.monotonic()
        deadline = start + self.config.timeout_sec
        report_id = uuid.uuid4().hex
        self.config.private_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.config.private_dir / f"{report_id}.json"
        report: dict = {
            "id": report_id,
            "video_path": str(video_path),
            "model": self.config.model_path.name,
            "analysis_profile": analysis_profile(self.config),
            "stages": [],
        }
        self.last_metrics = {}
        current_stage = "init"
        retries_used = 0
        escaping_stage: dict[str, Any] | None = None

        def notify(stage: str, fraction: float) -> None:
            if progress:
                progress(stage, min(1.0, max(0.0, fraction)))

        try:
            current_stage = "decoding"
            notify("decoding", 0.03)
            video = sample_video(Path(video_path), fps=self.config.scan_fps)
            report["duration_sec"] = video.duration_sec
            current_stage = "scene_calibration"
            notify("scene_calibration", 0.07)
            regions, grounding_stats = self.grounder.calibrate(video, deadline)
            report["grounding"] = grounding_stats
            rules = active_rules(self.config.rules_path)
            rules_by_id = {rule["id"]: rule for rule in rules}
            examples = reviewed_examples(self.config.private_dir / "fewshot_examples.json", rules_by_id)
            scan = overview_frames(video, self.config.overview_frames)
            chunks = overview_chunks(scan)
            if not chunks:
                raise ValueError("No overview frames available for analysis")

            explicit_candidates: list[dict[str, Any]] = []
            window_candidates: list[dict[str, Any]] = []

            for chunk_index, chunk in enumerate(chunks):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Analysis deadline reached")
                current_stage = "overview"
                notify("overview", 0.12 + 0.20 * chunk_index / max(1, len(chunks)))
                subwindows = chunk_subwindows(chunk, video.duration_sec)
                visual_chunk = sorted(chunk + detail_crops(chunk, regions), key=lambda frame: frame.timestamp_sec)
                base_ov_prompt = overview_prompt(rules, video.duration_sec, len(visual_chunk), examples, windows=subwindows)
                last_ov_reason = ""
                chunk_explicit: list[dict[str, Any]] = []
                chunk_win_candidates: list[dict[str, Any]] = []

                for attempt in range(MAX_STAGE_ATTEMPTS):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Analysis deadline reached")
                    prompt_to_use = (
                        base_ov_prompt
                        if attempt == 0
                        else format_retry_prompt(base_ov_prompt, "overview", last_ov_reason, subwindows=subwindows)
                    )
                    try:
                        overview, stats, raw = self.vlm.generate(
                            visual_chunk,
                            prompt_to_use,
                            max_pixels=self.config.overview_pixels,
                            max_tokens=min(650, self.config.generation_tokens),
                            deadline=deadline,
                        )
                    except TimeoutError as exc:
                        diag = _classify_error(exc, "overview")
                        gen_stats = getattr(exc, "generation_stats", None)
                        raw_resp = getattr(exc, "raw_response", None)
                        if isinstance(gen_stats, dict) or raw_resp is not None:
                            timeout_stage = {
                                "stage": "overview",
                                "chunk_index": chunk_index,
                                "attempt": attempt + 1,
                                "status": "failed_timeout",
                                "stats": gen_stats if isinstance(gen_stats, dict) else {},
                                "result": None,
                                "raw": raw_resp,
                                **diag,
                            }
                            report["stages"].append(timeout_stage)
                            escaping_stage = timeout_stage
                        raise
                    except ValueError as exc:
                        diag = _classify_error(exc, "overview_decode")
                        gen_stats = getattr(exc, "generation_stats", None)
                        raw_resp = getattr(exc, "raw_response", None)
                        decode_stage: dict[str, Any] | None = None
                        if isinstance(gen_stats, dict) or raw_resp is not None:
                            decode_stage = {
                                "stage": "overview",
                                "chunk_index": chunk_index,
                                "attempt": attempt + 1,
                                "status": "failed_decode",
                                "stats": gen_stats if isinstance(gen_stats, dict) else {},
                                "result": None,
                                "raw": raw_resp,
                                **diag,
                            }
                            report["stages"].append(decode_stage)
                        if attempt + 1 < MAX_STAGE_ATTEMPTS and diag["error_reason"] not in NON_RETRYABLE_REASONS and raw_resp is not None:
                            last_ov_reason = diag["error_reason"]
                            retries_used += 1
                            continue
                        escaping_stage = decode_stage
                        raise

                    stage_entry: dict[str, Any] = {
                        "stage": "overview",
                        "chunk_index": chunk_index,
                        "attempt": attempt + 1,
                        "status": "ok",
                        "stats": stats,
                        "result": overview,
                        "raw": raw,
                    }
                    report["stages"].append(stage_entry)
                    try:
                        chunk_explicit, chunk_win_candidates = validate_overview_payload(
                            overview,
                            subwindows,
                            chunk,
                            rules_by_id,
                            video.duration_sec,
                            chunk_index=chunk_index,
                        )
                        break
                    except ValueError as exc:
                        diag = _classify_error(exc, "overview_validation")
                        stage_entry["status"] = "failed_validation"
                        stage_entry.update(diag)
                        if attempt + 1 < MAX_STAGE_ATTEMPTS and diag["error_reason"] not in NON_RETRYABLE_REASONS:
                            last_ov_reason = diag["error_reason"]
                            retries_used += 1
                            continue
                        escaping_stage = stage_entry
                        raise

                for cand_copy in chunk_explicit:
                    if any(_same_episode(c, cand_copy) for c in explicit_candidates):
                        continue
                    explicit_candidates.append(cand_copy)
                window_candidates.extend(chunk_win_candidates)

            candidates: list[dict[str, Any]] = list(explicit_candidates)
            for win_cand in window_candidates:
                merged = False
                for existing in candidates:
                    if _same_episode(existing, win_cand):
                        existing["start_sec"] = min(finite_number(existing.get("start_sec"), existing["timecode_sec"]), win_cand["start_sec"])
                        existing["end_sec"] = max(finite_number(existing.get("end_sec"), existing["timecode_sec"]), win_cand["end_sec"])
                        if win_cand["confidence"] > existing["confidence"]:
                            existing["confidence"] = win_cand["confidence"]
                            existing["timecode_sec"] = win_cand["timecode_sec"]
                        merged = True
                        break
                if not merged:
                    candidates.append(dict(win_cand))

            candidates.sort(key=lambda c: (c["confidence"], -c["timecode_sec"]), reverse=True)

            violations = []
            for index, candidate in enumerate(candidates):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Analysis deadline reached")
                current_stage = "verification"
                rule = rules_by_id[candidate["rule_id"]]
                center = candidate["timecode_sec"]
                half_span = 18.0 if rule.get("requires_complete_action_sequence") else 12.0
                begin = max(0.0, min(center - 4.0, finite_number(candidate.get("start_sec"), center - 4.0) - 2.0))
                end = min(video.duration_sec, max(center + 4.0, finite_number(candidate.get("end_sec"), center + 4.0) + 2.0))
                begin = max(0.0, max(begin, center - half_span))
                end = min(video.duration_sec, min(end, center + half_span))
                detail = window_frames(video, begin, end, self.config.detail_frames, center=center)
                if not detail:
                    raise ValueError("No detail frames available for candidate verification")
                # Align bounds with 3-decimal prompt formatting so valid boundary
                # frame timestamps chosen by the VLM pass validation cleanly.
                begin = min(begin, math.floor(detail[0].timestamp_sec * 1000.0) / 1000.0)
                end = max(end, math.ceil(detail[-1].timestamp_sec * 1000.0) / 1000.0)
                visual_detail = sorted(detail + detail_crops(detail, regions), key=lambda frame: frame.timestamp_sec)
                frame_stamps = [round(frame.timestamp_sec, 3) for frame in detail]
                notify("verification", 0.35 + 0.55 * index / max(1, len(candidates)))
                base_ver_prompt = verification_prompt(rule, candidate, video.duration_sec, frame_timestamps=frame_stamps)
                last_ver_reason = ""
                confirmation: dict[str, Any] | None = None
                stage_record: dict[str, Any] = {}

                for attempt in range(MAX_STAGE_ATTEMPTS):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Analysis deadline reached")
                    ver_prompt = (
                        base_ver_prompt
                        if attempt == 0
                        else format_retry_prompt(base_ver_prompt, "verification", last_ver_reason, frame_timestamps=frame_stamps)
                    )
                    try:
                        verified, stats, raw = self.vlm.generate(
                            visual_detail,
                            ver_prompt,
                            max_pixels=self.config.detail_pixels,
                            max_tokens=min(700, self.config.generation_tokens),
                            deadline=deadline,
                        )
                    except TimeoutError as exc:
                        diag = _classify_error(exc, "verification")
                        gen_stats = getattr(exc, "generation_stats", None)
                        raw_resp = getattr(exc, "raw_response", None)
                        if isinstance(gen_stats, dict) or raw_resp is not None:
                            timeout_stage = {
                                "stage": "verification",
                                "attempt": attempt + 1,
                                "status": "failed_timeout",
                                "stats": gen_stats if isinstance(gen_stats, dict) else {},
                                "candidate": candidate,
                                "result": None,
                                "raw": raw_resp,
                                "window_start": begin,
                                "window_end": end,
                                "acceptance_schema": 1,
                                "evidence_valid_before_threshold": False,
                                **diag,
                            }
                            report["stages"].append(timeout_stage)
                            escaping_stage = timeout_stage
                        raise
                    except ValueError as exc:
                        diag = _classify_error(exc, "verification_decode")
                        gen_stats = getattr(exc, "generation_stats", None)
                        raw_resp = getattr(exc, "raw_response", None)
                        decode_stage: dict[str, Any] | None = None
                        if isinstance(gen_stats, dict) or raw_resp is not None:
                            decode_stage = {
                                "stage": "verification",
                                "attempt": attempt + 1,
                                "status": "failed_decode",
                                "stats": gen_stats if isinstance(gen_stats, dict) else {},
                                "candidate": candidate,
                                "result": None,
                                "raw": raw_resp,
                                "window_start": begin,
                                "window_end": end,
                                "acceptance_schema": 1,
                                "evidence_valid_before_threshold": False,
                                **diag,
                            }
                            report["stages"].append(decode_stage)
                        if attempt + 1 < MAX_STAGE_ATTEMPTS and diag["error_reason"] not in NON_RETRYABLE_REASONS and raw_resp is not None:
                            last_ver_reason = diag["error_reason"]
                            retries_used += 1
                            continue
                        escaping_stage = decode_stage
                        raise

                    stage_record = {
                        "stage": "verification",
                        "attempt": attempt + 1,
                        "status": "ok",
                        "stats": stats,
                        "candidate": candidate,
                        "result": verified,
                        "raw": raw,
                        "window_start": begin,
                        "window_end": end,
                        "acceptance_schema": 1,
                        "evidence_valid_before_threshold": False,
                    }
                    report["stages"].append(stage_record)
                    try:
                        confirmation = validate_confirmation(verified, rule["id"], begin, end)
                        break
                    except ValueError as exc:
                        diag = _classify_error(exc, "verification_validation")
                        stage_record["status"] = "failed_validation"
                        stage_record.update(diag)
                        if attempt + 1 < MAX_STAGE_ATTEMPTS and diag["error_reason"] not in NON_RETRYABLE_REASONS:
                            last_ver_reason = diag["error_reason"]
                            retries_used += 1
                            continue
                        escaping_stage = stage_record
                        raise

                if confirmation is None:
                    continue
                stage_record["evidence_valid_before_threshold"] = True
                if confirmation["confidence"] < self.config.confidence_threshold:
                    continue
                timestamp = confirmation["timecode_sec"]
                timestamp = min(detail, key=lambda frame: abs(frame.timestamp_sec - timestamp)).timestamp_sec
                if any(v["type"] == rule["title"] and abs(v["timecode_sec"] - round(timestamp, 3)) < 1.0 for v in violations):
                    continue
                description = confirmation["description"]
                requested = confirmation["evidence_timestamps"]
                timestamps = [timestamp]
                for value in requested:
                    value = finite_number(value, -1)
                    if begin <= value <= end:
                        timestamps.append(min(detail, key=lambda frame: abs(frame.timestamp_sec - value)).timestamp_sec)
                source = rule["source"]
                source_label = str(source.get("title", "Нормативный источник РК"))
                if source.get("paragraph"):
                    source_label += ", " + str(source["paragraph"])
                # Legal references always come from the reviewed static catalog.
                description += f". Требование: {rule.get('description', rule['title'])} Нормативный источник: {source_label}. {source['url']}"
                violations.append({
                    "type": rule["title"],
                    "timecode_sec": round(timestamp, 3),
                    "description": description,
                    "evidence_frames": evidence_frames(video, timestamps, count=3),
                })
            violations.sort(key=lambda v: v["timecode_sec"])
            result = {"verdict": "violation" if violations else "no_violation", "violations": violations}
            report["result"] = result
            notify("done", 1.0)
            return result
        except Exception as exc:
            # Controlled error metadata only is safe in operational logs; visual
            # narratives stay in the private artifact and must never be exposed.
            diag = (
                {
                    "error_stage": escaping_stage["error_stage"],
                    "error_type": escaping_stage["error_type"],
                    "error_reason": escaping_stage["error_reason"],
                    "error_function": escaping_stage["error_function"],
                    "error_line": escaping_stage["error_line"],
                }
                if escaping_stage is not None and "error_reason" in escaping_stage
                else _classify_error(exc, current_stage)
            )
            report.update(diag)
            report["raw_available"] = bool(escaping_stage is not None and escaping_stage.get("raw") is not None)
            raise
        finally:
            seconds = round(time.monotonic() - start, 3)
            self.last_metrics = {
                "seconds": seconds,
                "stages": len(report["stages"]),
                "retries": retries_used,
                "input_tokens": sum(s.get("stats", {}).get("input_tokens", 0) for s in report["stages"]),
                "output_tokens": sum(s.get("stats", {}).get("output_tokens", 0) for s in report["stages"]),
                "gpu_peak_mb": max((s.get("stats", {}).get("gpu_peak_mb", 0) for s in report["stages"]), default=0),
                "error_type": report.get("error_type"),
                "error_stage": report.get("error_stage"),
                "error_reason": report.get("error_reason"),
                "error_function": report.get("error_function"),
                "error_line": report.get("error_line"),
                "raw_available": report.get("raw_available", any(s.get("raw") is not None for s in report["stages"])),
            }
            report["metrics"] = self.last_metrics
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

# Active path: full-video chronological observation and holistic verification.
from .full_timeline import VideoAnalyzer

