"""Synthetic checks of evidence/decision boundaries, without model or dataset."""

import base64
import io
import json
from dataclasses import replace

import pytest
from PIL import Image

from forte.analysis.config import AnalysisConfig
from forte.analysis.pipeline import VideoAnalyzer, finite_number
from forte.analysis.prompts import rule_context
from forte.analysis.video import SampledVideo, VideoFrame, evidence_frames, overview_frames, window_frames
from forte.analysis.vlm import decode_json


def test_evidence_uses_real_timestamps_and_jpeg():
    video = SampledVideo([VideoFrame(t, Image.new("RGB", (64, 96), (30, 70, 100))) for t in (0., .534, 1.067, 1.601)], 1.7, 64, 96)
    evidence = evidence_frames(video, [.5, .6, 1.5, 0.], count=3)
    assert [e["timestamp_sec"] for e in evidence] == [0., .534, 1.601]
    for item in evidence:
        blob = base64.b64decode(item["image_base64"], validate=True)
        assert blob[:2] == b"\xff\xd8"
        assert Image.open(io.BytesIO(blob)).format == "JPEG"


def test_sampling_covers_timeline_and_center():
    video = SampledVideo([VideoFrame(float(i), Image.new("RGB", (32, 32)), motion=.7 if i == 32 else 0) for i in range(60)], 60., 32, 32)
    frames = overview_frames(video, 12)
    assert frames[0].timestamp_sec == 0
    assert frames[-1].timestamp_sec == 59
    assert len(frames) == 12
    detail = window_frames(video, 20, 40, 8, center=31)
    assert len(detail) == 8
    assert 31 in [f.timestamp_sec for f in detail]
    assert detail[0].timestamp_sec == 20 and detail[-1].timestamp_sec == 40


def test_nonfinite_model_values_cannot_be_timestamps():
    assert finite_number("NaN", -1) == -1
    assert finite_number(float("inf"), -1) == -1
    assert finite_number("2.4") == 2.4


def test_legal_exceptions_reach_local_model():
    rule = {"id": "demo", "exceptions": ["sealed package exception"], "requires_complete_action_sequence": True}
    context = json.loads(rule_context([rule]))[0]
    assert context["exceptions"] == rule["exceptions"]
    assert context["requires_complete_action_sequence"] is True


class FakeVLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, *args, **kwargs):
        value = next(self.responses)
        return value, {}, json.dumps(value)


def synthetic_analyzer(tmp_path, monkeypatch, responses):
    rule = {"id": "receipt", "title": "Synthetic rule", "supported": True, "source": {"title": "Test source", "url": "https://example.org/rule", "paragraph": "1"}}
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps({"rules": [rule]}), encoding="utf-8")
    config = replace(AnalysisConfig.from_env(), rules_path=rules_path, private_dir=tmp_path / "reports", grounding_enabled=False)
    analyzer = VideoAnalyzer(config)
    analyzer.vlm = FakeVLM(responses)
    video = SampledVideo([VideoFrame(float(t), Image.new("RGB", (32, 32))) for t in range(10)], 10., 32, 32)
    monkeypatch.setattr("forte.analysis.pipeline.sample_video", lambda *args, **kwargs: video)
    return analyzer


def synthetic_windows(active_second: bool = False):
    return [
        {
            "start_sec": 0.0,
            "end_sec": 2.0,
            "timecode_sec": 0.0,
            "observed": "Кассир сидит за рабочим столом, передач нет",
            "cues": ["idle"],
            "rule_ids": [],
        },
        {
            "start_sec": 3.0,
            "end_sec": 5.0,
            "timecode_sec": 4.0 if active_second else 3.0,
            "observed": "Кассир принимает купюры у клиента" if active_second else "Кассир работает за столом, передач нет",
            "cues": ["cash_receipt"] if active_second else ["idle"],
            "rule_ids": ["receipt"] if active_second else [],
        },
        {
            "start_sec": 6.0,
            "end_sec": 9.0,
            "timecode_sec": 6.0,
            "observed": "Кассир сидит на рабочем месте, передач нет",
            "cues": ["idle"],
            "rule_ids": [],
        },
    ]


def test_malformed_scan_is_failure_not_clean_video(tmp_path, monkeypatch):
    analyzer = synthetic_analyzer(tmp_path, monkeypatch, [{"view_usable": True}, {"view_usable": True}])
    with pytest.raises(ValueError, match="Malformed"):
        analyzer.analyze(tmp_path / "synthetic.mov")

    # Missing `windows` must fail closed even when `candidates` is present.
    no_win = {"view_usable": True, "timeline_complete": True, "candidates": []}
    analyzer_no_win = synthetic_analyzer(tmp_path, monkeypatch, [no_win, no_win])
    with pytest.raises(ValueError, match="Malformed candidate-scan windows"):
        analyzer_no_win.analyze(tmp_path / "synthetic.mov")


def test_valid_negative_is_empty(tmp_path, monkeypatch):
    overview = {"view_usable": True, "timeline_complete": True, "windows": synthetic_windows(False), "candidates": []}
    analyzer = synthetic_analyzer(tmp_path, monkeypatch, [overview])
    assert analyzer.analyze(tmp_path / "synthetic.mov") == {"verdict": "no_violation", "violations": []}


def test_confirmation_requires_visible_sequence_and_static_citation(tmp_path, monkeypatch):
    overview = {
        "view_usable": True,
        "timeline_complete": True,
        "windows": synthetic_windows(True),
        "candidates": [{"rule_id": "receipt", "start_sec": 3.0, "end_sec": 5.0, "timecode_sec": 4.0, "confidence": .9}],
    }
    verification = {"confirmed": True, "sequence_visible": False, "rule_id": "receipt", "confidence": .95, "timecode_sec": 4, "description": "Synthetic visible event", "evidence_timestamps": [3, 4, 5]}
    analyzer = synthetic_analyzer(tmp_path, monkeypatch, [overview, verification, verification])
    with pytest.raises(ValueError, match="Contradictory"):
        analyzer.analyze(tmp_path / "synthetic.mov")
    verification["sequence_visible"] = True
    analyzer = synthetic_analyzer(tmp_path, monkeypatch, [overview, verification])
    result = analyzer.analyze(tmp_path / "synthetic.mov")
    event = result["violations"][0]
    assert event["timecode_sec"] == 4
    assert "Test source, 1. https://example.org/rule" in event["description"]
    assert len(event["evidence_frames"]) == 3


def test_json_extraction_does_not_execute_text():
    assert decode_json('```json\n{"a":1}\n```') == {"a": 1}
    with pytest.raises(ValueError):
        decode_json("no structured response")
    with pytest.raises(ValueError, match="incomplete"):
        decode_json('{"candidates":[{"rule_id":"demo"}')


def test_failed_positive_counts_against_end_to_end_recall():
    from scripts.evaluate import summarize
    result = summarize([
        {"label": 1, "predicted": 1, "seconds": 10},
        {"label": 1, "predicted": None, "seconds": 20, "error_type": "TimeoutError"},
        {"label": 0, "predicted": 0, "seconds": 10},
    ])
    assert result["recall"] == .5
    assert result["accuracy"] == .6667
    assert result["failed_positive"] == 1
