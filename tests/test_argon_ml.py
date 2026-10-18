"""Synthetic unit tests for chunked action perception and normative verification."""

from __future__ import annotations

import json
from pathlib import Path

import av
import pytest
from PIL import Image, ImageDraw

from forte.analysis.config import AnalysisConfig
from forte.analysis.pipeline import (
    VideoAnalyzer,
    clean_observation,
    extract_window_candidates,
    overview_chunks,
    validate_confirmation,
    validate_overview_payload,
)
from forte.analysis.prompts import chunk_subwindows, overview_prompt, verification_prompt
from forte.analysis.video import VideoFrame
from forte.analysis.vlm import decode_json


def _write_synthetic_video(path: Path, duration_sec: int = 8, fps: int = 2) -> Path:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = 320
    stream.height = 240
    stream.pix_fmt = "yuv420p"
    total_frames = max(2, duration_sec * fps)
    for idx in range(total_frames):
        img = Image.new("RGB", (320, 240), (30, 35, 45))
        draw = ImageDraw.Draw(img)
        # Desk surface
        draw.rectangle([40, 150, 280, 220], fill=(90, 70, 50))
        # Moving object / figure across frames
        x = 60 + (idx * 12) % 180
        draw.rectangle([x, 90, x + 40, 145], fill=(220, 190, 60))
        frame = av.VideoFrame.from_image(img)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def _write_rules(path: Path) -> Path:
    rules = {
        "rules": [
            {
                "id": "cash_recount_on_receipt",
                "title": "Пересчет наличных при приеме",
                "description": "Кассир обязан пересчитать принимаемые банкноты.",
                "supported": True,
                "requires_complete_action_sequence": True,
                "visual_conditions": ["Прием наличных без пересчета"],
                "insufficient_evidence": ["Начало или конец приема вне кадра"],
                "exceptions": ["Пересчет выполнен на машинке"],
                "source": {
                    "title": "Правила кассовых операций РК",
                    "paragraph": "п. 12",
                    "url": "https://adilet.zan.kz/rus/docs/V2000020001",
                },
            },
            {
                "id": "cashier_leaves_during_service",
                "title": "Оставление рабочего места при обслуживании",
                "description": "Кассиру запрещено покидать кассу во время обслуживания.",
                "supported": True,
                "requires_complete_action_sequence": False,
                "visual_conditions": ["Кассир уходит от рабочего стола при клиенте"],
                "insufficient_evidence": ["Обслуживание завершено"],
                "exceptions": [],
                "source": {
                    "title": "Правила кассовых операций РК",
                    "paragraph": "п. 24",
                    "url": "https://adilet.zan.kz/rus/docs/V2000020001",
                },
            },
            {
                "id": "collection_bag_seal_check",
                "title": "Проверка пломбы инкассаторской сумки",
                "description": "При приеме сумки кассир проверяет целостность пломбы.",
                "supported": True,
                "requires_complete_action_sequence": True,
                "visual_conditions": ["Прием сумки без осмотра пломбы"],
                "insufficient_evidence": ["Передача сумки не видна"],
                "exceptions": [],
                "source": {
                    "title": "Правила инкассации РК",
                    "paragraph": "п. 8",
                    "url": "https://adilet.zan.kz/rus/docs/V2000020002",
                },
            },
            {
                "id": "camera_capable_device_in_cash_area",
                "title": "Устройство с камерой в кассовом узле",
                "description": "Работнику кассы запрещено использовать устройство с модулем камеры в помещении кассы.",
                "supported": True,
                "requires_complete_action_sequence": False,
                "visual_conditions": [
                    "Видна рабочая роль сотрудника кассы",
                    "Действие внутри кассового узла",
                    "Портативное устройство связи в руках сотрудника",
                    "Различим модуль камеры или процесс съемки",
                ],
                "insufficient_evidence": ["Неясный прямоугольный предмет или сканер", "Устройство у клиента в зале"],
                "exceptions": [],
                "source": {
                    "title": "Постановление Правления НБ РК № 231 (в ред. № 71 от 10.11.2025)",
                    "paragraph": "п. 12",
                    "url": "https://adilet.zan.kz/rus/docs/V2500037429",
                },
            },
        ]
    }
    path.write_text(json.dumps(rules, ensure_ascii=False), encoding="utf-8")
    return path


def _make_config(tmp_path: Path, overview_frames: int = 8) -> AnalysisConfig:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    grounding_dir = tmp_path / "grounding"
    grounding_dir.mkdir(parents=True, exist_ok=True)
    rules_path = _write_rules(tmp_path / "rules.json")
    return AnalysisConfig(
        model_path=model_dir,
        rules_path=rules_path,
        private_dir=tmp_path / "private",
        quantization="nf4",
        scan_fps=2,
        overview_frames=overview_frames,
        overview_pixels=120000,
        detail_pixels=120000,
        detail_frames=6,
        confidence_threshold=0.62,
        timeout_sec=60,
        generation_tokens=600,
        grounding_enabled=False,
        grounding_model_path=grounding_dir,
    )


def test_decode_json_refuses_truncated_outer_object() -> None:
    valid = '```json\n{"confirmed": false, "sequence_visible": true}\n```'
    assert decode_json(valid) == {"confirmed": False, "sequence_visible": True}

    truncated = '{"windows": [{"start_sec": 0.0, "observed": "Кассир принимает деньги"}], "candidates": ['
    with pytest.raises(ValueError, match="incomplete JSON object"):
        decode_json(truncated)


def test_overview_chunks_and_subwindows_cover_timeline() -> None:
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    frames = [VideoFrame(timestamp_sec=float(i), image=img, motion=float(i % 5)) for i in range(32)]
    chunks = overview_chunks(frames)
    assert len(chunks) == 4
    assert all(len(c) <= 9 for c in chunks)
    assert chunks[0][0].timestamp_sec == 0.0
    assert chunks[-1][-1].timestamp_sec == 31.0
    for idx in range(len(chunks) - 1):
        assert chunks[idx][-1].timestamp_sec == chunks[idx + 1][0].timestamp_sec

    sub = chunk_subwindows(chunks[0], duration=31.0)
    assert len(sub) == 3
    assert sub[0]["start_sec"] == 0.0
    assert sub[-1]["end_sec"] == chunks[0][-1].timestamp_sec


def test_window_perception_recovers_candidates_when_candidates_array_is_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config(tmp_path, overview_frames=8)
    video_path = _write_synthetic_video(tmp_path / "cash.mp4", duration_sec=6, fps=2)
    analyzer = VideoAnalyzer(cfg)
    monkeypatch.setattr(analyzer.grounder, "calibrate", lambda video, deadline: ([], {"regions": 0}))

    calls = []

    def fake_generate(frames, prompt, *, max_pixels, max_tokens, deadline):
        calls.append(prompt)
        if len(calls) == 1:
            sub = chunk_subwindows(frames, 6.0)
            payload = {
                "scene_summary": "Кассир принимает деньги у клиента за рабочим столом.",
                "windows": [
                    {
                        "start_sec": sub[0]["start_sec"],
                        "end_sec": sub[0]["end_sec"],
                        "timecode_sec": sub[0]["timestamps"][-1],
                        "observed": "Клиент передает пачку купюр кассиру через окно",
                        "cues": ["cash_receipt"],
                        "rule_ids": [],
                    },
                    {
                        "start_sec": sub[1]["start_sec"],
                        "end_sec": sub[1]["end_sec"],
                        "timecode_sec": sub[1]["timestamps"][0],
                        "observed": "Кассир сидит за столом, передач нет",
                        "cues": ["idle"],
                        "rule_ids": [],
                    },
                ],
                "candidates": [],
                "view_usable": True,
                "timeline_complete": True,
            }
        else:
            stamps = [round(float(f.timestamp_sec), 3) for f in frames]
            payload = {
                "description": "Кассир принимает банкноты от клиента и убирает их в ящик без пересчета.",
                "counterevidence": "нет",
                "uncertainty": "нет",
                "sequence_visible": True,
                "evidence_timestamps": stamps[:2],
                "timecode_sec": stamps[1],
                "rule_id": "cash_recount_on_receipt",
                "confirmed": True,
                "confidence": 0.86,
            }
        return payload, {"seconds": 0.2, "input_tokens": 400, "output_tokens": 60, "images": len(frames), "gpu_peak_mb": 1024.0}, json.dumps(payload)

    monkeypatch.setattr(analyzer.vlm, "generate", fake_generate)
    result = analyzer.analyze(video_path)
    assert result["verdict"] == "violation"
    assert len(result["violations"]) == 1
    v = result["violations"][0]
    assert v["type"] == "Пересчет наличных при приеме"
    assert "https://adilet.zan.kz/rus/docs/V2000020001" in v["description"]
    assert len(v["evidence_frames"]) >= 1
    assert analyzer.last_metrics["error_type"] is None
    assert analyzer.last_metrics["stages"] == 2


def test_negated_idle_windows_produce_no_false_candidates() -> None:
    rules_by_id = {
        "cash_recount_on_receipt": {"id": "cash_recount_on_receipt"},
        "cashier_leaves_during_service": {"id": "cashier_leaves_during_service"},
        "collection_bag_seal_check": {"id": "collection_bag_seal_check"},
    }
    windows = [
        {
            "start_sec": 0.0,
            "end_sec": 4.0,
            "timecode_sec": 2.0,
            "observed": "Кассир сидит за столом, не встает и не уходит, денег нет, сумок нет",
            "cues": ["idle"],
            "rule_ids": [],
        }
    ]
    derived = extract_window_candidates(windows, rules_by_id, duration=10.0)
    assert derived == []


def test_empty_windows_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config(tmp_path, overview_frames=6)
    video_path = _write_synthetic_video(tmp_path / "bad.mp4", duration_sec=4, fps=2)
    analyzer = VideoAnalyzer(cfg)
    monkeypatch.setattr(analyzer.grounder, "calibrate", lambda video, deadline: ([], {"regions": 0}))

    def fake_generate(frames, prompt, *, max_pixels, max_tokens, deadline):
        payload = {"scene_summary": "Кассир в кассе.", "windows": [], "candidates": [], "view_usable": True, "timeline_complete": True}
        return payload, {"seconds": 0.1, "input_tokens": 200, "output_tokens": 20, "images": len(frames), "gpu_peak_mb": 512.0}, json.dumps(payload)

    monkeypatch.setattr(analyzer.vlm, "generate", fake_generate)
    with pytest.raises(ValueError, match="Malformed candidate-scan windows"):
        analyzer.analyze(video_path)
    assert analyzer.last_metrics["error_type"] == "ValueError"


def test_multi_chunk_multi_event_retrieval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config(tmp_path, overview_frames=16)
    video_path = _write_synthetic_video(tmp_path / "multi.mp4", duration_sec=16, fps=2)
    analyzer = VideoAnalyzer(cfg)
    monkeypatch.setattr(analyzer.grounder, "calibrate", lambda video, deadline: ([], {"regions": 0}))

    overview_idx = 0

    def fake_generate(frames, prompt, *, max_pixels, max_tokens, deadline):
        nonlocal overview_idx
        if "candidates" in prompt and "view_usable" in prompt:
            sub = chunk_subwindows(frames, 16.0)
            windows_payload = []
            for s_idx, w in enumerate(sub):
                if overview_idx == 0 and s_idx == 0:
                    windows_payload.append({
                        "start_sec": w["start_sec"],
                        "end_sec": w["end_sec"],
                        "timecode_sec": w["timestamps"][-1],
                        "observed": "Инкассатор передает инкассаторскую сумку кассиру на стол",
                        "cues": ["bag_handling"],
                        "rule_ids": ["collection_bag_seal_check"],
                    })
                elif overview_idx == 1 and s_idx == len(sub) - 1:
                    windows_payload.append({
                        "start_sec": w["start_sec"],
                        "end_sec": w["end_sec"],
                        "timecode_sec": w["timestamps"][-1],
                        "observed": "Кассир встал из-за стола и вышел из помещения",
                        "cues": ["cashier_leaves"],
                        "rule_ids": ["cashier_leaves_during_service"],
                    })
                else:
                    windows_payload.append({
                        "start_sec": w["start_sec"],
                        "end_sec": w["end_sec"],
                        "timecode_sec": w["timestamps"][0],
                        "observed": "Кассир сидит за рабочим столом, передач нет",
                        "cues": ["idle"],
                        "rule_ids": [],
                    })
            overview_idx += 1
            payload = {
                "scene_summary": "События в кассовом помещении.",
                "windows": windows_payload,
                "candidates": [],
                "view_usable": True,
                "timeline_complete": True,
            }
        else:
            stamps = [round(float(f.timestamp_sec), 3) for f in frames]
            if '"rule_id": "collection_bag_seal_check"' in prompt:
                payload = {
                    "description": "Кассир принимает инкассаторскую сумку и убирает ее без осмотра пломбы.",
                    "counterevidence": "нет",
                    "uncertainty": "нет",
                    "sequence_visible": True,
                    "evidence_timestamps": stamps[:2],
                    "timecode_sec": stamps[0],
                    "rule_id": "collection_bag_seal_check",
                    "confirmed": True,
                    "confidence": 0.89,
                }
            elif '"rule_id": "cashier_leaves_during_service"' in prompt:
                payload = {
                    "description": "Кассир встает из-за стола и выходит во время обслуживания клиента.",
                    "counterevidence": "нет",
                    "uncertainty": "нет",
                    "sequence_visible": True,
                    "evidence_timestamps": stamps[-2:],
                    "timecode_sec": stamps[-1],
                    "rule_id": "cashier_leaves_during_service",
                    "confirmed": True,
                    "confidence": 0.91,
                }
            else:
                raise AssertionError("Unexpected verification prompt")
        return payload, {"seconds": 0.2, "input_tokens": 500, "output_tokens": 70, "images": len(frames), "gpu_peak_mb": 1200.0}, json.dumps(payload)

    monkeypatch.setattr(analyzer.vlm, "generate", fake_generate)
    result = analyzer.analyze(video_path)
    assert result["verdict"] == "violation"
    assert [v["type"] for v in result["violations"]] == [
        "Проверка пломбы инкассаторской сумки",
        "Оставление рабочего места при обслуживании",
    ]
    assert result["violations"][0]["timecode_sec"] < result["violations"][1]["timecode_sec"]


def test_close_separate_episodes_same_rule_not_collapsed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _make_config(tmp_path, overview_frames=8)
    video_path = _write_synthetic_video(tmp_path / "two_episodes.mp4", duration_sec=6, fps=2)
    analyzer = VideoAnalyzer(cfg)
    monkeypatch.setattr(analyzer.grounder, "calibrate", lambda video, deadline: ([], {"regions": 0}))

    call_idx = 0

    def fake_generate(frames, prompt, *, max_pixels, max_tokens, deadline):
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            sub = chunk_subwindows(frames, 6.0)
            payload = {
                "scene_summary": "Кассир дважды достает смартфон и снимает в кассе.",
                "windows": [
                    {
                        "start_sec": sub[0]["start_sec"],
                        "end_sec": sub[0]["end_sec"],
                        "timecode_sec": sub[0]["timestamps"][0],
                        "observed": "Кассир в кассе держит смартфон камерой вперед и снимает",
                        "cues": ["device_in_cash_area"],
                        "rule_ids": [],
                    },
                    {
                        "start_sec": sub[1]["start_sec"],
                        "end_sec": sub[1]["end_sec"],
                        "timecode_sec": sub[1]["timestamps"][-1],
                        "observed": "Сотрудник кассы снова поднимает телефон с камерой для съемки",
                        "cues": ["device_in_cash_area"],
                        "rule_ids": [],
                    },
                ],
                "candidates": [],
                "view_usable": True,
                "timeline_complete": True,
            }
        else:
            stamps = [round(float(f.timestamp_sec), 3) for f in frames]
            chosen = stamps[0] if call_idx == 2 else stamps[-1]
            payload = {
                "description": "Сотрудник в кассовом узле держит смартфон с видимым модулем камеры и ведет съемку.",
                "counterevidence": "нет",
                "uncertainty": "нет",
                "sequence_visible": True,
                "evidence_timestamps": [chosen],
                "timecode_sec": chosen,
                "rule_id": "camera_capable_device_in_cash_area",
                "confirmed": True,
                "confidence": 0.88,
            }
        return payload, {"seconds": 0.1, "input_tokens": 300, "output_tokens": 50, "images": len(frames), "gpu_peak_mb": 900.0}, json.dumps(payload)

    monkeypatch.setattr(analyzer.vlm, "generate", fake_generate)
    result = analyzer.analyze(video_path)
    assert result["verdict"] == "violation"
    assert len(result["violations"]) == 2
    assert all(v["type"] == "Устройство с камерой в кассовом узле" for v in result["violations"])
    assert "https://adilet.zan.kz/rus/docs/V2500037429" in result["violations"][0]["description"]


def test_strict_subwindow_and_candidate_validation_rejects_malformed_inputs() -> None:
    img = Image.new("RGB", (32, 32))
    chunk = [VideoFrame(float(t), img) for t in range(6)]
    sub = chunk_subwindows(chunk, duration=5.0)
    rules_by_id = {"cash_recount_on_receipt": {"id": "cash_recount_on_receipt"}}

    # 1. Missing second subwindow fails
    with pytest.raises(ValueError, match="subwindow count mismatch"):
        extract_window_candidates(
            [
                {
                    "start_sec": sub[0]["start_sec"],
                    "end_sec": sub[0]["end_sec"],
                    "timecode_sec": sub[0]["timestamps"][0],
                    "observed": "Кассир сидит за столом",
                    "cues": ["idle"],
                    "rule_ids": [],
                }
            ],
            rules_by_id,
            duration=5.0,
            expected_windows=sub,
        )

    # 2. Out-of-bounds or mismatched subwindow bounds fail (no silent clamping)
    with pytest.raises(ValueError, match="Window bounds do not match"):
        extract_window_candidates(
            [
                {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "timecode_sec": 0.0,
                    "observed": "Кассир сидит за столом",
                    "cues": ["idle"],
                    "rule_ids": [],
                },
                {
                    "start_sec": sub[1]["start_sec"],
                    "end_sec": sub[1]["end_sec"],
                    "timecode_sec": sub[1]["timestamps"][0],
                    "observed": "Кассир сидит за столом",
                    "cues": ["idle"],
                    "rule_ids": [],
                },
            ],
            rules_by_id,
            duration=5.0,
            expected_windows=sub,
        )

    # 3. Timecode not matching any supplied frame timestamp fails
    with pytest.raises(ValueError, match="supplied subwindow frame timestamps"):
        extract_window_candidates(
            [
                {
                    "start_sec": sub[0]["start_sec"],
                    "end_sec": sub[0]["end_sec"],
                    "timecode_sec": 0.5,
                    "observed": "Кассир сидит за столом",
                    "cues": ["idle"],
                    "rule_ids": [],
                },
                {
                    "start_sec": sub[1]["start_sec"],
                    "end_sec": sub[1]["end_sec"],
                    "timecode_sec": sub[1]["timestamps"][0],
                    "observed": "Кассир сидит за столом",
                    "cues": ["idle"],
                    "rule_ids": [],
                },
            ],
            rules_by_id,
            duration=5.0,
            expected_windows=sub,
        )

    # 4. Unknown cue, duplicate cue, non-string rule_id, or boolean timestamp fails
    for bad_cues, bad_rids, bad_ts in [
        (["unknown_cue"], [], 0.0),
        (["idle", "idle"], [], 0.0),
        (["idle", "cash_receipt"], [], 0.0),
        (["cash_receipt"], [123], 0.0),
        (["cash_receipt"], ["unknown_rule"], 0.0),
        (["idle"], [], True),
    ]:
        with pytest.raises(ValueError):
            extract_window_candidates(
                [
                    {
                        "start_sec": sub[0]["start_sec"],
                        "end_sec": sub[0]["end_sec"],
                        "timecode_sec": bad_ts,
                        "observed": "Кассир сидит за столом",
                        "cues": bad_cues,
                        "rule_ids": bad_rids,
                    },
                    {
                        "start_sec": sub[1]["start_sec"],
                        "end_sec": sub[1]["end_sec"],
                        "timecode_sec": sub[1]["timestamps"][0],
                        "observed": "Кассир сидит за столом",
                        "cues": ["idle"],
                        "rule_ids": [],
                    },
                ],
                rules_by_id,
                duration=5.0,
                expected_windows=sub,
            )

    # 5. Out-of-chunk candidate timestamp fails in validate_overview_payload
    valid_wins = [
        {
            "start_sec": w["start_sec"],
            "end_sec": w["end_sec"],
            "timecode_sec": w["timestamps"][0],
            "observed": "Кассир сидит за столом",
            "cues": ["idle"],
            "rule_ids": [],
        }
        for w in sub
    ]
    with pytest.raises(ValueError, match="outside the chunk"):
        validate_overview_payload(
            {
                "view_usable": True,
                "timeline_complete": True,
                "windows": valid_wins,
                "candidates": [{"rule_id": "cash_recount_on_receipt", "timecode_sec": 4.9, "confidence": 0.8}],
            },
            sub,
            chunk,
            rules_by_id,
            duration=10.0,
        )


def test_validate_confirmation_and_prompt_helpers() -> None:
    ok = validate_confirmation(
        {
            "confirmed": True,
            "sequence_visible": True,
            "rule_id": "cash_recount_on_receipt",
            "confidence": 0.8,
            "timecode_sec": 2.0,
            "description": "Кассир принимает наличные без пересчета согласно правилам https://example.com",
            "evidence_timestamps": [2.0],
        },
        "cash_recount_on_receipt",
        0.0,
        5.0,
    )
    assert ok is not None
    assert "https://" not in ok["description"]
    assert "согласно" not in ok["description"].lower()
    assert clean_observation("Действие кассира. п. 14 правил") == "Действие кассира"

    p_ov = overview_prompt([{"id": "cash_recount_on_receipt", "title": "Тест"}], 10.0, 4)
    assert "windows" in p_ov
    p_ver = verification_prompt({"id": "cash_recount_on_receipt", "title": "Тест"}, {"rule_id": "cash_recount_on_receipt"}, 10.0, [1.0, 2.0])
    assert p_ver.index('"description"') < p_ver.index('"confirmed"')


def test_overview_prompt_subwindow_field_alignment_and_probe_failure_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.probe_local_perception import run_synthetic_probe

    img = Image.new("RGB", (32, 32))
    chunk = [VideoFrame(float(t), img) for t in range(4)]
    sub = chunk_subwindows(chunk, duration=3.0)
    prompt = overview_prompt([{"id": "cash_recount_on_receipt", "title": "Тест"}], 3.0, len(chunk), windows=sub)
    assert "default_timecode_sec" not in prompt
    assert '"timecode_sec":' in prompt
    example_line = prompt.strip().splitlines()[-1]
    example_obj = decode_json(example_line)
    rules_by_id = {"cash_recount_on_receipt": {"id": "cash_recount_on_receipt"}}
    assert extract_window_candidates(example_obj["windows"], rules_by_id, duration=3.0, expected_windows=sub) == []

    # Verify probe persists raw response and stats to private_dir even when validation raises.
    cfg = _make_config(tmp_path, overview_frames=8)

    def broken_generate(self, frames, prompt_text, *, max_pixels, max_tokens, deadline):
        bad_payload = {
            "scene_summary": "Кассир за столом.",
            "windows": [{"start_sec": 0.0, "end_sec": 2.0, "observed": "Сидит за столом", "cues": ["idle"], "rule_ids": []}],
            "candidates": [],
            "view_usable": True,
            "timeline_complete": True,
        }
        stats = {"seconds": 0.15, "input_tokens": 310, "output_tokens": 42, "images": len(frames), "gpu_peak_mb": 750.0}
        return bad_payload, stats, json.dumps(bad_payload, ensure_ascii=False)

    monkeypatch.setattr("forte.analysis.vlm.LocalVLM.generate", broken_generate)
    with pytest.raises(ValueError):
        run_synthetic_probe(cfg, run_full_pipeline=False)

    probe_log = cfg.private_dir / "synthetic_probe_private.json"
    assert probe_log.is_file()
    saved = json.loads(probe_log.read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert saved[0]["stats"]["input_tokens"] == 310
    assert saved[0]["validation_error_type"] == "ValueError"
    assert not (cfg.private_dir / "probe_tmp").exists()


def test_bounded_format_retry_and_private_failed_decode_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_config(tmp_path, overview_frames=8)
    video_path = _write_synthetic_video(tmp_path / "retry.mp4", duration_sec=6, fps=2)
    analyzer = VideoAnalyzer(cfg)
    monkeypatch.setattr(analyzer.grounder, "calibrate", lambda video, deadline: ([], {"regions": 0}))

    prompts_seen: list[str] = []

    def retry_then_succeed(frames, prompt, *, max_pixels, max_tokens, deadline):
        prompts_seen.append(prompt)
        sub = chunk_subwindows(frames, 6.0)
        call_no = len(prompts_seen)
        if call_no == 1:
            # Overview attempt 1 fails with contradictory idle + active cue
            bad_ov = {
                "scene_summary": "Кассир принимает деньги.",
                "windows": [
                    {
                        "start_sec": sub[0]["start_sec"],
                        "end_sec": sub[0]["end_sec"],
                        "timecode_sec": sub[0]["timestamps"][0],
                        "observed": "Кассир принимает купюры у клиента",
                        "cues": ["idle", "cash_receipt"],
                        "rule_ids": [],
                    },
                    {
                        "start_sec": sub[1]["start_sec"],
                        "end_sec": sub[1]["end_sec"],
                        "timecode_sec": sub[1]["timestamps"][0],
                        "observed": "Кассир сидит за столом",
                        "cues": ["idle"],
                        "rule_ids": [],
                    },
                ],
                "candidates": [],
                "view_usable": True,
                "timeline_complete": True,
            }
            return bad_ov, {"seconds": 0.1, "input_tokens": 200, "output_tokens": 40, "images": len(frames), "gpu_peak_mb": 600.0}, json.dumps(bad_ov)
        if call_no == 2:
            # Overview attempt 2 receives format-correction prompt and succeeds
            assert "FORMAT CORRECTION REQUIRED" in prompt
            assert "contradictory_idle_cues" in prompt
            good_ov = {
                "scene_summary": "Кассир принимает деньги.",
                "windows": [
                    {
                        "start_sec": sub[0]["start_sec"],
                        "end_sec": sub[0]["end_sec"],
                        "timecode_sec": sub[0]["timestamps"][0],
                        "observed": "Кассир принимает купюры у клиента",
                        "cues": ["cash_receipt"],
                        "rule_ids": ["cash_recount_on_receipt"],
                    },
                    {
                        "start_sec": sub[1]["start_sec"],
                        "end_sec": sub[1]["end_sec"],
                        "timecode_sec": sub[1]["timestamps"][0],
                        "observed": "Кассир сидит за столом",
                        "cues": ["idle"],
                        "rule_ids": [],
                    },
                ],
                "candidates": [],
                "view_usable": True,
                "timeline_complete": True,
            }
            return good_ov, {"seconds": 0.1, "input_tokens": 220, "output_tokens": 42, "images": len(frames), "gpu_peak_mb": 620.0}, json.dumps(good_ov)
        if call_no == 3:
            # Verification attempt 1 fails inside decode_json with attached stats/raw
            try:
                decode_json('{"description": "Private unfinished model narrative')
            except ValueError as exc:
                exc.generation_stats = {
                    "seconds": 0.2,
                    "input_tokens": 450,
                    "output_tokens": 700,
                    "max_tokens": 700,
                    "token_limit_reached": True,
                    "images": len(frames),
                    "gpu_peak_mb": 980.0,
                }
                exc.raw_response = '{"description": "Private unfinished model narrative'
                raise
        # Verification attempt 2 receives format-correction prompt and returns a valid rejection (confirmed=False)
        assert "FORMAT CORRECTION REQUIRED" in prompt
        assert "incomplete_json" in prompt
        stamps = [round(float(f.timestamp_sec), 3) for f in frames]
        valid_reject = {
            "description": "Кассир принимает купюры и сразу пересчитывает их на машинке.",
            "counterevidence": "Виден пересчет на счетной машинке",
            "uncertainty": "нет",
            "sequence_visible": True,
            "evidence_timestamps": [stamps[0]],
            "timecode_sec": stamps[0],
            "rule_id": "cash_recount_on_receipt",
            "confirmed": False,
            "confidence": 0.15,
        }
        return valid_reject, {"seconds": 0.15, "input_tokens": 470, "output_tokens": 65, "images": len(frames), "gpu_peak_mb": 950.0}, json.dumps(valid_reject)

    monkeypatch.setattr(analyzer.vlm, "generate", retry_then_succeed)
    result = analyzer.analyze(video_path)
    # Valid rejection (confirmed=False) is NOT retried a second time
    assert len(prompts_seen) == 4
    assert result == {"verdict": "no_violation", "violations": []}
    assert analyzer.last_metrics["retries"] == 2
    assert analyzer.last_metrics["stages"] == 4
    assert analyzer.last_metrics["output_tokens"] == 40 + 42 + 700 + 65

    # Verify repeated decode failure fails after max 2 attempts and persists raw only in private report
    fail_calls = 0

    def always_bad_decode(frames, prompt, *, max_pixels, max_tokens, deadline):
        nonlocal fail_calls
        fail_calls += 1
        try:
            decode_json('{"scene_summary": "Unclosed secret text')
        except ValueError as exc:
            exc.generation_stats = {"seconds": 0.1, "input_tokens": 300, "output_tokens": 650, "max_tokens": 650, "token_limit_reached": True, "images": len(frames), "gpu_peak_mb": 800.0}
            exc.raw_response = '{"scene_summary": "Unclosed secret text'
            raise

    before_reports = set(cfg.private_dir.glob("*.json"))
    monkeypatch.setattr(analyzer.vlm, "generate", always_bad_decode)
    with pytest.raises(ValueError, match="incomplete JSON object") as exc_info:
        analyzer.analyze(video_path)
    assert "Unclosed secret text" not in str(exc_info.value)
    assert fail_calls == 2
    assert analyzer.last_metrics["retries"] == 1
    assert analyzer.last_metrics["error_stage"] == "overview_decode"
    assert analyzer.last_metrics["error_reason"] == "incomplete_json"
    assert analyzer.last_metrics["error_function"] == "decode_json"
    assert analyzer.last_metrics["error_line"] > 0
    assert analyzer.last_metrics["raw_available"] is True
    assert analyzer.last_metrics["output_tokens"] == 1300

    new_reports = [p for p in cfg.private_dir.glob("*.json") if p not in before_reports]
    assert len(new_reports) == 1
    rep = json.loads(new_reports[0].read_text(encoding="utf-8"))
    assert len(rep["stages"]) == 2
    assert rep["stages"][0]["status"] == "failed_decode"
    assert rep["stages"][0]["raw"] == '{"scene_summary": "Unclosed secret text'
    assert rep["stages"][0]["stats"]["token_limit_reached"] is True


def test_recovered_retry_followed_by_timeout_and_closed_error_reason_allowlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _make_config(tmp_path, overview_frames=8)
    video_path = _write_synthetic_video(tmp_path / "timeout_after_retry.mp4", duration_sec=6, fps=2)
    analyzer = VideoAnalyzer(cfg)
    monkeypatch.setattr(analyzer.grounder, "calibrate", lambda video, deadline: ([], {"regions": 0}))

    calls = 0

    def decode_fail_then_recover_then_timeout(frames, prompt, *, max_pixels, max_tokens, deadline):
        nonlocal calls
        calls += 1
        sub = chunk_subwindows(frames, 6.0)
        if calls == 1:
            try:
                decode_json('{"scene_summary": "broken')
            except ValueError as exc:
                exc.generation_stats = {"seconds": 0.1, "input_tokens": 200, "output_tokens": 50, "images": len(frames), "gpu_peak_mb": 500.0}
                exc.raw_response = '{"scene_summary": "broken'
                raise
        if calls == 2:
            good_ov = {
                "scene_summary": "Кассир принимает купюры.",
                "windows": [
                    {
                        "start_sec": sub[0]["start_sec"],
                        "end_sec": sub[0]["end_sec"],
                        "timecode_sec": sub[0]["timestamps"][0],
                        "observed": "Кассир принимает купюры у клиента",
                        "cues": ["cash_receipt"],
                        "rule_ids": ["cash_recount_on_receipt"],
                    },
                    {
                        "start_sec": sub[1]["start_sec"],
                        "end_sec": sub[1]["end_sec"],
                        "timecode_sec": sub[1]["timestamps"][0],
                        "observed": "Кассир сидит за столом",
                        "cues": ["idle"],
                        "rule_ids": [],
                    },
                ],
                "candidates": [],
                "view_usable": True,
                "timeline_complete": True,
            }
            return good_ov, {"seconds": 0.1, "input_tokens": 210, "output_tokens": 45, "images": len(frames), "gpu_peak_mb": 550.0}, json.dumps(good_ov)
        # Call 3 (verification attempt 1): post-generation TimeoutError with stats/raw attached
        timeout_exc = TimeoutError("Analysis deadline reached")
        timeout_exc.generation_stats = {"seconds": 0.3, "input_tokens": 480, "output_tokens": 110, "images": len(frames), "gpu_peak_mb": 920.0}  # type: ignore[attr-defined]
        timeout_exc.raw_response = '{"description": "Private post-generation timeout text"}'  # type: ignore[attr-defined]
        timeout_exc.error_reason = "timeout"  # type: ignore[attr-defined]
        raise timeout_exc

    before_reports = set(cfg.private_dir.glob("*.json"))
    monkeypatch.setattr(analyzer.vlm, "generate", decode_fail_then_recover_then_timeout)
    with pytest.raises(TimeoutError):
        analyzer.analyze(video_path)

    # Timeout must not be retried (calls == 3) and must not be mislabeled as the earlier recovered decode error.
    assert calls == 3
    assert analyzer.last_metrics["retries"] == 1
    assert analyzer.last_metrics["error_type"] == "TimeoutError"
    assert analyzer.last_metrics["error_stage"] == "verification"
    assert analyzer.last_metrics["error_reason"] == "timeout"
    assert analyzer.last_metrics["error_function"] == "analyze"
    assert analyzer.last_metrics["error_line"] > 0
    assert analyzer.last_metrics["raw_available"] is True
    assert analyzer.last_metrics["output_tokens"] == 50 + 45 + 110

    new_reports = [p for p in cfg.private_dir.glob("*.json") if p not in before_reports]
    assert len(new_reports) == 1
    rep = json.loads(new_reports[0].read_text(encoding="utf-8"))
    assert len(rep["stages"]) == 3
    assert rep["stages"][-1]["status"] == "failed_timeout"
    assert rep["stages"][-1]["raw"] == '{"description": "Private post-generation timeout text"}'

    # Verify untrusted error_reason cannot escape into public metrics.
    def untrusted_reason_fail(frames, prompt, *, max_pixels, max_tokens, deadline):
        err = ValueError("Custom failure with sensitive details")
        err.error_reason = "UNTRUSTED_PRIVATE_NARRATIVE_LEAK"  # type: ignore[attr-defined]
        raise err

    monkeypatch.setattr(analyzer.vlm, "generate", untrusted_reason_fail)
    with pytest.raises(ValueError):
        analyzer.analyze(video_path)
    assert analyzer.last_metrics["error_reason"] == "validation_value_error"
    assert "UNTRUSTED_PRIVATE_NARRATIVE_LEAK" not in json.dumps(analyzer.last_metrics)
