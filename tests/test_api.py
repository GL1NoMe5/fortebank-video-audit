"""Comprehensive test suite for ForteBank video audit FastAPI backend."""

from __future__ import annotations

import sqlite3
import os
import time
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from forte.api import create_app
from forte.jobs import JobManager
from forte.settings import Settings

TEST_API_KEY = "forte-test-secret-key-999"
SAMPLE_MOV_BYTES = b"\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00qt  \x00\x00\x00\x08wide"
SAMPLE_FRAME_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAACAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDhKKKK0Psz/9k="
)


class PicklableSuccessAnalyzer:
    """Top-level picklable fake analyzer for spawned subprocess tests."""

    def __init__(
        self,
        verdict: str = "violation",
        delay_sec: float = 0.05,
        marker_file: str | None = None,
    ) -> None:
        self.verdict = verdict
        self.delay_sec = delay_sec
        self.marker_file = marker_file

    def analyze(
        self,
        video_path: Path,
        progress: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        assert video_path.is_file()
        assert video_path.suffix == ".mov"
        if self.marker_file:
            Path(self.marker_file).write_text(str(video_path.name), encoding="utf-8")
        if progress is not None:
            progress("extracting_frames", 0.25)
        if self.delay_sec > 0:
            time.sleep(self.delay_sec)
        if progress is not None:
            progress("evaluating_rules", 0.75)

        if self.verdict == "no_violation":
            return {
                "verdict": "no_violation",
                "violations": [],
            }

        return {
            "verdict": "violation",
            "violations": [
                {
                    "type": "unattended_cash_drawer",
                    "timecode_sec": 14.5,
                    "description": "Cash drawer left open without teller presence.",
                    "evidence_frames": [
                        {
                            "timestamp_sec": 14.2,
                            "image_base64": SAMPLE_FRAME_B64,
                        },
                        {
                            "timestamp_sec": 14.8,
                            "image_base64": SAMPLE_FRAME_B64,
                        },
                    ],
                }
            ],
        }


class PicklableFailingAnalyzer:
    """Top-level picklable fake analyzer that raises an exception containing a local path."""

    def analyze(
        self,
        video_path: Path,
        progress: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        if progress is not None:
            progress("decoding", 0.1)
        raise RuntimeError(f"CUDA error while reading {video_path} at C:\\Users\\Secret\\model.pt")


class PicklableMalformedAnalyzer:
    """Top-level picklable fake analyzer that returns schema-invalid output."""

    def analyze(
        self,
        video_path: Path,
        progress: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        return {
            "verdict": "violation",
            "violations": [
                {
                    "type": "bad_frames",
                    "timecode_sec": 3.0,
                    "description": "Missing evidence frames must fail strict validation.",
                    "evidence_frames": [],
                }
            ],
        }


class PicklableAdaptiveAnalyzer:
    """Top-level picklable analyzer whose behavior depends on a control file or call order."""

    def __init__(self, hang_flag_path: str, hang_seconds: float = 30.0) -> None:
        self.hang_flag_path = hang_flag_path
        self.hang_seconds = hang_seconds

    def analyze(
        self,
        video_path: Path,
        progress: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        flag = Path(self.hang_flag_path)
        if flag.exists():
            flag.unlink()
            if progress is not None:
                progress("stuck_in_kernel", 0.15)
            time.sleep(self.hang_seconds)
        if progress is not None:
            progress("completed_fast", 0.9)
        return {"verdict": "no_violation", "violations": []}


class PersistentCountingAnalyzer:
    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.calls = 0

    def warmup(self) -> None:
        time.sleep(0.2)
        with Path(self.marker).open('a') as stream:
            stream.write(f'warmup:{os.getpid()}\n')

    def analyze(self, video_path: Path, progress=None) -> dict:
        self.calls += 1
        with Path(self.marker).open('a') as stream:
            stream.write(f'call{self.calls}:{os.getpid()}\n')
        return {'verdict': 'no_violation', 'violations': []}


class InternalTypeErrorAnalyzer:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    def analyze(self, video_path: Path, progress=None) -> dict:
        with Path(self.marker).open('a') as stream:
            stream.write('called\n')
        raise TypeError('Internal analyzer bug, not an argument mismatch.')


class WarmupHangsOnceAnalyzer:
    def __init__(self, flag: str, marker: str) -> None:
        self.flag, self.marker = flag, marker

    def warmup(self) -> None:
        with Path(self.marker).open('a') as stream:
            stream.write(f'{os.getpid()}\n')
        flag = Path(self.flag)
        if flag.exists():
            flag.unlink()
            time.sleep(30)

    def analyze(self, video_path: Path, progress=None) -> dict:
        return {'verdict': 'no_violation', 'violations': []}


def _make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    defaults: dict[str, Any] = {
        "root_dir": tmp_path,
        "api_key": TEST_API_KEY,
        "max_upload_bytes": 1024 * 1024,
        "max_queue_size": 4,
        "job_deadline_seconds": 10.0,
        "worker_poll_interval_seconds": 0.02,
        "subprocess_join_timeout_seconds": 1.0,
        "upload_chunk_size_bytes": 4096,
        "enable_video_transcode": False,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _wait_for_terminal_status(
    client: TestClient,
    job_id: str,
    headers: dict[str, str],
    timeout_sec: float = 8.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_payload: dict[str, Any] = {}
    while time.time() < deadline:
        resp = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        assert resp.status_code == 200
        last_payload = resp.json()
        if last_payload.get("status") in {"done", "failed"}:
            return last_payload
        time.sleep(0.04)
    pytest.fail(f"Job {job_id} did not reach terminal status in {timeout_sec}s; last={last_payload}")


def _wait_ready(client: TestClient, timeout_sec: float = 8.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if client.get('/healthz').json()['ready']:
            return
        time.sleep(0.02)
    pytest.fail('Worker did not finish warmup.')


def test_public_endpoints_do_not_expose_secrets_or_paths(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    app = create_app(settings=settings, analyzer_factory=PicklableSuccessAnalyzer)

    with TestClient(app) as client:
        _wait_ready(client)
        health = client.get("/healthz")
        assert health.status_code == 200
        h_body = health.json()
        assert h_body["status"] == "ok"
        assert h_body["ready"] is True
        assert h_body["worker_alive"] is True
        assert h_body["api_key_configured"] is True
        assert h_body["queue_capacity"] == settings.max_queue_size
        assert TEST_API_KEY not in health.text
        assert str(tmp_path) not in health.text

        cfg = client.get("/api/v1/config")
        assert cfg.status_code == 200
        c_body = cfg.json()
        assert c_body["max_upload_bytes"] == settings.max_upload_bytes
        assert c_body["allowed_extensions"] == [".mov"]
        assert c_body["requires_api_key"] is True
        assert TEST_API_KEY not in cfg.text
        assert str(tmp_path) not in cfg.text


def test_authentication_required_and_missing_server_key_fails_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FORTE_API_KEY", raising=False)
    settings = _make_settings(tmp_path, api_key=TEST_API_KEY)
    app = create_app(settings=settings, analyzer_factory=PicklableSuccessAnalyzer)

    with TestClient(app) as client:
        r_missing = client.post(
            "/api/v1/analyze",
            files={"file": ("audit.mov", SAMPLE_MOV_BYTES, "video/quicktime")},
        )
        assert r_missing.status_code == 401

        r_wrong = client.post(
            "/api/v1/analyze",
            headers={"X-API-Key": "wrong-key"},
            files={"file": ("audit.mov", SAMPLE_MOV_BYTES, "video/quicktime")},
        )
        assert r_wrong.status_code == 401

        r_job_no_auth = client.get("/api/v1/jobs/any-job-id")
        assert r_job_no_auth.status_code == 401

    unconfigured_settings = _make_settings(tmp_path / "no_key", api_key=None)
    unconfigured_app = create_app(
        settings=unconfigured_settings, analyzer_factory=PicklableSuccessAnalyzer
    )
    with TestClient(unconfigured_app) as client:
        r_unconfigured = client.post(
            "/api/v1/analyze",
            headers={"X-API-Key": ""},
            files={"file": ("audit.mov", SAMPLE_MOV_BYTES, "video/quicktime")},
        )
        assert r_unconfigured.status_code == 401
        health = client.get("/healthz").json()
        assert health["api_key_configured"] is False
        assert health["ready"] is False


def test_upload_validation_extension_mime_empty_and_size_limit(tmp_path: Path) -> None:
    max_bytes = 8192
    settings = _make_settings(tmp_path, max_upload_bytes=max_bytes)
    app = create_app(settings=settings, analyzer_factory=PicklableSuccessAnalyzer)
    headers = {"X-API-Key": TEST_API_KEY}

    with TestClient(app) as client:
        r_mp4 = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("clip.mp4", SAMPLE_MOV_BYTES, "video/quicktime")},
        )
        assert r_mp4.status_code == 422

        r_dot_only = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": (".mov", SAMPLE_MOV_BYTES, "video/quicktime")},
        )
        assert r_dot_only.status_code == 422

        r_empty = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("empty.mov", b"", "video/quicktime")},
        )
        assert r_empty.status_code == 422

        r_exe = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("spoofed.mov", b"MZ\x90\x00not-a-video", "video/quicktime")},
        )
        assert r_exe.status_code == 422

        oversized_payload = b"A" * (max_bytes + 100)
        r_large = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("huge.mov", oversized_payload, "video/quicktime")},
        )
        assert r_large.status_code == 413

        assert settings.uploads_dir is not None
        assert settings.temp_dir is not None
        assert list(settings.uploads_dir.iterdir()) == []
        assert list(settings.temp_dir.iterdir()) == []


def test_job_success_progress_and_strict_done_schema(tmp_path: Path) -> None:
    marker = tmp_path / "stored_name.txt"
    settings = _make_settings(tmp_path)
    analyzer = PicklableSuccessAnalyzer(
        verdict="violation",
        delay_sec=0.25,
        marker_file=str(marker),
    )
    app = create_app(settings=settings, analyzer_factory=analyzer)
    headers = {"X-API-Key": TEST_API_KEY}

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("../../untrusted/path/camera 01.MOV", SAMPLE_MOV_BYTES, "video/quicktime")},
        )
        assert resp.status_code == 202
        body = resp.json()
        assert set(body.keys()) == {"job_id"}
        job_id = body["job_id"]

        first_poll = client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        assert first_poll.status_code == 200
        first_json = first_poll.json()
        assert first_json["status"] in {"processing", "done"}
        if first_json["status"] == "processing":
            assert set(first_json.keys()).issubset({"job_id", "status", "stage", "progress"})

        done_json = _wait_for_terminal_status(client, job_id, headers)
        assert done_json["status"] == "done"
        assert set(done_json.keys()) == {"job_id", "status", "verdict", "violations"}
        assert done_json["job_id"] == job_id
        assert done_json["verdict"] == "violation"
        assert len(done_json["violations"]) == 1

        violation = done_json["violations"][0]
        assert set(violation.keys()) == {"type", "timecode_sec", "description", "evidence_frames"}
        assert isinstance(violation["timecode_sec"], (int, float))
        assert 1 <= len(violation["evidence_frames"]) <= 3
        for frame in violation["evidence_frames"]:
            assert set(frame.keys()) == {"timestamp_sec", "image_base64"}
            assert isinstance(frame["timestamp_sec"], (int, float))
            assert isinstance(frame["image_base64"], str) and len(frame["image_base64"]) > 0

        assert marker.read_text(encoding="utf-8") == f"{job_id}.mov"


def test_no_violation_done_schema(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    analyzer = PicklableSuccessAnalyzer(verdict="no_violation", delay_sec=0.02)
    app = create_app(settings=settings, analyzer_factory=analyzer)
    headers = {"X-API-Key": TEST_API_KEY}

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("clean.mov", SAMPLE_MOV_BYTES, "video/quicktime")},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        done_json = _wait_for_terminal_status(client, job_id, headers)
        assert set(done_json.keys()) == {"job_id", "status", "verdict", "violations"}
        assert done_json["status"] == "done"
        assert done_json["verdict"] == "no_violation"
        assert done_json["violations"] == []


def test_analyzer_exception_and_invalid_output_mark_job_failed(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    app_err = create_app(settings=settings, analyzer_factory=PicklableFailingAnalyzer)
    headers = {"X-API-Key": TEST_API_KEY}

    with TestClient(app_err) as client:
        resp = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("crash.mov", SAMPLE_MOV_BYTES, "video/quicktime")},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        failed_json = _wait_for_terminal_status(client, job_id, headers)
        assert failed_json["status"] == "failed"
        assert "verdict" not in failed_json
        assert "violations" not in failed_json
        assert failed_json["error_code"] == "analyzer_error"
        assert "C:\\Users" not in failed_json["error"]
        assert str(tmp_path) not in failed_json["error"]
        assert "CUDA" not in failed_json["error"]

    settings_bad = _make_settings(tmp_path / "malformed")
    app_bad = create_app(settings=settings_bad, analyzer_factory=PicklableMalformedAnalyzer)
    with TestClient(app_bad) as client:
        resp = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("bad_schema.mov", SAMPLE_MOV_BYTES, "video/quicktime")},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        failed_json = _wait_for_terminal_status(client, job_id, headers)
        assert failed_json["status"] == "failed"
        assert failed_json["error_code"] == "invalid_analyzer_output"
        assert "verdict" not in failed_json


def test_restart_recovery_marks_unfinished_jobs_failed(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    manager = JobManager(settings=settings, analyzer_factory=PicklableSuccessAnalyzer)
    assert settings.uploads_dir is not None
    assert settings.db_path is not None

    stale_video = settings.uploads_dir / "stalejob123.mov"
    stale_video.write_bytes(SAMPLE_MOV_BYTES)
    now = time.time()
    conn = sqlite3.connect(str(settings.db_path))
    try:
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, status, stage, progress, created_at, updated_at,
                deadline_at, video_path, verdict, violations_json,
                error_code, error_message, worker_pid
            ) VALUES ('stalejob123', 'processing', 'analyzing', 0.5, ?, ?, ?, ?, NULL, NULL, NULL, NULL, 99999);
            """,
            (now - 20.0, now - 10.0, now + 280.0, str(stale_video)),
        )
        conn.commit()
    finally:
        conn.close()

    app = create_app(settings=settings, analyzer_factory=PicklableSuccessAnalyzer)
    headers = {"X-API-Key": TEST_API_KEY}
    with TestClient(app) as client:
        resp = client.get("/api/v1/jobs/stalejob123", headers=headers)
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "failed"
        assert payload["error_code"] == "service_restart"
        assert "restart" in payload["error"].lower()


def test_queue_capacity_limit_returns_429_and_cleans_rejected_upload(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, max_queue_size=1)
    slow_analyzer = PicklableSuccessAnalyzer(verdict="no_violation", delay_sec=0.55)
    app = create_app(settings=settings, analyzer_factory=slow_analyzer)
    headers = {"X-API-Key": TEST_API_KEY}

    with TestClient(app) as client:
        r1 = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("first.mov", SAMPLE_MOV_BYTES, "video/quicktime")},
        )
        assert r1.status_code == 202
        first_job_id = r1.json()["job_id"]

        r2 = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("second.mov", SAMPLE_MOV_BYTES, "video/quicktime")},
        )
        assert r2.status_code == 429
        assert r2.json()["error_code"] == "queue_full"

        assert settings.uploads_dir is not None
        assert settings.temp_dir is not None
        upload_files = [p.name for p in settings.uploads_dir.iterdir()]
        assert upload_files == [f"{first_job_id}.mov"]
        assert list(settings.temp_dir.iterdir()) == []

        done_first = _wait_for_terminal_status(client, first_job_id, headers)
        assert done_first["status"] == "done"

        r3 = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("third.mov", SAMPLE_MOV_BYTES, "video/quicktime")},
        )
        assert r3.status_code == 202
        done_third = _wait_for_terminal_status(client, r3.json()["job_id"], headers)
        assert done_third["status"] == "done"


def test_deadline_includes_queue_wait_and_hanging_subprocess_recovery(tmp_path: Path) -> None:
    hang_flag = tmp_path / "hang_first.flag"
    hang_flag.write_text("1", encoding="utf-8")

    settings = _make_settings(
        tmp_path,
        max_queue_size=3,
        job_deadline_seconds=0.45,
        worker_poll_interval_seconds=0.02,
        subprocess_join_timeout_seconds=0.5,
    )
    adaptive_analyzer = PicklableAdaptiveAnalyzer(
        hang_flag_path=str(hang_flag),
        hang_seconds=30.0,
    )
    app = create_app(settings=settings, analyzer_factory=adaptive_analyzer)
    headers = {"X-API-Key": TEST_API_KEY}

    with TestClient(app) as client:
        _wait_ready(client)
        r_hang = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("hanging.mov", SAMPLE_MOV_BYTES, "video/quicktime")},
        )
        assert r_hang.status_code == 202
        job_hang = r_hang.json()["job_id"]

        r_queued = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("queued_expired.mov", SAMPLE_MOV_BYTES, "video/quicktime")},
        )
        assert r_queued.status_code == 202
        job_queued = r_queued.json()["job_id"]

        res_hang = _wait_for_terminal_status(client, job_hang, headers, timeout_sec=6.0)
        assert res_hang["status"] == "failed"
        assert res_hang["error_code"] == "job_timeout"

        res_queued = _wait_for_terminal_status(client, job_queued, headers, timeout_sec=6.0)
        assert res_queued["status"] == "failed"
        assert res_queued["error_code"] == "job_timeout"

        _wait_ready(client)
        r_after = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("recover.mov", SAMPLE_MOV_BYTES, "video/quicktime")},
        )
        assert r_after.status_code == 202
        job_after = r_after.json()["job_id"]
        res_after = _wait_for_terminal_status(client, job_after, headers, timeout_sec=6.0)
        assert res_after["status"] == "done"
        assert res_after["verdict"] == "no_violation"


def test_unknown_job_404_and_optional_video_endpoint(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, enable_video_transcode=False)
    app = create_app(settings=settings, analyzer_factory=PicklableSuccessAnalyzer(delay_sec=0.01))
    headers = {"X-API-Key": TEST_API_KEY}

    with TestClient(app) as client:
        r_missing = client.get("/api/v1/jobs/does-not-exist", headers=headers)
        assert r_missing.status_code == 404

        r_video_missing = client.get("/api/v1/jobs/does-not-exist/video", headers=headers)
        assert r_video_missing.status_code == 404

        submit = client.post(
            "/api/v1/analyze",
            headers=headers,
            files={"file": ("clip.mov", SAMPLE_MOV_BYTES, "video/quicktime")},
        )
        job_id = submit.json()["job_id"]
        _wait_for_terminal_status(client, job_id, headers)

        r_video = client.get(f"/api/v1/jobs/{job_id}/video", headers=headers)
        assert r_video.status_code == 503
        assert r_video.json()["error_code"] == "video_preview_unavailable"


def test_one_warmup_and_persistent_model_across_two_jobs(tmp_path: Path) -> None:
    marker = tmp_path / 'model_lifecycle.txt'
    app = create_app(_make_settings(tmp_path), PersistentCountingAnalyzer(str(marker)))
    headers = {'X-API-Key': TEST_API_KEY}
    with TestClient(app) as client:
        health = client.get('/healthz').json()
        assert health['worker_alive'] is True and health['ready'] is False
        _wait_ready(client)
        for _ in range(2):
            accepted = client.post('/api/v1/analyze', headers=headers,
                files={'file': ('clip.mov', SAMPLE_MOV_BYTES, 'video/quicktime')})
            assert _wait_for_terminal_status(client, accepted.json()['job_id'], headers)['status'] == 'done'
    lines = marker.read_text().splitlines()
    assert [line.split(':')[0] for line in lines] == ['warmup', 'call1', 'call2']
    assert len({line.split(':')[1] for line in lines}) == 1


def test_internal_typeerror_never_retries_analyzer(tmp_path: Path) -> None:
    marker = tmp_path / 'calls.txt'
    app = create_app(_make_settings(tmp_path), InternalTypeErrorAnalyzer(str(marker)))
    headers = {'X-API-Key': TEST_API_KEY}
    with TestClient(app) as client:
        accepted = client.post('/api/v1/analyze', headers=headers,
            files={'file': ('clip.mov', SAMPLE_MOV_BYTES, 'video/quicktime')})
        result = _wait_for_terminal_status(client, accepted.json()['job_id'], headers)
        assert result['status'] == 'failed'
    assert marker.read_text().splitlines() == ['called']


def test_public_assets_do_not_expose_local_review(tmp_path: Path) -> None:
    web = tmp_path / 'web'
    web.mkdir()
    (web / 'index.html').write_text('<h1>ForteBank</h1>')
    (web / 'review.html').write_text('PRIVATE REVIEW')
    app = create_app(_make_settings(tmp_path), PicklableSuccessAnalyzer)
    with TestClient(app) as client:
        assert client.get('/').status_code == 200
        assert client.get('/review.html').status_code == 404
        assert client.get('/../Dataset/anything').status_code == 404


def test_body_limit_and_auth_before_multipart_spooling(tmp_path: Path) -> None:
    app = create_app(_make_settings(tmp_path, max_upload_bytes=64), PicklableSuccessAnalyzer)
    with TestClient(app) as client:
        oversized = {'Content-Length': str(2 * 1024 * 1024)}
        assert client.post('/api/v1/analyze', headers=oversized).status_code == 401
        oversized['X-API-Key'] = TEST_API_KEY
        assert client.post('/api/v1/analyze', headers=oversized).status_code == 413
        headers = {'X-API-Key': TEST_API_KEY, 'Content-Type': 'multipart/form-data; boundary=test'}
        chunks = iter([b'--test\r\nContent-Disposition: form-data; name="file"; filename="clip.mov"\r\n\r\n',
            b'0' * (2 * 1024 * 1024), b'\r\n--test--\r\n'])
        response = client.post('/api/v1/analyze', headers=headers, content=chunks)
        assert response.status_code == 413


def test_evidence_requires_valid_pure_base64_jpeg() -> None:
    from forte.schemas import EvidenceFrame
    from pydantic import ValidationError
    assert EvidenceFrame(timestamp_sec=0, image_base64=SAMPLE_FRAME_B64).image_base64 == SAMPLE_FRAME_B64
    for invalid in ['not-base64', 'data:image/jpeg;base64,' + SAMPLE_FRAME_B64,
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=']:
        with pytest.raises(ValidationError):
            EvidenceFrame(timestamp_sec=0, image_base64=invalid)


def test_authenticated_h264_mov_preview_remux(tmp_path: Path) -> None:
    import av
    import numpy as np
    from io import BytesIO
    source = tmp_path / 'synthetic.mov'
    with av.open(str(source), 'w', format='mov') as container:
        stream = container.add_stream('libx264', rate=2)
        stream.width = stream.height = 16
        stream.pix_fmt = 'yuv420p'
        for _ in range(3):
            frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    app = create_app(_make_settings(tmp_path, enable_video_transcode=True), PicklableSuccessAnalyzer)
    headers = {'X-API-Key': TEST_API_KEY}
    with TestClient(app) as client:
        accepted = client.post('/api/v1/analyze', headers=headers,
            files={'file': ('synthetic.mov', source.read_bytes(), 'video/quicktime')})
        job_id = accepted.json()['job_id']
        assert client.get(f'/api/v1/jobs/{job_id}/video').status_code == 401
        response = client.get(f'/api/v1/jobs/{job_id}/video', headers=headers)
        assert response.status_code == 200
        assert response.headers['content-type'] == 'video/mp4'
        assert response.content.index(b'moov') < response.content.index(b'mdat')
        with av.open(BytesIO(response.content)) as preview:
            assert preview.streams.video[0].codec_context.name == 'h264'
            assert preview.streams.video[0].frames == 3


def test_stop_during_spawn_blocks_restart_until_old_supervisor_exits(tmp_path: Path) -> None:
    import multiprocessing
    import threading
    entered, release = threading.Event(), threading.Event()

    class DelayedProcess:
        pid, exitcode, alive = 999001, None, False

        def start(self):
            entered.set()
            assert release.wait(5)
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.alive, self.exitcode = False, -15

        def kill(self):
            self.terminate()

        def join(self, timeout=None):
            return None

    process = DelayedProcess()

    class DelayedContext:
        def Pipe(self, **kwargs):
            return multiprocessing.get_context('spawn').Pipe(**kwargs)

        def Process(self, **kwargs):
            return process

    manager = JobManager(_make_settings(tmp_path, subprocess_join_timeout_seconds=0.05), PicklableSuccessAnalyzer)
    manager._mp_ctx = DelayedContext()
    try:
        manager.start()
        assert entered.wait(2)
        old_thread = manager._worker_thread
        manager.stop()
        assert old_thread is not None and old_thread.is_alive()
        assert manager._worker_thread is old_thread
        with pytest.raises(RuntimeError, match='still shutting down'):
            manager.start()
        release.set()
        old_thread.join(2)
        assert not old_thread.is_alive()
        assert not process.is_alive()
    finally:
        release.set()
        manager.stop()


def test_idle_warmup_hang_is_replaced_without_upload(tmp_path: Path) -> None:
    import multiprocessing
    flag, marker = tmp_path / 'hang-once', tmp_path / 'warmup-pids'
    flag.write_text('synthetic', encoding='utf-8')
    app = create_app(_make_settings(tmp_path, worker_start_timeout_seconds=1.5),
        WarmupHangsOnceAnalyzer(str(flag), str(marker)))
    with TestClient(app) as client:
        assert client.get('/healthz').json()['ready'] is False
        _wait_ready(client, timeout_sec=9)
        pids = [int(line) for line in marker.read_text().splitlines()]
        assert len(pids) == 2 and pids[0] != pids[1]
        assert pids[0] not in [process.pid for process in multiprocessing.active_children()]
        assert client.get('/healthz').json()['queue_depth'] == 0


def test_result_validation_time_is_inside_job_deadline(tmp_path: Path, monkeypatch) -> None:
    from forte.schemas import AnalyzerOutput
    manager = JobManager(_make_settings(tmp_path, job_deadline_seconds=.05), PicklableSuccessAnalyzer)
    source = tmp_path / 'synthetic.mov'
    source.write_bytes(SAMPLE_MOV_BYTES)
    manager.enqueue_job('late-validation', source)
    validate = AnalyzerOutput.model_validate

    def slow_validation(payload):
        time.sleep(.08)
        return validate(payload)

    monkeypatch.setattr(AnalyzerOutput, 'model_validate', slow_validation)
    manager._finalize_job_outcome('late-validation', {'kind': 'result',
        'payload': {'verdict': 'no_violation', 'violations': []}}, None)
    record = manager.get_job_record('late-validation')
    assert record.status == 'failed' and record.error_code == 'job_timeout'


def test_terminal_failure_cannot_be_replaced_by_late_success(tmp_path: Path) -> None:
    manager = JobManager(_make_settings(tmp_path), PicklableSuccessAnalyzer)
    source = tmp_path / 'synthetic.mov'
    source.write_bytes(SAMPLE_MOV_BYTES)
    manager.enqueue_job('already-failed', source)
    manager._mark_failed('already-failed', 'job_timeout', 'Synthetic timeout.')
    manager._finalize_job_outcome('already-failed', {'kind': 'result',
        'payload': {'verdict': 'no_violation', 'violations': []}}, None)
    record = manager.get_job_record('already-failed')
    assert record.status == 'failed' and record.error_code == 'job_timeout'


@pytest.mark.parametrize('value', [float('inf'), float('nan'), 999.0])
def test_configuration_cannot_disable_five_minute_job_bound(tmp_path: Path, value: float) -> None:
    settings = _make_settings(tmp_path, job_deadline_seconds=value, worker_start_timeout_seconds=value)
    assert settings.job_deadline_seconds == 300
    assert settings.worker_start_timeout_seconds == 300
    env_settings = Settings.from_env({'FORTE_ROOT_DIR': str(tmp_path),
        'FORTE_JOB_DEADLINE_SECONDS': str(value), 'FORTE_WORKER_START_TIMEOUT_SECONDS': str(value)})
    assert env_settings.job_deadline_seconds == 300
    assert env_settings.worker_start_timeout_seconds == 300
