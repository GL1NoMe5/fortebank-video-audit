"""Public smoke client checks using only mock HTTP and generated tiny images."""

import base64
from copy import deepcopy
import io
import json

import httpx
from PIL import Image
import pytest

from scripts import smoke_public_api as smoke

KEY = "synthetic-only-secret-42"
ORIGIN = "https://smoke.example"
JOB_ID = "a" * 32
PRIVATE_TEXT = "private response description must never be printed"
HEALTH = {"status": "ok", "ready": True, "worker_alive": True, "api_key_configured": True, "queue_depth": 0, "queue_capacity": 16}
CONFIG = {"max_upload_bytes": 256 * 1024 * 1024, "max_upload_mib": 256, "allowed_extensions": [".mov"], "max_queue_size": 16,
          "job_deadline_seconds": 300.0, "auth_header": "X-API-Key", "requires_api_key": True, "video_preview_supported": True}


def done_response(with_evidence=False):
    body = {"job_id": JOB_ID, "status": "done", "verdict": "no_violation", "violations": []}
    if with_evidence:
        image = io.BytesIO()
        Image.new("RGB", (8, 8), (20, 60, 100)).save(image, format="JPEG")
        body.update(verdict="violation", violations=[{
            "type": "synthetic", "timecode_sec": 1.0, "description": PRIVATE_TEXT,
            "evidence_frames": [{"timestamp_sec": 1.0, "image_base64": base64.b64encode(image.getvalue()).decode("ascii")}],
        }])
    return body


def mock_client(*, terminal=None, override=None):
    calls = []
    terminal = terminal if terminal is not None else done_response()

    def handler(request):
        calls.append(request)
        if override:
            response = override(request)
            if response is not None:
                return response
        path = request.url.path
        if path == "/healthz":
            return httpx.Response(200, json=HEALTH)
        if path == "/api/v1/config":
            return httpx.Response(200, json=CONFIG)
        if path in ("/", "/styles.css", "/app.js"):
            return httpx.Response(200, content=b"synthetic public asset")
        if path == "/api/v1/analyze":
            if request.headers.get("X-API-Key") != KEY:
                return httpx.Response(401, json={"error": PRIVATE_TEXT})
            return httpx.Response(202, json={"job_id": JOB_ID})
        if path.startswith("/api/v1/jobs/"):
            if request.headers.get("X-API-Key") != KEY:
                return httpx.Response(401, json={"error": PRIVATE_TEXT})
            if path.endswith(JOB_ID):
                return httpx.Response(200, json=terminal)
        return httpx.Response(404, json={"error": PRIVATE_TEXT})

    return httpx.Client(transport=httpx.MockTransport(handler), trust_env=False, follow_redirects=False), calls


def run(client, **kwargs):
    return smoke.run_smoke(ORIGIN, KEY, client=client, synthetic_factory=lambda: b"synthetic-mov-placeholder", **kwargs)


def test_default_probes_are_read_only_and_do_not_generate_video():
    client, calls = mock_client()
    with client:
        summary = smoke.run_smoke(ORIGIN, KEY, client=client, synthetic_factory=lambda: pytest.fail("No video should be generated"))
    assert summary["ok"] is True
    assert summary["synthetic_submitted"] is summary["inference_done"] is False
    posts = [request for request in calls if request.method == "POST"]
    assert len(posts) == 1 and "X-API-Key" not in posts[0].headers and posts[0].content == b""
    authenticated = [request for request in calls if "X-API-Key" in request.headers]
    assert len(authenticated) == 1 and authenticated[0].method == "GET"
    assert any(request.url.path == "/runtime/secrets/api_key.txt" for request in calls)
    encoded = json.dumps(summary)
    assert KEY not in encoded and PRIVATE_TEXT not in encoded and ORIGIN not in encoded


@pytest.mark.parametrize("url", ["http://public.example", "http://192.168.1.4", "https://user:password@example.org", "https://example.org/?key=secret", "https://example.org/#secret", "https://example.org/base", "file:///tmp/x", "https://example.org\n"])
def test_unsafe_or_ambiguous_origins_rejected_before_requests(url):
    client, calls = mock_client()
    with client:
        summary = smoke.run_smoke(url, KEY, client=client)
    assert summary["ok"] is False
    assert not calls


@pytest.mark.parametrize("url", ["http://localhost:8000", "http://127.0.0.1:8000", "http://[::1]:8000", "https://smoke.example/"])
def test_local_http_and_https_origins_are_supported(url):
    assert smoke.validate_url(url) == url.rstrip("/")


def test_redirect_is_never_followed_with_credential():
    client, calls = mock_client(override=lambda request: httpx.Response(307, headers={"Location": "https://other.example/steal"})
                                if request.headers.get("X-API-Key") else None)
    with client:
        summary = run(client)
    assert summary["error_code"] == "redirect_rejected"
    assert all(request.url.host == "smoke.example" for request in calls)


@pytest.mark.parametrize("mutation", ["extra_secret_field", "wrong_extension", "auth_optional", "wrong_limit", "long_deadline"])
def test_public_config_must_match_contract(mutation):
    config = deepcopy(CONFIG)
    if mutation == "extra_secret_field":
        config["api_key"] = "unrecognized-secret"
    elif mutation == "wrong_extension":
        config["allowed_extensions"] = [".mp4"]
    elif mutation == "auth_optional":
        config["requires_api_key"] = False
    elif mutation == "wrong_limit":
        config["max_upload_mib"] = 1
    else:
        config["job_deadline_seconds"] = 400
    client, _ = mock_client(override=lambda request: httpx.Response(200, json=config) if request.url.path == "/api/v1/config" else None)
    with client:
        summary = run(client)
    assert summary["ok"] is False and summary["checks"]["public_config"] is False


def test_not_ready_stops_before_any_post():
    health = {**HEALTH, "ready": False, "status": "degraded"}
    client, calls = mock_client(override=lambda request: httpx.Response(200, json=health) if request.url.path == "/healthz" else None)
    with client:
        summary = run(client, submit_synthetic=True)
    assert summary["ok"] is False
    assert all(request.method == "GET" for request in calls)


def test_optional_synthetic_submission_validates_jpeg_without_exposing_content():
    client, calls = mock_client(terminal=done_response(with_evidence=True))
    with client:
        summary = run(client, submit_synthetic=True, require_done=True)
    assert summary["ok"] is summary["inference_done"] is summary["job_terminal"] is True
    assert summary["synthetic_failed"] is False and summary["evidence_frames_checked"] == 1
    uploads = [request for request in calls if request.method == "POST" and request.headers.get("X-API-Key")]
    assert len(uploads) == 1
    assert b"synthetic-geometric-smoke.mov" in uploads[0].content
    assert b"synthetic-mov-placeholder" in uploads[0].content
    assert PRIVATE_TEXT not in json.dumps(summary) and KEY not in json.dumps(summary)


@pytest.mark.parametrize("require_done", [False, True])
def test_failed_synthetic_job_is_transport_only(require_done):
    terminal = {"job_id": JOB_ID, "status": "failed", "error_code": "analyzer_error", "error": PRIVATE_TEXT}
    client, _ = mock_client(terminal=terminal)
    with client:
        summary = run(client, submit_synthetic=True, require_done=require_done)
    assert summary["ok"] is (not require_done)
    assert summary["job_terminal"] is summary["synthetic_failed"] is True
    assert summary["inference_done"] is False
    assert PRIVATE_TEXT not in json.dumps(summary)


@pytest.mark.parametrize("mutation", ["extra_done_field", "missing_evidence", "nonjpeg", "bad_timestamp", "huge_timestamp", "wrong_job", "inconsistent_verdict"])
def test_invalid_done_response_never_reports_inference_success(mutation):
    body = done_response(with_evidence=True)
    if mutation == "extra_done_field":
        body["private_path"] = PRIVATE_TEXT
    elif mutation == "missing_evidence":
        body["violations"][0]["evidence_frames"] = []
    elif mutation == "nonjpeg":
        body["violations"][0]["evidence_frames"][0]["image_base64"] = base64.b64encode(b"not a JPEG").decode("ascii")
    elif mutation == "bad_timestamp":
        body["violations"][0]["timecode_sec"] = -1
    elif mutation == "huge_timestamp":
        body["violations"][0]["timecode_sec"] = 10 ** 400
    elif mutation == "wrong_job":
        body["job_id"] = "b" * 32
    else:
        body["verdict"] = "no_violation"
    client, _ = mock_client(terminal=body)
    with client:
        summary = run(client, submit_synthetic=True)
    assert summary["ok"] is summary["inference_done"] is False
    assert PRIVATE_TEXT not in json.dumps(summary)


def test_polling_has_bounded_deadline_without_real_sleep():
    current = [0.0]
    terminal = {"job_id": JOB_ID, "status": "processing", "stage": PRIVATE_TEXT, "progress": 0.1}
    client, calls = mock_client(terminal=terminal)
    with client:
        summary = run(client, submit_synthetic=True, timeout_sec=0.2, poll_interval=0.1,
                      monotonic=lambda: current[0], sleep=lambda duration: current.__setitem__(0, current[0] + duration))
    assert summary["ok"] is False and summary["error_code"] == "terminal_before_deadline"
    assert current[0] == 0.2
    assert len([request for request in calls if request.url.path.endswith(JOB_ID)]) == 2


def test_terminal_response_after_deadline_is_not_success():
    current = [0.0]

    def delayed_terminal(request):
        if request.url.path.endswith(JOB_ID):
            current[0] = 2.0
            return httpx.Response(200, json=done_response())
        return None

    client, _ = mock_client(override=delayed_terminal)
    with client:
        summary = run(client, submit_synthetic=True, timeout_sec=1.0, monotonic=lambda: current[0])
    assert summary["ok"] is summary["inference_done"] is False
    assert summary["error_code"] == "response_within_deadline"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 10 ** 400, True, "2.0"])
def test_invalid_numeric_values_do_not_pass_schema_checks(value):
    assert smoke._number(value) is False


def test_server_credential_echo_and_exception_details_are_not_exposed():
    client, _ = mock_client(override=lambda request: httpx.Response(500, text=KEY))
    with client:
        summary = run(client)
    assert summary["error_code"] == "credential_exposed_by_endpoint" and KEY not in json.dumps(summary)

    def fail(request):
        raise httpx.ConnectError(KEY + PRIVATE_TEXT, request=request)

    client, _ = mock_client(override=fail)
    with client:
        summary = run(client)
    assert summary["error_class"] == "ConnectError"
    assert KEY not in json.dumps(summary) and PRIVATE_TEXT not in json.dumps(summary)


def test_main_uses_safe_client_options_and_emits_only_summary(monkeypatch, capsys):
    client, _ = mock_client()
    options = []
    monkeypatch.setenv("FORTE_API_KEY", KEY)
    monkeypatch.setattr(smoke.httpx, "Client", lambda **kwargs: options.append(kwargs) or client)
    assert smoke.main(["--url", ORIGIN]) == 0
    assert options[0]["trust_env"] is False and options[0]["follow_redirects"] is False
    captured = capsys.readouterr()
    assert json.loads(captured.out)["ok"] is True
    assert KEY not in captured.out + captured.err and PRIVATE_TEXT not in captured.out + captured.err


def test_argument_errors_do_not_echo_accidental_cli_secrets(capsys):
    assert smoke.main(["--url", ORIGIN, "--key", KEY]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error_code"] == "invalid_arguments"
    assert KEY not in captured.out + captured.err


def test_key_file_and_environment_are_only_credential_sources(tmp_path, monkeypatch):
    key_file = tmp_path / "synthetic-key.txt"
    key_file.write_text(KEY + "\n", encoding="utf-8")
    monkeypatch.setenv("FORTE_API_KEY", "environment-fallback")
    assert smoke.load_key(key_file) == KEY
    assert smoke.load_key(None) == "environment-fallback"
    monkeypatch.delenv("FORTE_API_KEY")
    with pytest.raises(smoke.SmokeFailure):
        smoke.load_key(None)
