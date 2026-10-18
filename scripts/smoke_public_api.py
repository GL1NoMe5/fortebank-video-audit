"""Check an explicitly supplied API origin without exposing response bodies.

Default probes do not create jobs. Credentials come only from --key-file or
FORTE_API_KEY. --submit-synthetic explicitly submits an in-memory geometric MOV;
a failed synthetic scene proves transport/terminal handling only, not inference
success. Add --require-done to require a valid completed inference response.
No endpoint discovery, source-video input, model loading, or browser is used.
"""

from __future__ import annotations

import argparse
import base64
import io
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

import httpx


class SmokeFailure(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's normal error includes supplied arguments, potentially a key.
        raise SmokeFailure("invalid_arguments")


def validate_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
        host = parsed.hostname
    except ValueError:
        raise SmokeFailure("invalid_url") from None
    if (not host or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path not in ("", "/")
            or any(ord(char) <= 32 or ord(char) >= 127 for char in value)
            or "\\" in value or (port is not None and not 1 <= port <= 65535)):
        raise SmokeFailure("invalid_url")
    local = host.lower() == "localhost"
    try:
        local = local or ipaddress.ip_address(host).is_loopback
    except ValueError:
        pass
    if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
        raise SmokeFailure("https_required")
    return value.rstrip("/")


def load_key(key_file: Path | None) -> str:
    key = key_file.read_text(encoding="utf-8").strip() if key_file else os.environ.get("FORTE_API_KEY", "").strip()
    if not key or len(key) > 4096 or any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise SmokeFailure("missing_or_invalid_key")
    return key


def make_synthetic_mov() -> bytes:
    """Generate six seconds of moving geometric shapes entirely in memory."""
    import av
    import numpy as np

    output = io.BytesIO()
    with av.open(output, "w", format="mov") as container:
        stream = container.add_stream("libx264", rate=4)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for index in range(24):
            pixels = np.full((240, 320, 3), (24, 32, 48), dtype=np.uint8)
            left = 10 + index * 9
            pixels[80:140, left:left + 50] = (40, 180, 220)
            pixels[175:195, 20:300] = (240, 180, 60)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


def _number(value, *, minimum=0, maximum=None):
    if type(value) not in (int, float) or value < minimum or (maximum is not None and value > maximum):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _valid_health(body):
    return (isinstance(body, dict)
            and set(body) == {"status", "ready", "worker_alive", "api_key_configured", "queue_depth", "queue_capacity"}
            and body["status"] == "ok"
            and all(body[name] is True for name in ("ready", "worker_alive", "api_key_configured"))
            and type(body["queue_depth"]) is int and type(body["queue_capacity"]) is int
            and 0 <= body["queue_depth"] <= body["queue_capacity"] and body["queue_capacity"] > 0)


def _valid_config(body):
    fields = {"max_upload_bytes", "max_upload_mib", "allowed_extensions", "max_queue_size", "job_deadline_seconds", "auth_header", "requires_api_key", "video_preview_supported"}
    return (isinstance(body, dict) and set(body) == fields
            and type(body["max_upload_bytes"]) is int and body["max_upload_bytes"] > 0
            and type(body["max_upload_mib"]) is int and body["max_upload_mib"] == body["max_upload_bytes"] // (1024 * 1024)
            and type(body["max_queue_size"]) is int and body["max_queue_size"] > 0
            and _number(body["job_deadline_seconds"], minimum=0.001, maximum=300)
            and body["allowed_extensions"] == [".mov"] and body["auth_header"] == "X-API-Key"
            and body["requires_api_key"] is True and type(body["video_preview_supported"]) is bool)


def validate_done(body: dict, job_id: str) -> int:
    """Validate exact public keys and JPEG evidence; return only an image count."""
    from PIL import Image

    if (not isinstance(body, dict) or set(body) != {"job_id", "status", "verdict", "violations"}
            or body["job_id"] != job_id or body["status"] != "done"
            or body["verdict"] not in ("violation", "no_violation") or not isinstance(body["violations"], list)
            or (body["verdict"] == "violation") != bool(body["violations"])):
        raise SmokeFailure("invalid_done_schema")
    count = 0
    for item in body["violations"]:
        if (not isinstance(item, dict) or set(item) != {"type", "timecode_sec", "description", "evidence_frames"}
                or not all(isinstance(item[field], str) and item[field].strip() for field in ("type", "description"))
                or not _number(item["timecode_sec"]) or not isinstance(item["evidence_frames"], list)
                or not 1 <= len(item["evidence_frames"]) <= 3):
            raise SmokeFailure("invalid_done_schema")
        for frame in item["evidence_frames"]:
            if (not isinstance(frame, dict) or set(frame) != {"timestamp_sec", "image_base64"}
                    or not _number(frame["timestamp_sec"]) or not isinstance(frame["image_base64"], str)):
                raise SmokeFailure("invalid_evidence")
            try:
                decoded = base64.b64decode(frame["image_base64"], validate=True)
                if not decoded.startswith(b"\xff\xd8\xff"):
                    raise ValueError()
                with Image.open(io.BytesIO(decoded)) as image:
                    if image.format != "JPEG":
                        raise ValueError()
                    image.verify()
            except Exception:
                raise SmokeFailure("invalid_evidence") from None
            count += 1
    return count


def run_smoke(url: str, key: str, *, client: httpx.Client, submit_synthetic=False,
              require_done=False, timeout_sec=330.0, poll_interval=2.0,
              monotonic=time.monotonic, sleep=time.sleep, synthetic_factory=make_synthetic_mov) -> dict:
    summary = {"ok": False, "checks": {}, "status_counts": {}, "synthetic_submitted": False,
               "job_terminal": False, "inference_done": False, "synthetic_failed": False,
               "evidence_frames_checked": 0}

    def check(name, valid):
        summary["checks"][name] = bool(valid)
        if not valid:
            raise SmokeFailure(name)

    def request(method, path, *, authenticated=False, timeout=20.0, **kwargs):
        response = client.request(method, url + path, headers={"X-API-Key": key} if authenticated else {},
                                  timeout=timeout, follow_redirects=False, **kwargs)
        code = str(response.status_code)
        summary["status_counts"][code] = summary["status_counts"].get(code, 0) + 1
        if 300 <= response.status_code < 400:
            raise SmokeFailure("redirect_rejected")
        if key.encode("ascii") in response.content:
            raise SmokeFailure("credential_exposed_by_endpoint")
        return response

    try:
        url = validate_url(url)
        if not key or any(ord(char) < 33 or ord(char) > 126 for char in key):
            raise SmokeFailure("missing_or_invalid_key")
        if require_done and not submit_synthetic:
            raise SmokeFailure("require_done_needs_submission")
        if not _number(timeout_sec, minimum=0.001, maximum=330) or not _number(poll_interval, minimum=0.001, maximum=10):
            raise SmokeFailure("invalid_poll_limits")
        response = request("GET", "/healthz")
        check("health_ready", response.status_code == 200 and _valid_health(response.json()))
        response = request("GET", "/api/v1/config")
        check("public_config", response.status_code == 200 and _valid_config(response.json()))
        check("unauthenticated_job_rejected", request("GET", "/api/v1/jobs/smoke-probe-no-such-job").status_code == 401)
        check("unauthenticated_upload_rejected", request("POST", "/api/v1/analyze", content=b"").status_code == 401)
        check("authenticated_unknown_job", request("GET", "/api/v1/jobs/smoke-probe-no-such-job", authenticated=True).status_code == 404)
        for index, path in enumerate(("/review.html", "/runtime/private/", "/runtime/secrets/api_key.txt", "/data/dataset_manifest.json", "/Dataset/", "/models/", "/.env")):
            check(f"private_path_{index}_blocked", request("GET", path).status_code == 404)
        for name, path in (("html", "/"), ("css", "/styles.css"), ("javascript", "/app.js")):
            check(f"static_{name}", request("GET", path).status_code == 200)
        if submit_synthetic:
            source = synthetic_factory()
            response = request("POST", "/api/v1/analyze", authenticated=True,
                               files={"file": ("synthetic-geometric-smoke.mov", source, "video/quicktime")})
            check("synthetic_accepted", response.status_code == 202)
            summary["synthetic_submitted"] = True
            accepted = response.json()
            check("accepted_schema", isinstance(accepted, dict) and set(accepted) == {"job_id"}
                  and isinstance(accepted["job_id"], str) and re.fullmatch(r"[0-9a-f]{32}", accepted["job_id"]) is not None)
            job_id = accepted["job_id"]
            deadline = monotonic() + timeout_sec
            while monotonic() < deadline:
                response = request("GET", f"/api/v1/jobs/{job_id}", authenticated=True, timeout=min(20.0, max(0.001, deadline - monotonic())))
                check("response_within_deadline", monotonic() <= deadline)
                check("poll_status", response.status_code == 200)
                body = response.json()
                check("poll_identity", isinstance(body, dict) and body.get("job_id") == job_id)
                if body.get("status") == "done":
                    summary["evidence_frames_checked"] = validate_done(body, job_id)
                    summary["job_terminal"] = summary["inference_done"] = True
                    break
                if body.get("status") == "failed":
                    check("failed_schema", set(body) == {"job_id", "status", "error_code", "error"}
                          and isinstance(body["error_code"], str) and isinstance(body["error"], str))
                    summary["job_terminal"] = summary["synthetic_failed"] = True
                    check("required_inference_done", not require_done)
                    break
                check("processing_schema", body.get("status") == "processing"
                      and {"job_id", "status"}.issubset(body) and set(body).issubset({"job_id", "status", "stage", "progress"})
                      and (body.get("stage") is None or isinstance(body["stage"], str))
                      and (body.get("progress") is None or _number(body["progress"], maximum=1)))
                remaining = deadline - monotonic()
                if remaining > 0:
                    sleep(min(poll_interval, remaining))
            check("terminal_before_deadline", summary["job_terminal"])
        summary["ok"] = True
    except SmokeFailure as exc:
        summary["error_code"] = exc.code
    except Exception as exc:
        summary["error_code"] = "request_or_validation_error"
        summary["error_class"] = type(exc).__name__
    return summary


def main(argv=None) -> int:
    try:
        parser = SafeArgumentParser(description=__doc__, allow_abbrev=False)
        parser.add_argument("--url", required=True, help="Explicit HTTPS origin; HTTP allowed only for localhost/loopback.")
        parser.add_argument("--key-file", type=Path, help="UTF-8 key file; otherwise use FORTE_API_KEY. Contents are never printed.")
        parser.add_argument("--submit-synthetic", action="store_true", help="Opt in to creating one job from generated shapes.")
        parser.add_argument("--require-done", action="store_true", help="Require valid inference completion; failed synthetic jobs otherwise test transport only.")
        args = parser.parse_args(argv)
        url, key = validate_url(args.url), load_key(args.key_file)
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=20.0) as client:
            summary = run_smoke(url, key, client=client, submit_synthetic=args.submit_synthetic, require_done=args.require_done)
    except SmokeFailure as exc:
        summary = {"ok": False, "error_code": exc.code}
    except Exception as exc:
        summary = {"ok": False, "error_code": "setup_error", "error_class": type(exc).__name__}
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
