"""Local-only human review. Never expose this app through the public tunnel.

Run: .venv/Scripts/python.exe scripts/review_local.py
Synthetic checks: .venv/Scripts/python.exe scripts/review_local.py --self-test
No inference, outbound requests, or footage/description logging is performed.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
import secrets
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate import splits


class ReviewEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule_id: str = Field(min_length=1, max_length=100)
    start_sec: float = Field(ge=0, allow_inf_nan=False)
    end_sec: float = Field(ge=0, allow_inf_nan=False)
    timecode_sec: float = Field(ge=0, allow_inf_nan=False)
    description: str = Field(max_length=4000)
    confirmed: StrictBool = False


class ReviewSave(BaseModel):
    model_config = ConfigDict(extra="forbid")
    events: list[ReviewEvent] = Field(default_factory=list, max_length=100)
    review_complete: StrictBool = False
    base_revision: int = Field(default=0, ge=0, strict=True)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _within(path: Path, parent: Path) -> bool:
    return path.resolve().is_relative_to(parent.resolve())


def _path(root: Path, value: str) -> Path:
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def create_app(root: Path = ROOT) -> FastAPI:
    root = root.resolve()
    manifest = _read_json(root / "data" / "dataset_manifest.json")
    rulebook = _read_json(root / "config" / "normative_rules_rk.json")
    records = manifest["records"]
    dataset_root = _path(root, manifest["dataset_root"])
    if not _within(dataset_root, root) or dataset_root == root:
        raise ValueError("Dataset must be a dedicated directory inside the workspace")
    rules = {rule["id"]: rule for rule in rulebook["rules"]}
    by_id, videos = {}, {}
    for record in records:
        video_id = record["id"]
        if not isinstance(video_id, str) or not video_id or any(c in video_id for c in "/\\") or video_id in by_id:
            raise ValueError("Invalid or duplicate manifest ID")
        candidate = _path(root, record["path"])
        if not _within(candidate, dataset_root) or candidate.suffix.lower() != ".mov":
            raise ValueError("Manifest video is outside its allowed dataset directory")
        duration = float(record["metadata"]["duration_sec"])
        if not math.isfinite(duration) or duration <= 0 or record["label"] not in (0, 1):
            raise ValueError("Invalid manifest metadata")
        by_id[video_id], videos[video_id] = record, candidate
    holdout = {record["id"] for record in splits(records)["holdout"]}
    private_dir = root / "runtime" / "private"
    labels_path = root / "runtime" / "review_labels.json"
    html_path = root / "web" / "review.html"
    token = secrets.token_urlsafe(32)
    lock = threading.RLock()
    app = FastAPI(title="ForteBank: локальная ручная разметка", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def loopback_only(request: Request, call_next):
        try:
            loopback = request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
        except ValueError:
            loopback = False
        host = request.url.hostname
        if not loopback or host not in {"localhost", "127.0.0.1", "::1"}:
            return JSONResponse({"detail": "Доступ разрешён только с этого компьютера."}, status_code=403)
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "Межсайтовый запрос запрещён."}, status_code=403)
        origin = request.headers.get("origin")
        if origin:
            parsed = urlsplit(origin)
            if parsed.scheme != request.url.scheme or parsed.netloc != request.url.netloc:
                return JSONResponse({"detail": "Недопустимый источник запроса."}, status_code=403)
        if request.method not in {"GET", "HEAD"}:
            if not secrets.compare_digest(request.headers.get("x-review-token", ""), token):
                return JSONResponse({"detail": "Обновите страницу перед сохранением."}, status_code=403)
            try:
                content_length = int(request.headers.get("content-length", "0"))
            except ValueError:
                return JSONResponse({"detail": "Недопустимая длина запроса."}, status_code=400)
            if content_length > 1_000_000:
                return JSONResponse({"detail": "Слишком большой запрос."}, status_code=413)
        response = await call_next(request)
        response.headers.update({
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'; form-action 'self'",
        })
        return response

    def record_for(video_id: str) -> dict:
        if video_id not in by_id:
            raise HTTPException(404, "Ролик отсутствует в манифесте.")
        return by_id[video_id]

    def load_labels() -> dict:
        if not labels_path.exists():
            return {"schema_version": 1, "split_seed": 26, "records": {}}
        try:
            payload = _read_json(labels_path)
            if payload.get("schema_version") != 1 or not isinstance(payload.get("records"), dict):
                raise ValueError("Invalid review schema")
            return payload
        except (OSError, ValueError, TypeError):
            raise HTTPException(500, "Файл разметки повреждён или недоступен; он не перезаписан.") from None

    @app.get("/")
    def index():
        return FileResponse(html_path, media_type="text/html")

    @app.get("/bootstrap")
    def bootstrap():
        return {
            "token": token,
            "rules": [{key: rule.get(key) for key in ("id", "title", "description", "supported", "insufficient_evidence", "exceptions")} for rule in rules.values()],
            "records": [{
                "id": record["id"], "label": record["label"], "annotated_sec": record.get("annotated_sec"),
                "duration_sec": record["metadata"]["duration_sec"],
                "split": "holdout" if record["id"] in holdout else "train",
                "source_type_label_available": False,
            } for record in records],
        }

    @app.get("/video/{video_id}")
    def video(video_id: str):
        record_for(video_id)
        candidate = videos[video_id]
        # Re-check the resolved target so a replaced symlink cannot escape the allowlist.
        if candidate.resolve() != candidate or not _within(candidate, dataset_root) or not candidate.is_file():
            raise HTTPException(404, "Локальный видеофайл недоступен.")
        return FileResponse(candidate, media_type="video/quicktime", filename=candidate.name, content_disposition_type="inline")

    @app.get("/labels/{video_id}")
    def labels(video_id: str):
        record_for(video_id)
        with lock:
            return load_labels()["records"].get(video_id, {"events": [], "review_complete": False, "revision": 0})

    @app.post("/labels/{video_id}")
    def save_labels(video_id: str, body: ReviewSave):
        record = record_for(video_id)
        duration = float(record["metadata"]["duration_sec"])
        events = []
        for event in body.events:
            if event.rule_id not in rules:
                raise HTTPException(422, "Неизвестный тип события.")
            if not 0 <= event.start_sec <= event.timecode_sec <= event.end_sec <= duration:
                raise HTTPException(422, "Нужен интервал: 0 ≤ начало ≤ таймкод ≤ конец ≤ длительность ролика.")
            if event.confirmed and not event.description.strip():
                raise HTTPException(422, "Для подтверждённого события добавьте фактическое описание.")
            saved = event.model_dump()
            saved["description"] = saved["description"].strip()
            saved["normative_supported"] = bool(rules[event.rule_id].get("supported"))
            events.append(saved)
        with lock:
            payload = load_labels()
            previous = payload["records"].get(video_id, {})
            revision = previous.get("revision", 0)
            if revision != body.base_revision:
                raise HTTPException(409, "Разметка изменена в другой вкладке. Перезагрузите ролик, прежде чем сохранять.")
            saved_record = {
                "video_id": video_id, "binary_label": record["label"], "annotated_sec": record.get("annotated_sec"),
                "source_type_label_available": False, "split": "holdout" if video_id in holdout else "train",
                "calibration_allowed": video_id not in holdout,
                "review_complete": body.review_complete, "events": events,
                "revision": revision + 1, "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            payload["records"][video_id] = saved_record
            labels_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = labels_path.with_name(f".{labels_path.name}.{secrets.token_hex(8)}.tmp")
            try:
                with temporary.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, labels_path)
            finally:
                temporary.unlink(missing_ok=True)
        return {"saved": True, "revision": saved_record["revision"], "calibration_allowed": saved_record["calibration_allowed"]}

    @app.get("/report/{video_id}")
    def report(video_id: str):
        record_for(video_id)
        if not private_dir.is_dir() or not _within(private_dir, root / "runtime"):
            return {"found": False}
        candidates = sorted(private_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        for candidate in candidates:
            try:
                if not _within(candidate, private_dir) or candidate.stat().st_size > 50_000_000:
                    continue
                payload = _read_json(candidate)
                value = payload.get("video_path")
                if isinstance(value, str) and _path(root, value) == videos[video_id]:
                    return {"found": True, "report": payload}
            except (OSError, ValueError, TypeError):
                continue
        return {"found": False}

    return app


def self_test() -> None:
    """Exercise only fabricated paths, bytes and descriptions in a temporary directory."""
    from fastapi.testclient import TestClient

    with tempfile.TemporaryDirectory(prefix="forte-review-test-") as temporary:
        root = Path(temporary)
        for name in ("data", "config", "Dataset", "runtime/private", "web"):
            (root / name).mkdir(parents=True)
        rows = [{"id": f"synthetic-{i}", "label": i % 2, "annotated_sec": 2 if i % 2 else None, "path": f"Dataset/{i}.mov", "metadata": {"duration_sec": 10}} for i in range(10)]
        for i in range(10):
            (root / "Dataset" / f"{i}.mov").write_bytes(b"synthetic-video-bytes")
        (root / "data/dataset_manifest.json").write_text(json.dumps({"dataset_root": "Dataset", "records": rows}), encoding="utf-8")
        (root / "config/normative_rules_rk.json").write_text(json.dumps({"rules": [{"id": "synthetic_rule", "title": "Synthetic", "supported": True}]}), encoding="utf-8")
        (root / "web/review.html").write_text("<!doctype html><title>Synthetic</title>", encoding="utf-8")
        (root / "runtime/private/synthetic.json").write_text(json.dumps({"video_path": str(root / rows[0]["path"]), "result": {"verdict": "no_violation", "violations": []}}), encoding="utf-8")
        app = create_app(root)
        with TestClient(app, base_url="http://127.0.0.1:8766", client=("127.0.0.1", 12345)) as client:
            boot = client.get("/bootstrap").json()
            headers = {"X-Review-Token": boot["token"]}
            held_id = next(row["id"] for row in boot["records"] if row["split"] == "holdout")
            train_id = next(row["id"] for row in boot["records"] if row["split"] == "train")
            body = {"base_revision": 0, "review_complete": True, "events": [{"rule_id": "synthetic_rule", "start_sec": 1, "timecode_sec": 2, "end_sec": 3, "description": "Synthetic observation only", "confirmed": True}]}
            assert client.get("/").status_code == 200
            assert client.get("/video/not-in-manifest").status_code == 404
            ranged = client.get("/video/synthetic-0", headers={"Range": "bytes=0-8"})
            assert ranged.status_code == 206 and ranged.content == b"synthetic"
            assert client.get("/report/synthetic-0").json()["found"] is True
            assert client.get("/report/synthetic-1").json()["found"] is False
            assert client.post(f"/labels/{held_id}", json=body).status_code == 403
            assert client.post(f"/labels/{held_id}", json=body, headers={**headers, "Origin": "https://example.invalid"}).status_code == 403
            saved = client.post(f"/labels/{held_id}", json=body, headers=headers)
            assert saved.status_code == 200 and saved.json()["calibration_allowed"] is False
            assert client.post(f"/labels/{held_id}", json=body, headers=headers).status_code == 409
            body["base_revision"] = 1
            body["events"][0]["end_sec"] = 0
            assert client.post(f"/labels/{held_id}", json=body, headers=headers).status_code == 422
            stored = client.get(f"/labels/{held_id}").json()
            assert stored["events"][0]["confirmed"] is True and stored["revision"] == 1
            body["events"][0]["end_sec"] = 3
            body["events"][0]["rule_id"] = "not_in_rulebook"
            assert client.post(f"/labels/{held_id}", json=body, headers=headers).status_code == 422
            empty = client.post(f"/labels/{train_id}", json={"base_revision": 0, "review_complete": True, "events": []}, headers=headers)
            assert empty.status_code == 200 and empty.json()["calibration_allowed"] is True
            assert len(_read_json(root / "runtime/review_labels.json")["records"]) == 2
            assert client.get("/bootstrap", headers={"Host": "external.invalid"}).status_code == 403
            assert client.get("/bootstrap", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
            assert client.get("/api/v1/analyze").status_code == 404
            (root / "runtime/review_labels.json").write_text("{invalid-json", encoding="utf-8")
            assert client.post(f"/labels/{train_id}", json={"base_revision": 1, "events": []}, headers=headers).status_code == 500
            assert (root / "runtime/review_labels.json").read_text(encoding="utf-8") == "{invalid-json"
        with TestClient(app, base_url="http://127.0.0.1:8766", client=("192.0.2.1", 12345)) as remote:
            assert remote.get("/bootstrap").status_code == 403
        rows[0]["path"] = "web/review.html"
        (root / "data/dataset_manifest.json").write_text(json.dumps({"dataset_root": "Dataset", "records": rows}), encoding="utf-8")
        try:
            create_app(root)
        except ValueError:
            pass
        else:
            raise AssertionError("Escaping manifest path was accepted")
    print("Synthetic review checks passed; no dataset media or private reports were read.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    import uvicorn

    print(f"Local human review: http://127.0.0.1:{args.port} (do not publish or tunnel)", flush=True)
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, proxy_headers=False, access_log=False)


if __name__ == "__main__":
    main()
