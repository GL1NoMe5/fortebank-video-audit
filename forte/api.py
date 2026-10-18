"""FastAPI application for ForteBank local video audit infrastructure."""

from __future__ import annotations

import logging
import importlib.util
import os
import secrets
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path, PurePath
from typing import Any, Callable

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from forte.jobs import (
    JobManager,
    JobNotFoundError,
    QueueFullError,
    sanitize_diagnostic_text,
)
from forte.schemas import (
    AnalyzeAcceptedResponse,
    ErrorResponse,
    HealthResponse,
    JobDoneResponse,
    JobFailedResponse,
    JobProcessingResponse,
    PublicConfigResponse,
)
from forte.settings import Settings, load_settings

logger = logging.getLogger("forte.api")
_PREVIEW_LOCK = threading.Lock()

_DISALLOWED_MIME_PREFIXES = (
    "text/html",
    "text/javascript",
    "application/javascript",
    "application/x-msdownload",
    "application/x-executable",
    "application/x-dosexec",
    "application/x-sh",
)

_DISALLOWED_MAGIC_PREFIXES = (
    b"MZ",
    b"\x7fELF",
    b"#!",
    b"<!DOCTYPE html",
    b"<!doctype html",
    b"<html",
    b"<script",
)


class APIHTTPError(Exception):
    """Structured HTTP error with status_code, detail, and error_code."""

    def __init__(self, status_code: int, detail: str, error_code: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code


class UploadGuard:
    """Reject unauthorized/oversized streams before multipart spooling."""
    def __init__(self, app: Any, settings: Settings) -> None:
        self.app, self.settings = app, settings

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get('type') != 'http' or scope.get('path') != '/api/v1/analyze':
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get('headers', []))
        expected = (self.settings.api_key or os.environ.get('FORTE_API_KEY') or '').strip()
        supplied = headers.get(b'x-api-key', b'')
        if not expected or not secrets.compare_digest(supplied, expected.encode('utf-8')):
            await JSONResponse({'detail': 'Invalid or missing API key.', 'error_code': 'unauthorized'}, status_code=401)(scope, receive, send)
            return
        limit = self.settings.max_upload_bytes + 1024 * 1024
        try:
            declared = int(headers.get(b'content-length', b'0'))
        except ValueError:
            declared = limit + 1
        if declared > limit:
            await JSONResponse({'detail': 'Upload exceeds the size limit.', 'error_code': 'file_too_large'}, status_code=413)(scope, receive, send)
            return
        received = 0

        async def bounded_receive() -> dict:
            nonlocal received
            message = await receive()
            if message.get('type') == 'http.request':
                received += len(message.get('body', b''))
                if received > limit:
                    raise HTTPException(413, 'Upload exceeds the size limit.')
            return message
        await self.app(scope, bounded_receive, send)


def _extract_safe_extension(untrusted_filename: str | None) -> str:
    """Extract lowercase file extension from an untrusted client filename without path traversal."""
    if not untrusted_filename:
        return ""
    normalized = untrusted_filename.replace("\\", "/").strip()
    base_name = PurePath(normalized).name.strip()
    if not base_name or base_name in {".", ".."}:
        return ""
    suffix = PurePath(base_name).suffix.lower()
    stem = PurePath(base_name).stem.strip()
    if not stem or not suffix:
        return ""
    return suffix


def _has_disallowed_header(header_bytes: bytes) -> bool:
    stripped = header_bytes.lstrip()
    for prefix in _DISALLOWED_MAGIC_PREFIXES:
        if header_bytes.startswith(prefix) or stripped.startswith(prefix):
            return True
    return False


def _safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _video_preview_available(settings: Settings) -> bool:
    if not settings.enable_video_transcode:
        return False
    return importlib.util.find_spec("av") is not None


def _remux_preview(source: Path, destination: Path, timeout: float) -> None:
    """Copy compressed H264 packets into MP4; never decode frames in the API."""
    import av
    deadline = time.monotonic() + timeout
    with av.open(str(source)) as input_container:
        video = input_container.streams.video[0]
        if video.codec_context.name != "h264":
            raise ValueError("Browser preview requires H264 video.")
        with av.open(str(destination), mode="w", format="mp4", options={"movflags": "+faststart"}) as output_container:
            output_stream = output_container.add_stream_from_template(video)
            for packet in input_container.demux(video):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Preview generation timed out.")
                if packet.dts is None:
                    continue
                packet.stream = output_stream
                output_container.mux(packet)


def create_app(
    settings: Settings | None = None,
    analyzer_factory: Callable[[], Any] | Callable[..., Any] | None = None,
) -> FastAPI:
    """Application factory for ForteBank video audit backend."""

    resolved_settings = settings if settings is not None else load_settings()
    resolved_settings.ensure_directories()
    job_manager = JobManager(settings=resolved_settings, analyzer_factory=analyzer_factory)

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        app_instance.state.settings = resolved_settings
        app_instance.state.job_manager = job_manager
        previous_tempdir = tempfile.tempdir
        tempfile.tempdir = str(resolved_settings.temp_dir)
        job_manager.start()
        try:
            yield
        finally:
            job_manager.stop()
            tempfile.tempdir = previous_tempdir

    app_instance = FastAPI(
        title="ForteBank Video Audit API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app_instance.state.settings = resolved_settings
    app_instance.state.job_manager = job_manager
    app_instance.add_middleware(UploadGuard, settings=resolved_settings)
    app_instance.add_middleware(CORSMiddleware,
        allow_origins=['https://gl1nome5.github.io'],
        allow_methods=['GET', 'POST'], allow_headers=['X-API-Key', 'Content-Type'])

    @app_instance.exception_handler(APIHTTPError)
    async def _handle_api_http_error(_request: Request, exc: APIHTTPError) -> JSONResponse:
        payload = ErrorResponse(
            detail=sanitize_diagnostic_text(exc.detail, fallback="Request failed."),
            error_code=exc.error_code,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=payload.model_dump(mode="json"),
        )

    @app_instance.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first_msg = "Invalid request payload."
        errors = exc.errors()
        if errors:
            loc = ".".join(str(part) for part in errors[0].get("loc", ()))
            msg = str(errors[0].get("msg", "Invalid value"))
            first_msg = f"{loc}: {msg}" if loc else msg
        payload = ErrorResponse(
            detail=sanitize_diagnostic_text(first_msg, fallback="Invalid request payload."),
            error_code="validation_error",
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=payload.model_dump(mode="json"),
        )

    @app_instance.exception_handler(Exception)
    async def _handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled server error: %s", type(exc).__name__)
        payload = ErrorResponse(
            detail="Internal server error.",
            error_code="internal_error",
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=payload.model_dump(mode="json"),
        )

    def require_api_key(
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        current_settings: Settings = request.app.state.settings
        configured_key = current_settings.api_key or os.environ.get("FORTE_API_KEY")
        configured_key = configured_key.strip() if configured_key and configured_key.strip() else None

        if not configured_key:
            secrets.compare_digest((x_api_key or "").encode("utf-8"), b"__unconfigured__")
            raise APIHTTPError(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed: API key is not configured or invalid.",
                error_code="unauthorized",
            )

        if not x_api_key:
            secrets.compare_digest(b"", configured_key.encode("utf-8"))
            raise APIHTTPError(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing X-API-Key header.",
                error_code="unauthorized",
            )

        matched = secrets.compare_digest(
            x_api_key.encode("utf-8"),
            configured_key.encode("utf-8"),
        )
        if not matched:
            raise APIHTTPError(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid X-API-Key header.",
                error_code="unauthorized",
            )

    @app_instance.get("/healthz", response_model=HealthResponse)
    async def healthz(request: Request) -> HealthResponse:
        current_settings: Settings = request.app.state.settings
        manager: JobManager = request.app.state.job_manager
        configured_key = current_settings.api_key or os.environ.get("FORTE_API_KEY")
        api_key_ok = bool(configured_key and configured_key.strip())
        worker_ok = manager.is_worker_alive()
        queue_depth = manager.get_active_job_count()
        ready = bool(api_key_ok and manager.is_worker_ready())
        return HealthResponse(
            status="ok" if ready else "degraded",
            ready=ready,
            worker_alive=worker_ok,
            api_key_configured=api_key_ok,
            queue_depth=queue_depth,
            queue_capacity=current_settings.max_queue_size,
        )

    @app_instance.get("/api/v1/config", response_model=PublicConfigResponse)
    async def get_public_config(request: Request) -> PublicConfigResponse:
        current_settings: Settings = request.app.state.settings
        return PublicConfigResponse(
            max_upload_bytes=current_settings.max_upload_bytes,
            max_upload_mib=current_settings.max_upload_bytes // (1024 * 1024),
            allowed_extensions=list(current_settings.allowed_extensions),
            max_queue_size=current_settings.max_queue_size,
            job_deadline_seconds=current_settings.job_deadline_seconds,
            auth_header="X-API-Key",
            requires_api_key=True,
            video_preview_supported=_video_preview_available(current_settings),
        )

    @app_instance.post(
        "/api/v1/analyze",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=AnalyzeAcceptedResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def analyze_video(
        request: Request,
        file: UploadFile | None = File(default=None),
    ) -> AnalyzeAcceptedResponse:
        current_settings: Settings = request.app.state.settings
        manager: JobManager = request.app.state.job_manager

        if file is None:
            raise APIHTTPError(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Multipart field 'file' is required.",
                error_code="missing_file",
            )

        ext = _extract_safe_extension(file.filename)
        if ext not in current_settings.allowed_extensions:
            await file.close()
            allowed_str = ", ".join(current_settings.allowed_extensions)
            raise APIHTTPError(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unsupported file extension. Only {allowed_str} uploads are accepted.",
                error_code="invalid_extension",
            )

        content_type = (file.content_type or "").strip().lower()
        if any(content_type.startswith(prefix) for prefix in _DISALLOWED_MIME_PREFIXES):
            await file.close()
            raise APIHTTPError(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Uploaded content type is not permitted for video analysis.",
                error_code="invalid_content_type",
            )

        if not manager.has_capacity():
            await file.close()
            raise APIHTTPError(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Analysis queue is full (max_queue_size={current_settings.max_queue_size}). Please retry shortly.",
                error_code="queue_full",
            )

        assert current_settings.temp_dir is not None
        assert current_settings.uploads_dir is not None
        current_settings.ensure_directories()

        job_id = manager.create_job_Id()
        temp_path = current_settings.temp_dir / f"{job_id}_{uuid.uuid4().hex}.part"
        final_path = current_settings.uploads_dir / f"{job_id}{ext}"
        total_bytes = 0
        header_sample = b""

        try:
            with temp_path.open("wb") as out_file:
                while True:
                    chunk = await file.read(current_settings.upload_chunk_size_bytes)
                    if not chunk:
                        break
                    if len(header_sample) < 64:
                        needed = 64 - len(header_sample)
                        header_sample += chunk[:needed]
                    total_bytes += len(chunk)
                    if total_bytes > current_settings.max_upload_bytes:
                        raise APIHTTPError(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"Uploaded file exceeds the maximum limit of {current_settings.max_upload_bytes} bytes.",
                            error_code="file_too_large",
                        )
                    out_file.write(chunk)

            if total_bytes == 0:
                raise APIHTTPError(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Uploaded video file cannot be empty.",
                    error_code="empty_file",
                )

            if _has_disallowed_header(header_sample):
                raise APIHTTPError(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Uploaded file header does not match a valid video container.",
                    error_code="invalid_file_content",
                )

            os.replace(temp_path, final_path)
            try:
                manager.enqueue_job(job_id=job_id, video_path=final_path)
            except QueueFullError as exc:
                _safe_unlink(final_path)
                raise APIHTTPError(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=str(exc),
                    error_code="queue_full",
                ) from exc
        except Exception:
            _safe_unlink(temp_path)
            _safe_unlink(final_path)
            raise
        finally:
            await file.close()

        return AnalyzeAcceptedResponse(job_id=job_id)

    @app_instance.get(
        "/api/v1/jobs/{job_id}",
        response_model=JobProcessingResponse | JobDoneResponse | JobFailedResponse,
        dependencies=[Depends(require_api_key)],
    )
    async def get_job_status(request: Request, job_id: str) -> JSONResponse:
        manager: JobManager = request.app.state.job_manager
        try:
            job_response = manager.get_job_response(job_id)
        except JobNotFoundError as exc:
            raise APIHTTPError(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Requested job was not found.",
                error_code="job_not_found",
            ) from exc

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=job_response.model_dump(mode="json", exclude_none=True),
        )

    @app_instance.get('/api/v1/jobs/{job_id}/timeline', dependencies=[Depends(require_api_key)])
    def get_job_timeline(request: Request, job_id: str) -> JSONResponse:
        import json
        from forte.analysis.timeline_memory import MEMORY, file_hash, request_metadata_path
        manager: JobManager = request.app.state.job_manager
        try:
            record = manager.get_job_record(job_id)
        except JobNotFoundError as exc:
            raise APIHTTPError(404, 'Requested job was not found.', 'job_not_found') from exc
        if record.status != 'done' or not record.video_path.is_file():
            raise APIHTTPError(409, 'The completed timeline is not available.', 'timeline_not_ready')
        digest = file_hash(record.video_path)
        folder = MEMORY / digest
        chronology = folder / 'chronology.json'
        analysis = request_metadata_path(record.video_path, digest)
        if not chronology.exists() or not analysis.exists():
            raise APIHTTPError(404, 'This job has no full timeline.', 'timeline_not_found')
        log = json.loads(chronology.read_text(encoding='utf-8'))
        details = json.loads(analysis.read_text(encoding='utf-8'))
        return JSONResponse({'intervals':log['intervals'], 'metrics':details['metrics']})

    @app_instance.get(
        "/api/v1/jobs/{job_id}/video",
        dependencies=[Depends(require_api_key)],
    )
    def get_job_video(request: Request, job_id: str) -> FileResponse:
        current_settings: Settings = request.app.state.settings
        manager: JobManager = request.app.state.job_manager
        try:
            record = manager.get_job_record(job_id)
        except JobNotFoundError as exc:
            raise APIHTTPError(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Requested job was not found.",
                error_code="job_not_found",
            ) from exc

        assert current_settings.derived_dir is not None
        assert current_settings.temp_dir is not None
        mp4_path = current_settings.derived_dir / f"{record.job_id}.mp4"
        if mp4_path.is_file():
            return FileResponse(path=str(mp4_path), media_type="video/mp4", filename=f"{record.job_id}.mp4")

        if not record.video_path.is_file():
            raise APIHTTPError(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source video for this job is unavailable.",
                error_code="video_not_found",
            )

        if not _video_preview_available(current_settings):
            raise APIHTTPError(503, 'Browser video preview is unavailable; use the uploaded local file.',
                'video_preview_unavailable')
        temp_mp4 = current_settings.temp_dir / f'{record.job_id}_{uuid.uuid4().hex}.mp4'
        try:
            with _PREVIEW_LOCK:
                if not mp4_path.is_file():
                    _remux_preview(record.video_path, temp_mp4, current_settings.transcode_timeout_seconds)
                    os.replace(temp_mp4, mp4_path)
        except Exception as exc:
            _safe_unlink(temp_mp4)
            raise APIHTTPError(503, 'Browser preview could not be generated; use the uploaded local file.',
                'video_preview_unavailable') from exc

        return FileResponse(path=str(mp4_path), media_type="video/mp4", filename=f"{record.job_id}.mp4")

    web_dir = resolved_settings.web_dir
    if web_dir is not None and web_dir.is_dir():
        index_file = web_dir / "index.html"
        if index_file.is_file():
            @app_instance.get("/", include_in_schema=False)
            async def serve_index() -> FileResponse:
                return FileResponse(str(index_file), media_type="text/html")

        @app_instance.get('/styles.css', include_in_schema=False)
        async def serve_css() -> FileResponse:
            return FileResponse(str(web_dir / 'styles.css'), media_type='text/css')

        @app_instance.get('/app.js', include_in_schema=False)
        async def serve_js() -> FileResponse:
            return FileResponse(str(web_dir / 'app.js'), media_type='application/javascript')

    return app_instance


app = create_app()
