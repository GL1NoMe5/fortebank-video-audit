"""Runtime configuration for the ForteBank video audit API."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

DEFAULT_MAX_UPLOAD_BYTES = 256 * 1024 * 1024  # 256 MiB
DEFAULT_ALLOWED_EXTENSIONS = (".mov",)
DEFAULT_MAX_QUEUE_SIZE = 16
DEFAULT_JOB_DEADLINE_SECONDS = 300.0  # 5 minutes end-to-end including queue time
DEFAULT_WORKER_POLL_INTERVAL_SECONDS = 0.05
DEFAULT_WORKER_START_TIMEOUT_SECONDS = 300.0
DEFAULT_SUBPROCESS_JOIN_TIMEOUT_SECONDS = 2.0
DEFAULT_UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024  # 1 MiB
DEFAULT_TRANSCODE_TIMEOUT_SECONDS = 30.0


def _default_root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_int(raw: str | None, default: int, minimum: int = 1) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        parsed = int(raw.strip())
    except ValueError:
        return default
    return max(minimum, parsed)


def _parse_float(raw: str | None, default: float, minimum: float = 0.001) -> float:
    if raw is None or not raw.strip():
        return default
    try:
        parsed = float(raw.strip())
    except ValueError:
        return default
    if not math.isfinite(parsed):
        return default
    return max(minimum, parsed)


def _parse_extensions(raw: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_ALLOWED_EXTENSIONS
    if isinstance(raw, (tuple, list)):
        items = [str(item).strip().lower() for item in raw if str(item).strip()]
    else:
        items = [part.strip().lower() for part in str(raw).split(",") if part.strip()]
    normalized: list[str] = []
    for item in items:
        ext = item if item.startswith(".") else f".{item}"
        if ext not in normalized:
            normalized.append(ext)
    return tuple(normalized) if normalized else DEFAULT_ALLOWED_EXTENSIONS


@dataclass(frozen=True)
class Settings:
    """Immutable settings resolved from FORTE_* environment variables or test overrides."""

    root_dir: Path = field(default_factory=_default_root_dir)
    runtime_dir: Path | None = None
    uploads_dir: Path | None = None
    temp_dir: Path | None = None
    derived_dir: Path | None = None
    db_path: Path | None = None
    web_dir: Path | None = None
    api_key: str | None = None
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS
    max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE
    job_deadline_seconds: float = DEFAULT_JOB_DEADLINE_SECONDS
    worker_poll_interval_seconds: float = DEFAULT_WORKER_POLL_INTERVAL_SECONDS
    worker_start_timeout_seconds: float = DEFAULT_WORKER_START_TIMEOUT_SECONDS
    subprocess_join_timeout_seconds: float = DEFAULT_SUBPROCESS_JOIN_TIMEOUT_SECONDS
    upload_chunk_size_bytes: int = DEFAULT_UPLOAD_CHUNK_SIZE_BYTES
    enable_video_transcode: bool = True
    transcode_timeout_seconds: float = DEFAULT_TRANSCODE_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        resolved_root = Path(self.root_dir).resolve()
        resolved_runtime = (
            Path(self.runtime_dir).resolve()
            if self.runtime_dir is not None
            else (resolved_root / "runtime").resolve()
        )
        resolved_uploads = (
            Path(self.uploads_dir).resolve()
            if self.uploads_dir is not None
            else (resolved_runtime / "uploads").resolve()
        )
        resolved_temp = (
            Path(self.temp_dir).resolve()
            if self.temp_dir is not None
            else (resolved_runtime / "tmp").resolve()
        )
        resolved_derived = (
            Path(self.derived_dir).resolve()
            if self.derived_dir is not None
            else (resolved_runtime / "derived").resolve()
        )
        resolved_db = (
            Path(self.db_path).resolve()
            if self.db_path is not None
            else (resolved_runtime / "jobs.sqlite3").resolve()
        )
        resolved_web = (
            Path(self.web_dir).resolve()
            if self.web_dir is not None
            else (resolved_root / "web").resolve()
        )
        cleaned_key = self.api_key if (self.api_key and self.api_key.strip()) else None

        object.__setattr__(self, "root_dir", resolved_root)
        object.__setattr__(self, "runtime_dir", resolved_runtime)
        object.__setattr__(self, "uploads_dir", resolved_uploads)
        object.__setattr__(self, "temp_dir", resolved_temp)
        object.__setattr__(self, "derived_dir", resolved_derived)
        object.__setattr__(self, "db_path", resolved_db)
        object.__setattr__(self, "web_dir", resolved_web)
        object.__setattr__(self, "api_key", cleaned_key)
        object.__setattr__(self, "allowed_extensions", _parse_extensions(self.allowed_extensions))
        object.__setattr__(self, "max_upload_bytes", max(1, int(self.max_upload_bytes)))
        object.__setattr__(self, "max_queue_size", max(1, int(self.max_queue_size)))
        deadline = float(self.job_deadline_seconds)
        if not math.isfinite(deadline):
            deadline = DEFAULT_JOB_DEADLINE_SECONDS
        object.__setattr__(self, "job_deadline_seconds", min(300.0, max(0.01, deadline)))
        startup_timeout = float(self.worker_start_timeout_seconds)
        if not math.isfinite(startup_timeout):
            startup_timeout = DEFAULT_WORKER_START_TIMEOUT_SECONDS
        object.__setattr__(self, "worker_start_timeout_seconds", min(300.0, max(0.05, startup_timeout)))
        object.__setattr__(
            self,
            "worker_poll_interval_seconds",
            max(0.005, float(self.worker_poll_interval_seconds)),
        )
        object.__setattr__(
            self,
            "subprocess_join_timeout_seconds",
            max(0.05, float(self.subprocess_join_timeout_seconds)),
        )
        object.__setattr__(
            self,
            "upload_chunk_size_bytes",
            max(1024, int(self.upload_chunk_size_bytes)),
        )
        object.__setattr__(
            self,
            "transcode_timeout_seconds",
            max(0.1, float(self.transcode_timeout_seconds)),
        )

    def ensure_directories(self) -> None:
        """Create runtime directories required by the service."""
        assert self.runtime_dir is not None
        assert self.uploads_dir is not None
        assert self.temp_dir is not None
        assert self.derived_dir is not None
        assert self.db_path is not None

        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.derived_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, **overrides: Any) -> Settings:
        """Return a copy of settings with updated fields."""
        return replace(self, **overrides)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: Any) -> Settings:
        """Load settings from FORTE_* environment variables with optional keyword overrides."""
        source = os.environ if env is None else env

        root_raw = source.get("FORTE_ROOT_DIR")
        root_dir = Path(root_raw) if root_raw else _default_root_dir()

        runtime_raw = source.get("FORTE_RUNTIME_DIR")
        uploads_raw = source.get("FORTE_UPLOADS_DIR")
        temp_raw = source.get("FORTE_TEMP_DIR")
        derived_raw = source.get("FORTE_DERIVED_DIR")
        db_raw = source.get("FORTE_DB_PATH")
        web_raw = source.get("FORTE_WEB_DIR")

        queue_raw = (
            source.get("FORTE_MAX_QUEUE_SIZE")
            or source.get("FORTE_QUEUE_CAPACITY")
            or source.get("FORTE_QUEUE_SIZE")
        )
        deadline_raw = (
            source.get("FORTE_JOB_DEADLINE_SECONDS")
            or source.get("FORTE_DEADLINE_SECONDS")
        )

        kwargs: dict[str, Any] = {
            "root_dir": root_dir,
            "runtime_dir": Path(runtime_raw) if runtime_raw else None,
            "uploads_dir": Path(uploads_raw) if uploads_raw else None,
            "temp_dir": Path(temp_raw) if temp_raw else None,
            "derived_dir": Path(derived_raw) if derived_raw else None,
            "db_path": Path(db_raw) if db_raw else None,
            "web_dir": Path(web_raw) if web_raw else None,
            "api_key": source.get("FORTE_API_KEY"),
            "max_upload_bytes": _parse_int(
                source.get("FORTE_MAX_UPLOAD_BYTES"),
                DEFAULT_MAX_UPLOAD_BYTES,
                minimum=1,
            ),
            "allowed_extensions": _parse_extensions(source.get("FORTE_ALLOWED_EXTENSIONS")),
            "max_queue_size": _parse_int(queue_raw, DEFAULT_MAX_QUEUE_SIZE, minimum=1),
            "job_deadline_seconds": _parse_float(
                deadline_raw,
                DEFAULT_JOB_DEADLINE_SECONDS,
                minimum=0.01,
            ),
            "worker_poll_interval_seconds": _parse_float(
                source.get("FORTE_WORKER_POLL_INTERVAL_SECONDS"),
                DEFAULT_WORKER_POLL_INTERVAL_SECONDS,
                minimum=0.005,
            ),
            "worker_start_timeout_seconds": _parse_float(
                source.get("FORTE_WORKER_START_TIMEOUT_SECONDS"),
                DEFAULT_WORKER_START_TIMEOUT_SECONDS,
                minimum=0.05,
            ),
            "subprocess_join_timeout_seconds": _parse_float(
                source.get("FORTE_SUBPROCESS_JOIN_TIMEOUT_SECONDS"),
                DEFAULT_SUBPROCESS_JOIN_TIMEOUT_SECONDS,
                minimum=0.05,
            ),
            "upload_chunk_size_bytes": _parse_int(
                source.get("FORTE_UPLOAD_CHUNK_SIZE_BYTES"),
                DEFAULT_UPLOAD_CHUNK_SIZE_BYTES,
                minimum=1024,
            ),
            "enable_video_transcode": _parse_bool(
                source.get("FORTE_ENABLE_VIDEO_TRANSCODE"),
                True,
            ),
            "transcode_timeout_seconds": _parse_float(
                source.get("FORTE_TRANSCODE_TIMEOUT_SECONDS"),
                DEFAULT_TRANSCODE_TIMEOUT_SECONDS,
                minimum=0.1,
            ),
        }
        kwargs.update(overrides)
        return cls(**kwargs)


def load_settings(env: Mapping[str, str] | None = None, **overrides: Any) -> Settings:
    """Helper to construct validated Settings."""
    return Settings.from_env(env=env, **overrides)
