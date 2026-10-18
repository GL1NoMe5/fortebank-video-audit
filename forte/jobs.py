"""Persistent SQLite job manager, bounded disk queue, and isolated GPU worker."""

from __future__ import annotations

import inspect
import json
import logging
import math
import multiprocessing
import pickle
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from forte.schemas import (
    AnalyzerOutput,
    JobDoneResponse,
    JobFailedResponse,
    JobProcessingResponse,
)
from forte.settings import Settings

logger = logging.getLogger("forte.jobs")
PUBLIC_FAILURE_MESSAGES = {
    'service_restart': 'Job was interrupted by a service restart. Submit it again.',
    'job_timeout': 'The five-minute processing deadline was exceeded. The worker will recover automatically.',
    'missing_video_file': 'The uploaded video is no longer available.',
    'invalid_analyzer_output': 'Local analysis did not produce a valid result.',
    'analyzer_error': 'Local video analysis failed. Please try again or contact the operator.',
    'worker_crashed': 'The analysis worker stopped unexpectedly and is restarting.',
}

_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s'\"`:,;)]+")
_POSIX_PATH_RE = re.compile(r"(?:/[A-Za-z0-9._-]+){2,}")


class QueueFullError(Exception):
    """Raised when the job queue has reached its configured capacity."""


class JobNotFoundError(Exception):
    """Raised when a requested job ID does not exist."""


@dataclass(frozen=True)
class JobRecord:
    """Internal representation of a persisted job row."""

    job_id: str
    status: str
    stage: str | None
    progress: float | None
    created_at: float
    updated_at: float
    deadline_at: float
    video_path: Path
    verdict: str | None
    violations_json: str | None
    error_code: str | None
    error_message: str | None
    worker_pid: int | None


def sanitize_diagnostic_text(text: str | None, fallback: str = "Analysis failed.") -> str:
    """Strip local filesystem paths, tracebacks, and control characters from error strings."""
    if not text:
        return fallback
    first_line = ""
    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Traceback (most recent call last)"):
            continue
        if stripped.startswith("File \""):
            continue
        first_line = stripped
        break
    if not first_line:
        first_line = fallback
    sanitized = _WINDOWS_PATH_RE.sub("<path>", first_line)
    sanitized = _POSIX_PATH_RE.sub("<path>", sanitized)
    if len(sanitized) > 280:
        sanitized = sanitized[:277] + "..."
    return sanitized


def _production_analyzer_factory() -> Any:
    """Lazily load the production local VideoAnalyzer implementation.

    Never returns a fake or fallback implementation.
    """
    from forte.analysis.pipeline import VideoAnalyzer  # type: ignore[import-not-found]

    return VideoAnalyzer()


def _resolve_analyzer(analyzer_or_factory: Any) -> Any:
    """Resolve a factory once in the GPU process, never once per video."""
    if hasattr(analyzer_or_factory, "analyze") and not isinstance(analyzer_or_factory, type):
        return analyzer_or_factory
    elif isinstance(analyzer_or_factory, type):
        return analyzer_or_factory()
    elif callable(analyzer_or_factory):
        try:
            sig = inspect.signature(analyzer_or_factory)
            required_positional = [
                p
                for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and p.default is inspect.Parameter.empty
            ]
        except (TypeError, ValueError):
            required_positional = []

        if len(required_positional) >= 1:
            return analyzer_or_factory
        else:
            return analyzer_or_factory()
    else:
        raise TypeError("Configured analyzer factory is not callable and has no analyze method.")


def _invoke_analyzer(analyzer: Any, video_path: Path, progress_cb: Callable[[str, float], None]) -> dict[str, Any]:
    target = analyzer.analyze if hasattr(analyzer, "analyze") else analyzer
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return target(video_path, progress=progress_cb)
    for args, kwargs in [((video_path,), {"progress": progress_cb}), ((video_path, progress_cb), {}), ((video_path,), {})]:
        try:
            sig.bind(*args, **kwargs)
        except TypeError:
            continue
        return target(*args, **kwargs)
    raise TypeError("Analyzer has an unsupported call signature.")


def _subprocess_worker_entry(
    analyzer_factory: Any,
    connection: Connection,
) -> None:
    """Persistent GPU worker; warm up one real analyzer and accept sequential jobs."""
    try:
        try:
            analyzer = _resolve_analyzer(analyzer_factory)
            warmup = getattr(analyzer, "warmup", None)
            if callable(warmup):
                warmup()
            connection.send({"kind": "ready"})
        except Exception as exc:
            logger.exception('Local analyzer initialization failed.')
            connection.send({"kind": "startup_error", "message": f"Analyzer initialization failed ({type(exc).__name__})."})
            return
        while True:
            try:
                request = connection.recv()
            except EOFError:
                return
            if request.get("kind") == "shutdown":
                return
            job_id = request["job_id"]

            def _report_progress(stage: str, progress: float) -> None:
                numeric = float(progress)
                if not math.isfinite(numeric):
                    numeric = 0.0
                connection.send({"kind": "progress", "job_id": job_id,
                    "stage": sanitize_diagnostic_text(str(stage or "analyzing"), fallback="analyzing")[:80],
                    "progress": max(0.0, min(1.0, numeric))})

            try:
                result = _invoke_analyzer(analyzer, Path(request["video_path"]), _report_progress)
                connection.send({"kind": "result", "job_id": job_id, "payload": result})
            except Exception as exc:
                logger.exception('Local analyzer failed for job %s.', job_id)
                connection.send({"kind": "error", "job_id": job_id,
                    "message": f"Local analysis failed ({type(exc).__name__})."})
    finally:
        try:
            connection.close()
        except Exception:
            pass


def _is_picklable(obj: Any) -> bool:
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


class JobManager:
    """Durable SQLite job manager with a single background worker and subprocess isolation."""

    def __init__(
        self,
        settings: Settings,
        analyzer_factory: Callable[[], Any] | Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self._analyzer_factory = (
            analyzer_factory if analyzer_factory is not None else _production_analyzer_factory
        )
        self._db_lock = threading.RLock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._current_process: multiprocessing.Process | None = None
        self._worker_connection: Connection | None = None
        self._worker_started_at: float | None = None
        self._worker_ready = threading.Event()
        self._restart_not_before = 0.0
        if not _is_picklable(self._analyzer_factory):
            raise TypeError("Analyzer factory must be picklable for isolated Windows spawn workers.")
        self._proc_lock = threading.Lock()
        self._mp_ctx = multiprocessing.get_context("spawn")

        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        assert self.settings.db_path is not None
        conn = sqlite3.connect(
            str(self.settings.db_path),
            timeout=10.0,
            isolation_level=None,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        with self._db_lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        stage TEXT,
                        progress REAL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        deadline_at REAL NOT NULL,
                        video_path TEXT NOT NULL,
                        verdict TEXT,
                        violations_json TEXT,
                        error_code TEXT,
                        error_message TEXT,
                        worker_pid INTEGER
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);"
                )
            finally:
                conn.close()

    def recover_unfinished_jobs(self) -> int:
        """Mark unfinished jobs from previous runs as failed instead of silently requeueing."""
        now = time.time()
        with self._db_lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'failed',
                        stage = 'failed',
                        error_code = 'service_restart',
                        error_message = 'Job was interrupted by a service restart and was not requeued.',
                        worker_pid = NULL,
                        updated_at = ?
                    WHERE status = 'processing';
                    """,
                    (now,),
                )
                recovered = int(cursor.rowcount or 0)
            finally:
                conn.close()
        if recovered > 0:
            logger.warning("Marked %d unfinished job(s) as failed on startup recovery.", recovered)
        return recovered

    def start(self) -> None:
        """Recover unfinished jobs and start the single GPU analysis worker thread."""
        with self._db_lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                if self._stop_event.is_set():
                    raise RuntimeError('Previous analysis worker is still shutting down.')
                return
            self.recover_unfinished_jobs()
            self._stop_event.clear()
            self._wake_event.clear()
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name="forte-gpu-worker",
                daemon=True,
            )
            self._worker_thread.start()
            logger.info("Started Forte GPU analysis worker thread.")

    def stop(self) -> None:
        """Stop the worker thread and terminate any active child analyzer process."""
        self._stop_event.set()
        self._wake_event.set()
        self._terminate_active_subprocess()
        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, self.settings.subprocess_join_timeout_seconds * 2.0))
        self._terminate_active_subprocess()
        if thread is not None and thread.is_alive():
            logger.warning('Analysis supervisor is still shutting down; restart is blocked until it exits.')
        else:
            self._worker_thread = None
            logger.info("Stopped Forte GPU analysis worker thread.")

    def is_worker_alive(self) -> bool:
        thread = self._worker_thread
        return bool(thread is not None and thread.is_alive() and not self._stop_event.is_set())

    def is_worker_ready(self) -> bool:
        return self.is_worker_alive() and self._worker_ready.is_set()

    def get_active_job_count(self) -> int:
        """Return number of jobs currently in 'processing' state (queued + running)."""
        with self._db_lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM jobs WHERE status = 'processing';"
                ).fetchone()
                return int(row["cnt"]) if row is not None else 0
            finally:
                conn.close()

    def has_capacity(self) -> bool:
        return self.get_active_job_count() < self.settings.max_queue_size

    def create_job_Id(self) -> str:
        return uuid.uuid4().hex

    def enqueue_job(self, job_id: str, video_path: Path, created_at: float | None = None) -> JobRecord:
        """Atomically persist a new job in SQLite if queue capacity allows."""
        now = time.time() if created_at is None else float(created_at)
        deadline_at = now + self.settings.job_deadline_seconds

        with self._db_lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE;")
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM jobs WHERE status = 'processing';"
                ).fetchone()
                active_count = int(row["cnt"]) if row is not None else 0
                if active_count >= self.settings.max_queue_size:
                    conn.execute("ROLLBACK;")
                    raise QueueFullError(
                        f"Analysis queue is full (capacity={self.settings.max_queue_size})."
                    )

                conn.execute(
                    """
                    INSERT INTO jobs (
                        job_id, status, stage, progress, created_at, updated_at,
                        deadline_at, video_path, verdict, violations_json,
                        error_code, error_message, worker_pid
                    ) VALUES (?, 'processing', 'queued', 0.0, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL);
                    """,
                    (job_id, now, now, deadline_at, str(video_path.resolve())),
                )
                conn.execute("COMMIT;")
            except Exception:
                if conn.in_transaction:
                    conn.execute("ROLLBACK;")
                raise
            finally:
                conn.close()

        self._wake_event.set()
        logger.info("Enqueued job %s (deadline in %.2fs).", job_id, self.settings.job_deadline_seconds)
        return self.get_job_record(job_id)

    def get_job_record(self, job_id: str) -> JobRecord:
        with self._db_lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE job_id = ?;",
                    (job_id,),
                ).fetchone()
            finally:
                conn.close()

        if row is None:
            raise JobNotFoundError(job_id)
        return self._row_to_record(row)

    def get_job_response(
        self, job_id: str
    ) -> JobProcessingResponse | JobDoneResponse | JobFailedResponse:
        """Build a schema-validated response object for GET /api/v1/jobs/{job_id}."""
        record = self.get_job_record(job_id)

        if record.status == "processing":
            now = time.time()
            if now >= record.deadline_at and record.stage == "queued":
                self._mark_failed(
                    job_id=record.job_id,
                    error_code="job_timeout",
                    error_message="Job exceeded the end-to-end deadline while waiting in queue.",
                )
                record = self.get_job_record(job_id)
            else:
                return JobProcessingResponse(
                    job_id=record.job_id,
                    status="processing",
                    stage=record.stage or "queued",
                    progress=0.0 if record.progress is None else max(0.0, min(1.0, float(record.progress))),
                )

        if record.status == "done":
            try:
                violations_raw = json.loads(record.violations_json or "[]")
                validated = AnalyzerOutput.model_validate(
                    {"verdict": record.verdict, "violations": violations_raw}
                )
                return JobDoneResponse(
                    job_id=record.job_id,
                    status="done",
                    verdict=validated.verdict,
                    violations=validated.violations,
                )
            except Exception as exc:
                logger.error("Corrupted done record for job %s: %s", job_id, type(exc).__name__)
                self._mark_failed(
                    job_id=record.job_id,
                    error_code="invalid_analyzer_output",
                    error_message="Stored analysis result failed schema validation.",
                )
                record = self.get_job_record(job_id)

        return JobFailedResponse(
            job_id=record.job_id,
            status="failed",
            error_code=record.error_code or "analysis_failed",
            error=PUBLIC_FAILURE_MESSAGES.get(record.error_code or '', 'Local video analysis failed.'),
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=str(row["job_id"]),
            status=str(row["status"]),
            stage=str(row["stage"]) if row["stage"] is not None else None,
            progress=float(row["progress"]) if row["progress"] is not None else None,
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            deadline_at=float(row["deadline_at"]),
            video_path=Path(str(row["video_path"])),
            verdict=str(row["verdict"]) if row["verdict"] is not None else None,
            violations_json=str(row["violations_json"]) if row["violations_json"] is not None else None,
            error_code=str(row["error_code"]) if row["error_code"] is not None else None,
            error_message=str(row["error_message"]) if row["error_message"] is not None else None,
            worker_pid=int(row["worker_pid"]) if row["worker_pid"] is not None else None,
        )

    def _fetch_next_queued_job(self) -> JobRecord | None:
        with self._db_lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT * FROM jobs
                    WHERE status = 'processing'
                    ORDER BY created_at ASC, rowid ASC
                    LIMIT 1;
                    """
                ).fetchone()
            finally:
                conn.close()
        return self._row_to_record(row) if row is not None else None

    def _update_progress(
        self,
        job_id: str,
        stage: str,
        progress: float,
        worker_pid: int | None = None,
    ) -> None:
        now = time.time()
        clamped = max(0.0, min(1.0, float(progress)))
        with self._db_lock:
            conn = self._connect()
            try:
                if worker_pid is not None:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET stage = ?, progress = ?, worker_pid = ?, updated_at = ?
                        WHERE job_id = ? AND status = 'processing';
                        """,
                        (stage, clamped, int(worker_pid), now, job_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE jobs
                        SET stage = ?, progress = ?, updated_at = ?
                        WHERE job_id = ? AND status = 'processing';
                        """,
                        (stage, clamped, now, job_id),
                    )
            finally:
                conn.close()

    def _mark_done(self, job_id: str, validated: AnalyzerOutput) -> None:
        violations_payload = [
            item.model_dump(mode="json") for item in validated.violations
        ]
        serialized = json.dumps(violations_payload, ensure_ascii=False)
        completed = False
        with self._db_lock:
            conn = self._connect()
            try:
                now = time.time()
                cursor = conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'done',
                        stage = 'completed',
                        progress = 1.0,
                        verdict = ?,
                        violations_json = ?,
                        error_code = NULL,
                        error_message = NULL,
                        worker_pid = NULL,
                        updated_at = ?
                    WHERE job_id = ? AND status = 'processing' AND deadline_at > ?;
                    """,
                    (
                        validated.verdict,
                        serialized,
                        now,
                        job_id,
                        now,
                    ),
                )
                completed = cursor.rowcount == 1
            finally:
                conn.close()
        if completed:
            logger.info("Job %s completed with verdict=%s.", job_id, validated.verdict)
        else:
            record = self.get_job_record(job_id)
            if record.status == 'processing' and time.time() >= record.deadline_at:
                self._mark_failed(job_id, 'job_timeout', 'Analysis result missed the end-to-end deadline.')

    def _mark_failed(self, job_id: str, error_code: str, error_message: str) -> None:
        now = time.time()
        safe_message = sanitize_diagnostic_text(error_message, fallback="Video analysis failed.")
        with self._db_lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = 'failed',
                        stage = 'failed',
                        verdict = NULL,
                        violations_json = NULL,
                        error_code = ?,
                        error_message = ?,
                        worker_pid = NULL,
                        updated_at = ?
                    WHERE job_id = ?;
                    """,
                    (error_code, safe_message, now, job_id),
                )
            finally:
                conn.close()
        logger.warning("Job %s failed (%s): %s", job_id, error_code, safe_message)

    def _terminate_process(self, proc: multiprocessing.Process | None) -> None:
        if proc is None:
            return
        try:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=min(0.25, self.settings.subprocess_join_timeout_seconds))
            if proc.is_alive() and hasattr(proc, "kill"):
                proc.kill()
                proc.join(timeout=self.settings.subprocess_join_timeout_seconds)
        except Exception as exc:
            logger.warning("Error terminating analyzer subprocess: %s", type(exc).__name__)

    def _terminate_active_subprocess(self) -> None:
        with self._proc_lock:
            proc, connection = self._current_process, self._worker_connection
            self._current_process = None
            self._worker_connection = None
            self._worker_started_at = None
            self._worker_ready.clear()
        self._terminate_process(proc)
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def _ensure_process(self) -> tuple[multiprocessing.Process, Connection]:
        with self._proc_lock:
            proc, connection = self._current_process, self._worker_connection
        if proc is not None and proc.is_alive() and connection is not None:
            return proc, connection
        self._terminate_active_subprocess()
        parent_conn, child_conn = self._mp_ctx.Pipe(duplex=True)
        proc = self._mp_ctx.Process(target=_subprocess_worker_entry,
            args=(self._analyzer_factory, child_conn), name='forte-analysis', daemon=True)
        try:
            proc.start()
        except Exception:
            parent_conn.close()
            child_conn.close()
            raise
        child_conn.close()
        with self._proc_lock:
            if self._stop_event.is_set():
                self._terminate_process(proc)
                parent_conn.close()
                raise RuntimeError('Service stopped while spawning the analyzer.')
            self._current_process = proc
            self._worker_connection = parent_conn
            self._worker_started_at = time.monotonic()
        return proc, parent_conn

    def _warmup_timed_out(self) -> bool:
        with self._proc_lock:
            started = self._worker_started_at
        return bool(not self._worker_ready.is_set() and started is not None
            and time.monotonic() - started >= self.settings.worker_start_timeout_seconds)

    def _observe_idle_worker(self, connection: Connection) -> None:
        while connection.poll(0):
            message = connection.recv()
            if message.get('kind') == 'ready':
                self._worker_ready.set()
            elif message.get('kind') == 'startup_error':
                logger.error('Analyzer warmup failed: %s', message.get('message', 'unknown'))
                self._terminate_active_subprocess()
                self._restart_not_before = time.monotonic() + 5.0
                return

    def _worker_loop(self) -> None:
        poll_interval = self.settings.worker_poll_interval_seconds
        while not self._stop_event.is_set():
            job = None
            try:
                job = self._fetch_next_queued_job()
                if job is not None and time.time() >= job.deadline_at:
                    self._mark_failed(job.job_id, 'job_timeout',
                        'Job exceeded the end-to-end deadline while waiting in queue.')
                    continue
                if time.monotonic() < self._restart_not_before:
                    self._wake_event.wait(timeout=poll_interval)
                    self._wake_event.clear()
                    continue
                proc, connection = self._ensure_process()
                if job is None:
                    self._observe_idle_worker(connection)
                    if self._warmup_timed_out():
                        logger.error('Analyzer warmup timed out; replacing the idle worker.')
                        self._terminate_active_subprocess()
                        self._restart_not_before = time.monotonic() + 0.1
                    self._wake_event.wait(timeout=poll_interval)
                    self._wake_event.clear()
                    continue
                if not job.video_path.is_file():
                    self._mark_failed(job.job_id, 'missing_video_file',
                        'Uploaded video file is no longer available for analysis.')
                    continue
                self._execute_job_in_subprocess(job, proc, connection)
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger.error('Analysis worker failure: %s', type(exc).__name__)
                if job is not None:
                    self._mark_failed(job.job_id, 'worker_crashed',
                        'Analysis worker stopped unexpectedly; a replacement is starting.')
                self._terminate_active_subprocess()
                self._restart_not_before = time.monotonic() + 0.1

    def _execute_job_in_subprocess(self, job: JobRecord,
        proc: multiprocessing.Process, connection: Connection) -> None:
        self._update_progress(job.job_id,
            stage='starting' if self._worker_ready.is_set() else 'warming_up',
            progress=0.0, worker_pid=proc.pid)
        connection.send({'kind': 'analyze', 'job_id': job.job_id,
            'video_path': str(job.video_path)})
        terminal_message = None
        while not self._stop_event.is_set():
            remaining = job.deadline_at - time.time()
            if remaining <= 0:
                self._terminate_active_subprocess()
                self._mark_failed(job.job_id, 'job_timeout',
                    'Analysis exceeded its deadline; the worker was terminated and will restart.')
                return
            if self._warmup_timed_out():
                self._terminate_active_subprocess()
                self._mark_failed(job.job_id, 'worker_crashed',
                    'Analyzer warmup exceeded its deadline; the worker will restart.')
                return
            if connection.poll(min(0.05, remaining)):
                message = connection.recv()
                if not isinstance(message, dict):
                    continue
                kind = message.get('kind')
                if kind == 'ready':
                    self._worker_ready.set()
                    self._update_progress(job.job_id, stage='starting', progress=0.0)
                    continue
                if kind == 'startup_error':
                    terminal_message = {'kind': 'error', 'message': message.get('message')}
                    self._terminate_active_subprocess()
                    self._restart_not_before = time.monotonic() + 5.0
                    break
                if message.get('job_id') != job.job_id:
                    continue
                if kind == 'progress':
                    self._update_progress(job.job_id, str(message.get('stage') or 'analyzing'),
                        float(message.get('progress') or 0.0))
                elif kind in {'result', 'error'}:
                    terminal_message = message
                    break
            if not proc.is_alive():
                self._worker_ready.clear()
                break
        if self._stop_event.is_set():
            return
        if time.time() >= job.deadline_at:
            self._terminate_active_subprocess()
            self._mark_failed(job.job_id, 'job_timeout', 'Analysis exceeded its end-to-end deadline.')
            return
        self._finalize_job_outcome(job.job_id, terminal_message, proc.exitcode)

    def _finalize_job_outcome(
        self,
        job_id: str,
        terminal_message: dict[str, Any] | None,
        exit_code: int | None,
    ) -> None:
        if terminal_message is None:
            self._mark_failed(
                job_id=job_id,
                error_code="worker_crashed",
                error_message=f"Worker subprocess exited unexpectedly (code={exit_code}).",
            )
            return

        kind = terminal_message.get("kind")
        if kind == "error":
            self._mark_failed(
                job_id=job_id,
                error_code="analyzer_error",
                error_message=str(terminal_message.get("message") or "Analyzer raised an exception."),
            )
            return

        if kind == "result":
            payload = terminal_message.get("payload")
            try:
                validated = AnalyzerOutput.model_validate(payload)
            except (ValidationError, ValueError, TypeError) as exc:
                self._mark_failed(
                    job_id=job_id,
                    error_code="invalid_analyzer_output",
                    error_message="Analyzer returned invalid output schema.",
                )
                return
            self._mark_done(job_id=job_id, validated=validated)
            return

        self._mark_failed(
            job_id=job_id,
            error_code="unknown_worker_state",
            error_message="Worker finished in an unrecognized state.",
        )
