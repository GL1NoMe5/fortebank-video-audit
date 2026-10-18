"""Own and supervise one local API and its optional temporary HTTPS tunnel.

This command starts a public tunnel. Use --self-test for synthetic checks only.
Secrets are passed in the API child environment, never command-line arguments.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import queue
import re
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

import psutil

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
DEADLINE = datetime.fromisoformat("2026-10-19T00:00:00+05:00")
URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com(?![a-z0-9.-])", re.IGNORECASE)
STOP_NAME = "serve.stop"
STATE_NAME = "service_state.json"
API_HTTP_STARTUP_GRACE = 120.0
API_HTTP_UNRESPONSIVE_LIMIT = 60.0
CONNECTION_INDEX = re.compile(r"\bconnIndex=(\d+)\b")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError):
        return {}


class InstanceLock:
    """Kernel-backed lock: stale files do not prevent a subsequent start."""

    def __init__(self, path: Path):
        self.path, self.handle = path, None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.handle.write(b"1")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            raise RuntimeError("Another service supervisor owns this workspace lock") from None
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


def secret_key(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(secrets.token_urlsafe(32) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if os.name != "nt":
                path.chmod(0o600)
        except FileExistsError:
            pass
    value = path.read_text(encoding="utf-8").strip()
    if len(value) < 32 or any(character.isspace() for character in value):
        raise ValueError("Existing API key file is invalid; it was not replaced")
    return value


def command_fingerprint(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command, ensure_ascii=True).encode("utf-8")).hexdigest()


def identity(pid: int) -> dict:
    process = psutil.Process(pid)
    return {"pid": pid, "created_at": process.create_time(), "executable": process.exe(), "command_sha256": command_fingerprint(process.cmdline())}


def validated_process(value: dict) -> psutil.Process | None:
    try:
        pid = int(value["pid"])
        if pid <= 0:
            return None
        process = psutil.Process(pid)
        if abs(process.create_time() - float(value["created_at"])) > 0.01:
            return None
        if os.path.normcase(process.exe()) != os.path.normcase(value["executable"]):
            return None
        if command_fingerprint(process.cmdline()) != value["command_sha256"]:
            return None
        return process
    except (KeyError, TypeError, ValueError, psutil.Error, OSError):
        return None


def owns_listener(value: dict, port: int) -> bool:
    parent = validated_process(value)
    if parent is None:
        return False
    try:
        candidates = [parent, *parent.children(recursive=True)]
        for process in candidates:
            try:
                if any(connection.status == psutil.CONN_LISTEN and connection.laddr.port == port and connection.laddr.ip == "127.0.0.1" for connection in process.net_connections(kind="tcp")):
                    return True
            except psutil.Error:
                continue
    except psutil.Error:
        pass
    return False


def port_occupied(port: int) -> bool:
    with socket.socket() as connection:
        connection.settimeout(0.5)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def terminate_owned(value: dict, timeout: float = 8) -> bool:
    """Never terminate a reused PID; include only currently proven descendants."""
    parent = validated_process(value)
    if parent is None or parent.pid == os.getpid():
        return False
    owned = []
    try:
        for process in parent.children(recursive=True):
            try:
                owned.append(identity(process.pid))
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    owned.append(value)
    waiting = []
    for item in reversed(owned):
        process = validated_process(item)
        if process is not None:
            try:
                process.terminate()
                waiting.append(process)
            except psutil.Error:
                pass
    _, alive = psutil.wait_procs(waiting, timeout=timeout)
    alive_ids = {process.pid for process in alive}
    for item in owned:
        if item["pid"] in alive_ids:
            process = validated_process(item)
            if process is not None:
                try:
                    process.kill()
                except psutil.Error:
                    pass
    return True


def backoff(failures: int) -> float:
    return float(min(30, 2 ** min(max(failures - 1, 0), 5)))


def tunnel_connection_event(line: str) -> tuple[str, str | None] | None:
    """Track edge connections, not origin errors or mere process liveness."""
    match = CONNECTION_INDEX.search(line)
    connection = match.group(1) if match else None
    if "Registered tunnel connection" in line:
        return "connected", connection or "unknown"
    if any(message in line for message in (
        "Unregistered tunnel connection", "Connection terminated",
        "Failed to serve tunnel connection", "Serve tunnel error",
        "Retrying connection in up to", "Tunnel server stopped",
        "all connections terminated",
    )):
        return "disconnected", connection
    return None


def scrub(line: str, key: str) -> str:
    value = line.replace(key, "[REDACTED_API_KEY]") if key else line
    if any(marker in value for marker in ('"image_base64"', '"violations"', '"video_path"', '"description"')):
        return "[private analysis payload suppressed]"
    return value.rstrip()[:16000]


def logger_for(name: str, path: Path) -> logging.Logger:
    logger = logging.getLogger(f"forte.supervisor.{name}.{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    return logger


def hidden_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class OwnedJob:
    """Windows closes the entire owned process tree if the supervisor dies."""

    def __init__(self, pid: int):
        self.handle = None
        if os.name != "nt":
            return
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("per_process_time", ctypes.c_longlong), ("per_job_time", ctypes.c_longlong), ("flags", wintypes.DWORD), ("minimum_working_set", ctypes.c_size_t), ("maximum_working_set", ctypes.c_size_t), ("active_process_limit", wintypes.DWORD), ("affinity", ctypes.c_size_t), ("priority_class", wintypes.DWORD), ("scheduling_class", wintypes.DWORD)]
        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", IoCounters), ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel = kernel
        job = kernel.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = ExtendedLimits()
            limits.basic.flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            process = kernel.OpenProcess(0x0100 | 0x0001, False, pid)  # SET_QUOTA | TERMINATE
            if not process:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if not kernel.AssignProcessToJobObject(job, process):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                kernel.CloseHandle(process)
            self.handle = job
        except BaseException:
            kernel.CloseHandle(job)
            raise

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class AwakeGuard:
    def __init__(self):
        self.active = False

    def refresh(self, now: datetime | None = None) -> bool:
        wanted = (now or datetime.now(timezone.utc)) < DEADLINE
        if os.name == "nt":
            flags = 0x80000000 | (0x00000001 if wanted else 0)
            success = ctypes.windll.kernel32.SetThreadExecutionState(flags)
            if not success:
                raise OSError("SetThreadExecutionState failed")
            self.active = wanted
        return self.active

    def release(self):
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        self.active = False


class Supervisor:
    def __init__(self, root: Path, key: str):
        self.root, self.key = root, key
        self.runtime = root / "runtime"
        self.instance = secrets.token_hex(16)
        self.me = identity(os.getpid())
        self.stop_event = threading.Event()
        self.events = queue.Queue()
        self.awake = AwakeGuard()
        self.api_ready = False
        self.api_http_responsive = False
        self.api_unresponsive_since = None
        self.url = None
        self.tunnel_connected = False
        self.tunnel_connections: set[str] = set()
        self.endpoint_generation = 0
        self.state_status = "starting"
        self.logs = self.runtime / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.log = logger_for("supervisor", self.logs / "supervisor.log")
        self.child_logs = {name: logger_for(name, self.logs / f"{name}.log") for name in ("api", "cloudflared")}
        self.children = {name: {"process": None, "identity": None, "failures": 0, "next_start": 0.0, "started": 0.0, "reader": None, "job": None} for name in ("api", "cloudflared")}
        self.python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.cloudflared = root / "tools" / "cloudflared" / ("cloudflared.exe" if os.name == "nt" else "cloudflared")
        self.api_marker = self.runtime / "control" / f"api-stop-{self.instance}.json"
        self.opener = build_opener(ProxyHandler({}))

    def event(self, event: str, **fields):
        self.log.info(json.dumps({"event": event, **fields}, ensure_ascii=True))

    def state(self):
        atomic_json(self.runtime / STATE_NAME, {
            "schema_version": 1, "instance": self.instance, "supervisor": self.me,
            "status": self.state_status, "updated_at": timestamp(), "api_ready": self.api_ready,
            "api_http_responsive": self.api_http_responsive,
            "api_unresponsive_seconds": round(time.monotonic() - self.api_unresponsive_since, 1) if self.api_unresponsive_since is not None else 0.0,
            "public_url": self.url, "tunnel_connected": self.tunnel_connected,
            "tunnel_status_source": "cloudflared_connection_events",
            "sleep_prevention_until": DEADLINE.isoformat(), "sleep_prevention_active": self.awake.active,
            "children": {name: {"identity": entry["identity"], "running": entry["process"] is not None and entry["process"].poll() is None, "failures": entry["failures"]} for name, entry in self.children.items()},
        })
        if (
            not self.stop_event.is_set()
            and self.state_status == "running"
            and self.api_ready
            and self.tunnel_connected
            and self.url
            and self.api_http_responsive
            and self.api_unresponsive_since is None
            and read_json(self.runtime / "public_endpoint.json").get("status") == "connected"
        ):
            self.endpoint("connected")

    def endpoint(self, status: str):
        atomic_json(self.runtime / "public_endpoint.json", {
            "schema_version": 1, "url": self.url, "updated_at": timestamp(), "status": status,
            "instance": self.instance, "generation": self.endpoint_generation,
            "api_ready": self.api_ready, "tunnel_connected": self.tunnel_connected, "temporary": True,
            "connection_status_source": "cloudflared_connection_events",
        })

    def read_child(self, name: str, process: subprocess.Popen):
        assert process.stdout is not None
        try:
            for line in iter(lambda: process.stdout.readline(262144), ""):
                safe = scrub(line, self.key)
                if safe:
                    self.child_logs[name].info(safe)
                if name == "cloudflared":
                    match = URL_PATTERN.search(line)
                    if match:
                        self.events.put((process.pid, "url", match.group(0)))
                    connection_event = tunnel_connection_event(line)
                    if connection_event is not None:
                        self.events.put((process.pid, *connection_event))
        finally:
            process.stdout.close()

    def start(self, name: str):
        entry = self.children[name]
        env = os.environ.copy()
        env.pop("FORTE_API_KEY", None)
        env.update({"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
        if name == "api":
            env.update({
                "FORTE_API_KEY": self.key, "FORTE_SERVICE_INSTANCE": self.instance,
                "FORTE_ROOT_DIR": str(self.root), "FORTE_RUNTIME_DIR": str(self.runtime),
                "FORTE_WEB_DIR": str(self.root / "web"),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            })
            env.setdefault("FORTE_MAX_UPLOAD_BYTES", "94371840")  # 90 MiB leaves multipart headroom for the tunnel.
            env["PATH"] = str(self.root / "tools" / "ffmpeg" / "bin") + os.pathsep + env.get("PATH", "")
            command = [str(self.python), "-u", str(self.root / "scripts" / "serve.py"), "--api-child"]
        else:
            command = [str(self.cloudflared), "tunnel", "--url", "http://127.0.0.1:8000", "--protocol", "http2", "--no-autoupdate"]
        try:
            flags = hidden_flags() | (0x00000004 if os.name == "nt" else 0)  # CREATE_SUSPENDED
            process = subprocess.Popen(command, cwd=self.root, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=flags)
            entry["job"] = OwnedJob(process.pid)
            entry.update(process=process, identity=identity(process.pid), started=time.monotonic())
            if name == "api":
                self.api_unresponsive_since = None
                self.api_http_responsive = False
            else:
                self.tunnel_connections.clear()
                self.tunnel_connected = False
            if os.name == "nt":
                psutil.Process(process.pid).resume()
            reader = threading.Thread(target=self.read_child, args=(name, process), daemon=True)
            entry["reader"] = reader
            reader.start()
            self.event("child_started", child=name, pid=process.pid)
        except (OSError, psutil.Error) as exc:
            # If PID inspection races with an immediate exit, keep/clean only the handle we just created.
            if "process" in locals() and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            if entry["job"] is not None:
                entry["job"].close()
                entry["job"] = None
            entry.update(process=None, identity=None)
            entry["failures"] += 1
            entry["next_start"] = time.monotonic() + backoff(entry["failures"])
            self.event("child_start_failed", child=name, error_type=type(exc).__name__, retry_seconds=backoff(entry["failures"]))

    def poll_children(self):
        now = time.monotonic()
        for name, entry in self.children.items():
            process = entry["process"]
            if process is not None and process.poll() is not None:
                if now - entry["started"] >= 120:
                    entry["failures"] = 0
                entry["failures"] += 1
                entry["next_start"] = now + backoff(entry["failures"])
                self.event("child_exited", child=name, exit_code=process.returncode, retry_seconds=backoff(entry["failures"]))
                if entry["job"] is not None:
                    entry["job"].close()
                    entry["job"] = None
                entry.update(process=None, identity=None)
                if name == "api":
                    self.api_ready = False
                    self.api_http_responsive = False
                    self.api_unresponsive_since = None
                    if self.url:
                        self.endpoint("degraded")
                else:
                    self.url, self.tunnel_connected = None, False
                    self.tunnel_connections.clear()
                    self.endpoint("restarting")
            if entry["process"] is None and now >= entry["next_start"]:
                if name == "api" or self.api_ready:
                    self.start(name)

    def api_http_watchdog_due(self, responsive: bool, now: float) -> bool:
        """Readiness is intentionally absent: warmup belongs to the backend."""
        self.api_http_responsive = responsive
        entry = self.children["api"]
        process = entry["process"]
        if responsive or process is None or process.poll() is not None:
            self.api_unresponsive_since = None
            return False
        if self.api_unresponsive_since is None:
            self.api_unresponsive_since = now
        return (now - entry["started"] >= API_HTTP_STARTUP_GRACE
                and now - self.api_unresponsive_since >= API_HTTP_UNRESPONSIVE_LIMIT)

    def health(self):
        ready = False
        responsive = False
        owned = owns_listener(self.children["api"]["identity"] or {}, 8000)
        if owned:
            try:
                with self.opener.open("http://127.0.0.1:8000/healthz", timeout=2) as response:
                    body = response.read(16384)
                    responsive = True
                    payload = json.loads(body)
                    ready = response.status == 200 and isinstance(payload, dict) and payload.get("ready") is True
            except HTTPError as exc:
                # A completed HTTP error is responsive, although not ready.
                responsive = True
                exc.close()
            except (OSError, ValueError):
                pass
        elif port_occupied(8000):
            # A foreign listener must never be published after an API crash or port race.
            tunnel = self.children["cloudflared"]
            if tunnel["process"] is not None:
                terminate_owned(tunnel["identity"] or {})
                if tunnel["job"] is not None:
                    tunnel["job"].close()
                    tunnel["job"] = None
                tunnel.update(process=None, identity=None, next_start=time.monotonic() + 1)
                self.url, self.tunnel_connected = None, False
                self.tunnel_connections.clear()
                self.endpoint("foreign_port_owner")
                self.event("tunnel_closed_for_foreign_port_owner")
        if ready != self.api_ready:
            self.api_ready = ready
            self.event("api_readiness_changed", ready=ready)
            if self.url:
                self.endpoint("connected" if self.tunnel_connected and ready else "degraded")
        now = time.monotonic()
        if self.api_http_watchdog_due(responsive, now):
            self.event("api_http_unresponsive_restart", seconds=round(now - self.api_unresponsive_since, 1))
            self.api_unresponsive_since = now
            api = self.children["api"]
            terminate_owned(api["identity"] or {}, timeout=2)
            if api["job"] is not None:
                api["job"].close()
                api["job"] = None
            # poll_children performs the bounded-backoff restart. The existing
            # cloudflared process and URL remain untouched during recovery.

    def pending_events(self):
        while True:
            try:
                pid, kind, value = self.events.get_nowait()
            except queue.Empty:
                return
            process = self.children["cloudflared"]["process"]
            if process is None or process.pid != pid or process.poll() is not None:
                continue
            if kind == "url" and value != self.url:
                self.url = value
                self.endpoint_generation += 1
                self.endpoint("connected" if self.tunnel_connected and self.api_ready else "degraded" if self.tunnel_connected else "connecting")
                self.event("public_endpoint_changed", url=self.url)
            elif kind in ("connected", "disconnected"):
                if kind == "connected":
                    self.tunnel_connections.add(value or "unknown")
                elif value is None:
                    self.tunnel_connections.clear()
                else:
                    self.tunnel_connections.discard(value)
                    self.tunnel_connections.discard("unknown")
                previous = self.tunnel_connected
                self.tunnel_connected = bool(self.tunnel_connections)
                if previous != self.tunnel_connected:
                    self.event("tunnel_connection_state_changed", connected=self.tunnel_connected)
                    self.endpoint("connected" if self.tunnel_connected and self.api_ready else "degraded" if self.tunnel_connected else "reconnecting")

    def stop_requested(self) -> bool:
        request = read_json(self.runtime / STOP_NAME)
        return request.get("instance") == self.instance and request.get("pid") == os.getpid()

    def recover_previous(self):
        previous = read_json(self.runtime / STATE_NAME)
        previous_instance = previous.get("instance", "")
        if not re.fullmatch(r"[a-f0-9]{32}", previous_instance):
            return
        if validated_process(previous.get("supervisor", {})) is not None:
            raise RuntimeError("A validated supervisor is already running")
        entries = previous.get("children", {})
        old_api = entries.get("api", {}).get("identity") or {}
        old_tunnel = entries.get("cloudflared", {}).get("identity") or {}
        if validated_process(old_api) is not None:
            atomic_json(self.runtime / "control" / f"api-stop-{previous_instance}.json", {"instance": previous_instance})
            until = time.monotonic() + 15
            while validated_process(old_api) is not None and time.monotonic() < until:
                time.sleep(0.25)
        for name, value in (("cloudflared", old_tunnel), ("api", old_api)):
            if terminate_owned(value):
                self.event("previous_owned_child_stopped", child=name, pid=value["pid"])

    def shutdown(self):
        self.state_status = "stopping"
        marker_written = False
        try:
            try:
                self.state()
                atomic_json(self.api_marker, {"instance": self.instance})
                marker_written = True
            except OSError as exc:
                self.event("shutdown_state_write_failed", error_type=type(exc).__name__)
            # Close public access first; API gets its own graceful shutdown marker.
            terminate_owned(self.children["cloudflared"]["identity"] or {})
            api = self.children["api"]["identity"] or {}
            until = time.monotonic() + (25 if marker_written else 0)
            while validated_process(api) is not None and time.monotonic() < until:
                time.sleep(0.25)
            terminate_owned(api)
        finally:
            # Disk failures and cleanup errors must not leak GPU workers or the awake request.
            try:
                for entry in self.children.values():
                    if entry["job"] is not None:
                        entry["job"].close()
                        entry["job"] = None
                    process = entry["process"]
                    if process is not None:
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            terminate_owned(entry["identity"] or {}, timeout=2)
            finally:
                self.awake.release()
        self.api_ready, self.tunnel_connected, self.url = False, False, None
        self.api_http_responsive, self.api_unresponsive_since = False, None
        self.tunnel_connections.clear()
        self.state_status = "stopped"
        try:
            self.endpoint("stopped")
            self.state()
        except OSError as exc:
            self.event("final_state_write_failed", error_type=type(exc).__name__)
        self.event("supervisor_stopped")

    def run(self):
        if not self.python.is_file() or not self.cloudflared.is_file():
            raise FileNotFoundError("Project venv Python or tools/cloudflared binary is missing")
        self.recover_previous()
        if port_occupied(8000):
            raise RuntimeError("Port 8000 is occupied by a process not owned by this supervisor")
        self.endpoint("waiting_for_api")
        for name in ("SIGINT", "SIGTERM"):
            signal.signal(getattr(signal, name), lambda *_: self.stop_event.set())
        self.event("supervisor_started", pid=os.getpid(), sleep_prevention_until=DEADLINE.isoformat())
        last_health = last_awake = last_state = 0.0
        try:
            while not self.stop_event.is_set() and not self.stop_requested():
                now = time.monotonic()
                if now - last_awake >= 30:
                    try:
                        self.awake.refresh()
                    except OSError as exc:
                        self.event("sleep_prevention_failed", error_type=type(exc).__name__)
                    last_awake = now
                self.poll_children()
                if now - last_health >= 2:
                    self.health()
                    last_health = now
                self.pending_events()
                self.state_status = "running" if self.api_ready and self.tunnel_connected and self.url else "waiting_for_api" if not self.api_ready else "connecting_tunnel"
                if now - last_state >= 5:
                    self.state()
                    last_state = now
                self.stop_event.wait(0.5)
        finally:
            self.shutdown()


def api_child() -> None:
    """One Uvicorn worker, with an explicit cooperative shutdown channel."""
    instance = os.environ.get("FORTE_SERVICE_INSTANCE", "")
    if not re.fullmatch(r"[a-f0-9]{32}", instance) or not os.environ.get("FORTE_API_KEY"):
        raise RuntimeError("API child must be started by the supervisor")
    sys.path.insert(0, str(ROOT))
    import uvicorn

    marker = ROOT / "runtime" / "control" / f"api-stop-{instance}.json"
    server = uvicorn.Server(uvicorn.Config("forte.api:app", host="127.0.0.1", port=8000, workers=1, access_log=False, proxy_headers=False, log_level="info", timeout_graceful_shutdown=20))
    def watch_stop():
        while not server.should_exit:
            if read_json(marker).get("instance") == instance:
                server.should_exit = True
                return
            time.sleep(0.5)
    threading.Thread(target=watch_stop, daemon=True).start()
    server.run()


def status_or_stop(stop: bool = False, expected_instance: str | None = None) -> int:
    value = read_json(ROOT / "runtime" / STATE_NAME)
    if expected_instance is not None and value.get("instance") != expected_instance:
        print("Supervisor instance changed; no stop request was written.")
        return 1
    process = validated_process(value.get("supervisor", {}))
    if process is None or value.get("status") == "stopped":
        print("No validated running supervisor; no process was stopped.")
        return 1
    if stop:
        atomic_json(ROOT / "runtime" / STOP_NAME, {"instance": value["instance"], "pid": process.pid, "requested_at": timestamp()})
        print("Graceful stop requested for the validated service supervisor.")
    else:
        print(json.dumps({"pid": process.pid, "status": value.get("status"), "api_ready": value.get("api_ready"), "url": value.get("public_url"), "updated_at": value.get("updated_at")}, ensure_ascii=True))
    return 0


def self_test() -> None:
    """No API, model, tunnel, real secret, sleep mutation, or media is accessed."""
    import io
    from unittest.mock import patch

    with tempfile.TemporaryDirectory(prefix="forte-supervisor-test-") as temporary:
        root = Path(temporary)
        with InstanceLock(root / "test.lock"):
            try:
                with InstanceLock(root / "test.lock"):
                    raise AssertionError("Duplicate instance acquired lock")
            except RuntimeError:
                pass
        with InstanceLock(root / "test.lock"):
            pass
        value = secret_key(root / "synthetic_key.txt")
        assert len(value) >= 32 and secret_key(root / "synthetic_key.txt") == value
        assert value not in scrub("Token=" + value, value)
        assert scrub('{"description":"synthetic"}', value) == "[private analysis payload suppressed]"
        assert [backoff(n) for n in range(1, 9)] == [1, 2, 4, 8, 16, 30, 30, 30]
        assert URL_PATTERN.search("URL https://test-only.trycloudflare.com available").group(0) == "https://test-only.trycloudflare.com"
        assert URL_PATTERN.search("https://test.trycloudflare.com.evil.invalid") is None
        atomic_json(root / "state.json", {"a": 1})
        atomic_json(root / "state.json", {"a": 2})
        assert read_json(root / "state.json") == {"a": 2}
        cleaned = []
        class FakeJob:
            def close(self):
                cleaned.append("job")
        class FakeAwake:
            def release(self):
                cleaned.append("awake")
        def disk_failure(*_):
            raise OSError("Synthetic disk failure")
        simulated = Supervisor.__new__(Supervisor)
        simulated.instance = "a" * 32
        simulated.api_marker = root / "synthetic-stop.json"
        simulated.children = {name: {"process": None, "identity": None, "job": FakeJob()} for name in ("api", "cloudflared")}
        simulated.awake = FakeAwake()
        simulated.tunnel_connections = set()
        simulated.state = disk_failure
        simulated.endpoint = disk_failure
        simulated.event = lambda *args, **kwargs: None
        simulated.shutdown()
        assert cleaned == ["job", "job", "awake"]
        class FakeProcess:
            def __init__(self, pid):
                self.pid = pid
            def poll(self):
                return None

        simulated = Supervisor.__new__(Supervisor)
        simulated.api_ready = False
        simulated.api_http_responsive = False
        simulated.api_unresponsive_since = None
        simulated.url = "https://synthetic-only.trycloudflare.com"
        simulated.endpoint_generation = 1
        simulated.tunnel_connections = set()
        simulated.tunnel_connected = False
        simulated.events = queue.Queue()
        endpoints, terminated = [], []
        simulated.endpoint = lambda status: endpoints.append(status)
        simulated.event = lambda *args, **kwargs: None
        simulated.children = {
            "api": {"process": FakeProcess(101), "identity": {"pid": 101}, "started": 0.0, "job": None},
            "cloudflared": {"process": FakeProcess(202), "identity": {"pid": 202}, "started": 0.0, "job": None},
        }
        assert not simulated.api_http_watchdog_due(False, 1)
        assert not simulated.api_http_watchdog_due(False, 61)  # Startup grace.
        assert simulated.api_http_watchdog_due(False, 121)
        assert not simulated.api_http_watchdog_due(True, 122)
        assert not simulated.api_http_watchdog_due(False, 123)
        assert not simulated.api_http_watchdog_due(False, 182)
        assert simulated.api_http_watchdog_due(False, 183)
        class FakeOpener:
            def __init__(self, body=b'{"ready":false}', error=None):
                self.body, self.error = body, error
            def open(self, *_args, **_kwargs):
                if self.error:
                    raise self.error
                response = io.BytesIO(self.body)
                response.status = 200
                return response

        with patch(__name__ + ".owns_listener", return_value=True), \
             patch(__name__ + ".terminate_owned", side_effect=lambda item, **kwargs: terminated.append(item["pid"])), \
             patch(__name__ + ".time.monotonic", return_value=300) as now:
            simulated.opener = FakeOpener()
            simulated.health()  # Arbitrarily long model warmup is still responsive.
            assert simulated.api_http_responsive and not simulated.api_ready and not terminated
            simulated.opener = FakeOpener(body=b"[]")
            simulated.health()  # Bad readiness schema is not a transport hang.
            assert simulated.api_http_responsive and not terminated
            simulated.opener = FakeOpener(error=HTTPError("http://127.0.0.1", 503, "synthetic", None, None))
            simulated.health()
            assert simulated.api_http_responsive and not terminated
            simulated.opener = FakeOpener(error=TimeoutError())
            simulated.health()
            now.return_value = 359
            simulated.health()
            assert not terminated
            now.return_value = 360
            simulated.health()
            assert terminated == [101]
            assert simulated.children["cloudflared"]["process"].pid == 202
            assert simulated.url == "https://synthetic-only.trycloudflare.com"

        assert tunnel_connection_event("Registered tunnel connection connIndex=0") == ("connected", "0")
        assert tunnel_connection_event("Unregistered tunnel connection connIndex=0") == ("disconnected", "0")
        assert tunnel_connection_event("Retrying connection in up to 1s connIndex=1") == ("disconnected", "1")
        assert tunnel_connection_event("Unable to reach the origin service connIndex=0") is None
        def connection_event(kind, value, pid=202):
            simulated.events.put((pid, kind, value))
            simulated.pending_events()
        connection_event("connected", "0")
        connection_event("connected", "1")
        connection_event("disconnected", "0")
        assert simulated.tunnel_connected and simulated.tunnel_connections == {"1"}
        connection_event("disconnected", "1")
        assert not simulated.tunnel_connected and endpoints[-1] == "reconnecting"
        assert simulated.url == "https://synthetic-only.trycloudflare.com" and simulated.endpoint_generation == 1
        connection_event("connected", "0", pid=999)  # Stale process events are ignored.
        assert not simulated.tunnel_connected
        connection_event("connected", "0")
        assert simulated.tunnel_connected and endpoints[-1] == "degraded"  # API is not ready.
        connection_event("disconnected", None)
        assert not simulated.tunnel_connected
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            assert owns_listener(identity(os.getpid()), listener.getsockname()[1])
        flags = hidden_flags() | (0x00000004 if os.name == "nt" else 0)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        job = None
        try:
            job = OwnedJob(child.pid)
            owned = identity(child.pid)
            if os.name == "nt":
                psutil.Process(child.pid).resume()
            assert validated_process(owned) is not None
            wrong = {**owned, "created_at": owned["created_at"] - 100}
            assert terminate_owned(wrong, timeout=0.1) is False and child.poll() is None
            assert terminate_owned(owned, timeout=2) is True
            child.wait(timeout=5)
        finally:
            if job is not None:
                job.close()
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
    print("Synthetic supervisor checks passed; no service or tunnel was launched.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--self-test", action="store_true")
    actions.add_argument("--api-child", action="store_true", help=argparse.SUPPRESS)
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--stop", action="store_true")
    parser.add_argument("--instance", help="Only stop this exact supervisor instance")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.api_child:
        api_child()
        return 0
    if args.status or args.stop:
        return status_or_stop(args.stop, args.instance)
    try:
        with InstanceLock(ROOT / "runtime" / "supervisor.lock"):
            key = secret_key(ROOT / "runtime" / "secrets" / "api_key.txt")
            Supervisor(ROOT, key).run()
    except Exception as exc:
        # Exception messages can contain environment or model output; keep console type-only.
        print(f"Supervisor stopped: {type(exc).__name__}. Inspect runtime/logs/supervisor.log.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
