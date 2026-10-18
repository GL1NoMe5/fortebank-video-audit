"""Reproducible local generator for the private jury submission package.

Reads runtime endpoint metadata and the secret API key at execution time,
ingests active local model download provenance and explicit evaluation
summary JSON files, loads the normative rule registry at runtime, and writes
the self-contained submission package strictly inside
``runtime/private_submission/``.

Security and honesty invariants:
- Never prints or logs the API key, endpoint URL, or report details to stdout.
- Never writes the package outside ``runtime/private_submission/``.
- Never invents missing/stale/disconnected endpoint URLs or missing API keys.
- Separates ``LOCAL_METADATA_READY`` from ``EXTERNAL_REACHABILITY_NOT_TESTED``:
  performs no network calls, no model loading, and no video file reads.
- Parses evaluation JSON containers to render only the 18 allowlisted aggregate
  numeric fields from ``summary``; discards ``records``, ``provenance.identity``,
  ``selected_ids``, ``selected_records``, and any per-video details.
- Validates canonical ``provenance.identity_sha256`` digests for internal
  metadata consistency without claiming weight re-hashing or rerun evaluations.
- Clearly separates ``train`` and ``holdout`` splits and incompatible model/
  provenance fingerprints without pooling or cherry-picking best scores.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

DEFAULT_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_MAX_ENDPOINT_AGE_SEC: float = 300.0

ALLOWLISTED_SUMMARY_FIELDS: tuple[str, ...] = (
    "videos",
    "completed",
    "failed",
    "tp",
    "fp",
    "tn",
    "fn",
    "precision",
    "recall",
    "accuracy",
    "completed_accuracy",
    "failed_positive",
    "mean_seconds",
    "max_seconds",
    "median_localization_error_sec",
    "localized_positive_count",
    "localized_within_3_sec",
    "over_300_seconds",
)

INTEGER_SUMMARY_FIELDS: frozenset[str] = frozenset(
    {
        "videos",
        "completed",
        "failed",
        "tp",
        "fp",
        "tn",
        "fn",
        "failed_positive",
        "localized_positive_count",
        "localized_within_3_sec",
        "over_300_seconds",
    }
)

PROBABILITY_SUMMARY_FIELDS: frozenset[str] = frozenset(
    {
        "precision",
        "recall",
        "accuracy",
        "completed_accuracy",
    }
)

DURATION_SUMMARY_FIELDS: frozenset[str] = frozenset(
    {
        "mean_seconds",
        "max_seconds",
        "median_localization_error_sec",
    }
)

PLACEHOLDER_MARKERS: tuple[str, ...] = (
    "placeholder",
    "example.com",
    "example.org",
    "your-api-key",
    "your_api_key",
    "changeme",
    "replace_me",
    "todo",
    "<api_key>",
    "<public_url>",
    "tbd",
)


def resolve_private_output_path(
    repo_root: Path,
    output_path: Path | str | None = None,
) -> Path:
    """Validate that the target output file resides inside runtime/private_submission."""
    root = repo_root.resolve()
    private_dir = (root / "runtime" / "private_submission").resolve()
    if output_path is None:
        target = private_dir / "SUBMISSION_PACKAGE.md"
    else:
        candidate = Path(output_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        target = candidate.resolve()

    try:
        rel = target.relative_to(private_dir)
    except ValueError as exc:
        raise ValueError(
            "Output path must be located inside runtime/private_submission/"
        ) from exc

    if str(rel) in ("", ".") or target.is_dir():
        raise ValueError(
            "Output path must specify a file inside runtime/private_submission/"
        )
    return target


def _looks_like_placeholder(value: str) -> bool:
    lower = value.strip().lower()
    if not lower:
        return True
    return any(marker in lower for marker in PLACEHOLDER_MARKERS)


def read_api_key_state(secret_path: Path) -> dict[str, Any]:
    """Read the API key from runtime/secrets/api_key.txt without inventing values."""
    if not secret_path.is_file():
        return {
            "ready": False,
            "key": None,
            "reason": "missing runtime/secrets/api_key.txt",
        }
    try:
        raw = secret_path.read_text(encoding="utf-8").strip()
    except OSError:
        return {
            "ready": False,
            "key": None,
            "reason": "unreadable runtime/secrets/api_key.txt",
        }

    if not raw:
        return {
            "ready": False,
            "key": None,
            "reason": "empty runtime/secrets/api_key.txt",
        }
    if _looks_like_placeholder(raw) or len(raw) < 8 or any(ch.isspace() for ch in raw):
        return {
            "ready": False,
            "key": None,
            "reason": "invalid or placeholder value in runtime/secrets/api_key.txt",
        }
    return {
        "ready": True,
        "key": raw,
        "reason": "loaded from runtime/secrets/api_key.txt",
    }


def _is_sha256_hex(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if len(stripped) != 64:
        return False
    return all(ch in "0123456789abcdefABCDEF" for ch in stripped)


def _parse_aware_iso_timestamp(raw_value: Any) -> datetime | None:
    if not isinstance(raw_value, str):
        return None
    text = raw_value.strip()
    if not text:
        return None
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        return None
    return parsed


def read_public_endpoint_state(
    endpoint_path: Path,
    *,
    reference_time: datetime | None = None,
    max_age_sec: float = DEFAULT_MAX_ENDPOINT_AGE_SEC,
) -> dict[str, Any]:
    """Read and validate runtime/public_endpoint.json without network access.

    Actual endpoint schema:
    {schema_version:1, url, status:'connected'|'degraded'|'reconnecting'|...,
     updated_at, instance, generation, api_ready:bool, tunnel_connected:bool,
     temporary:true}
    """
    default_state: dict[str, Any] = {
        "ready": False,
        "local_metadata_status": "LOCAL_METADATA_NOT_READY",
        "external_reachability_status": "EXTERNAL_REACHABILITY_NOT_TESTED",
        "schema_version": "not recorded",
        "url": None,
        "status": "missing",
        "instance": "not recorded",
        "generation": "not recorded",
        "updated_at": "not recorded",
        "api_ready": False,
        "tunnel_connected": False,
        "temporary": None,
        "age_seconds": None,
        "max_age_seconds": max_age_sec,
        "issues": [],
    }
    if not endpoint_path.is_file():
        default_state["issues"] = ["missing runtime/public_endpoint.json"]
        return default_state

    try:
        payload = json.loads(endpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        default_state["status"] = "invalid_json"
        default_state["issues"] = ["invalid JSON in runtime/public_endpoint.json"]
        return default_state

    if not isinstance(payload, Mapping):
        default_state["status"] = "invalid_payload"
        default_state["issues"] = ["runtime/public_endpoint.json root must be an object"]
        return default_state

    issues: list[str] = []

    schema_ver = payload.get("schema_version")
    if isinstance(schema_ver, bool) or schema_ver != 1:
        issues.append("missing or invalid schema_version in runtime/public_endpoint.json (expected 1)")
        schema_ver_str = str(schema_ver) if schema_ver is not None else "not recorded"
    else:
        schema_ver_str = "1"

    raw_url = payload.get("url")
    url_str = str(raw_url).strip().rstrip("/") if isinstance(raw_url, str) else ""
    if not url_str:
        issues.append("missing public URL in runtime/public_endpoint.json")
    else:
        parsed = urlparse(url_str)
        host = (parsed.hostname or "").lower()
        if parsed.scheme.lower() != "https" or not host:
            issues.append("public URL must be a valid https:// URL")
        elif host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
            issues.append("public URL points to localhost rather than a public tunnel")
        elif _looks_like_placeholder(url_str):
            issues.append("public URL contains a placeholder hostname")

    status_raw = payload.get("status")
    status_str = str(status_raw).strip() if isinstance(status_raw, str) and status_raw.strip() else "missing"
    if status_str != "connected":
        issues.append(f"endpoint status is not ready ({status_str}; expected 'connected')")

    if payload.get("stale") is True:
        issues.append("endpoint is marked stale")

    api_ready_raw = payload.get("api_ready")
    if api_ready_raw is not True:
        issues.append(f"api_ready flag must be strictly true (got {api_ready_raw!r})")

    tunnel_connected_raw = payload.get("tunnel_connected")
    if tunnel_connected_raw is not True:
        issues.append(f"tunnel_connected flag must be strictly true (got {tunnel_connected_raw!r})")

    temporary_raw = payload.get("temporary")
    if temporary_raw is not True:
        issues.append(f"temporary flag must be strictly true (got {temporary_raw!r})")

    instance_raw = payload.get("instance")
    if isinstance(instance_raw, str) and instance_raw.strip() and not _looks_like_placeholder(instance_raw):
        instance_str = instance_raw.strip()
    else:
        instance_str = "not recorded"
        issues.append("missing or invalid instance identifier in runtime/public_endpoint.json")

    generation_raw = payload.get("generation")
    if (
        isinstance(generation_raw, int)
        and not isinstance(generation_raw, bool)
        and generation_raw >= 1
    ):
        generation_str = str(generation_raw)
    else:
        generation_str = "not recorded"
        issues.append("missing or invalid generation integer in runtime/public_endpoint.json")

    updated_raw = payload.get("updated_at")
    updated_str = str(updated_raw).strip() if isinstance(updated_raw, str) and updated_raw.strip() else "not recorded"
    age_seconds: float | None = None
    if updated_str == "not recorded":
        issues.append("missing updated_at timestamp in runtime/public_endpoint.json")
    else:
        parsed_dt = _parse_aware_iso_timestamp(updated_str)
        if parsed_dt is None:
            issues.append("invalid or timezone-naive updated_at timestamp in runtime/public_endpoint.json")
        else:
            ref_dt = reference_time if reference_time is not None else datetime.now(timezone.utc)
            if ref_dt.tzinfo is None or ref_dt.tzinfo.utcoffset(ref_dt) is None:
                raise ValueError("reference_time must be timezone-aware")
            age_seconds = (ref_dt - parsed_dt).total_seconds()
            if age_seconds < 0:
                issues.append("updated_at timestamp is in the future")
            elif age_seconds > max_age_sec:
                issues.append(
                    f"endpoint metadata is stale (age {age_seconds:.1f}s exceeds max_age {max_age_sec:.1f}s)"
                )

    is_ready = len(issues) == 0
    return {
        "ready": is_ready,
        "local_metadata_status": "LOCAL_METADATA_READY" if is_ready else "LOCAL_METADATA_NOT_READY",
        "external_reachability_status": "EXTERNAL_REACHABILITY_NOT_TESTED",
        "schema_version": schema_ver_str,
        "url": url_str if url_str else None,
        "status": status_str,
        "instance": instance_str,
        "generation": generation_str,
        "updated_at": updated_str,
        "api_ready": api_ready_raw is True,
        "tunnel_connected": tunnel_connected_raw is True,
        "temporary": temporary_raw is True,
        "age_seconds": age_seconds,
        "max_age_seconds": max_age_sec,
        "issues": issues,
    }


def resolve_active_model_dir(
    repo_root: Path,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve active VLM directory compatible with FORTE_MODEL_PATH and AnalysisConfig.from_env()."""
    root = repo_root.resolve()
    env_map = os.environ if env is None else env
    raw_override = str(env_map.get("FORTE_MODEL_PATH") or "").strip()
    if raw_override:
        candidate = Path(raw_override)
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve()
    return (root / "models" / "Qwen3-VL-4B-Instruct").resolve()


def resolve_grounding_model_dir(
    repo_root: Path,
    env: Mapping[str, str] | None = None,
) -> tuple[bool, Path]:
    """Resolve optional Grounding DINO directory compatible with AnalysisConfig.from_env()."""
    root = repo_root.resolve()
    env_map = os.environ if env is None else env
    enabled = str(env_map.get("FORTE_GROUNDING_ENABLED", "1")).strip() == "1"
    raw_override = str(env_map.get("FORTE_GROUNDING_MODEL_PATH") or "").strip()
    if raw_override:
        candidate = Path(raw_override)
        if not candidate.is_absolute():
            candidate = root / candidate
        return enabled, candidate.resolve()
    return enabled, (root / "models" / "grounding-dino-tiny").resolve()


def resolve_model_provenance_path(
    repo_root: Path,
    model_provenance_path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[Path, Path]:
    """Return (active_model_dir, download_provenance_json_path) without loading models."""
    root = repo_root.resolve()
    if model_provenance_path is not None:
        candidate = Path(model_provenance_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()
        if resolved.is_dir() or (not resolved.is_file() and resolved.suffix.lower() != ".json"):
            return resolved, (resolved / "download_provenance.json").resolve()
        return resolved.parent, resolved

    active_model_dir = resolve_active_model_dir(root, env=env)
    return active_model_dir, (active_model_dir / "download_provenance.json").resolve()


def _validate_weight_integrity_list(
    raw_weights: Any,
) -> tuple[bool, list[dict[str, Any]], int]:
    if not isinstance(raw_weights, list) or len(raw_weights) == 0:
        return False, [], 0
    cleaned: list[dict[str, Any]] = []
    total_bytes = 0
    for entry in raw_weights:
        if not isinstance(entry, Mapping):
            return False, [], 0
        filename = entry.get("filename")
        size_bytes = entry.get("bytes")
        sha256_hex = entry.get("sha256")
        if not isinstance(filename, str) or not filename.strip() or _looks_like_placeholder(filename):
            return False, [], 0
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            return False, [], 0
        if not _is_sha256_hex(sha256_hex):
            return False, [], 0
        cleaned.append(
            {
                "filename": filename.strip(),
                "bytes": int(size_bytes),
                "sha256": str(sha256_hex).strip().lower(),
            }
        )
        total_bytes += int(size_bytes)
    return True, cleaned, total_bytes


def read_model_provenance(
    provenance_path: Path | None,
    *,
    model_dir: Path | None = None,
) -> dict[str, Any]:
    """Read local model download_provenance.json without claiming weight file re-hashing.

    Real downloader schema:
    {status:'complete', repo_id, revision, license, weight_integrity:[{filename,bytes,sha256}]}
    """
    resolved_dir = (
        model_dir.resolve()
        if model_dir is not None
        else (provenance_path.resolve().parent if provenance_path is not None else None)
    )
    dir_display = str(resolved_dir) if resolved_dir is not None else "not specified"

    if provenance_path is None or not provenance_path.is_file():
        return {
            "complete": False,
            "verified": False,
            "label": "unverified",
            "model_dir": dir_display,
            "provenance_path": str(provenance_path) if provenance_path is not None else "not specified",
            "repo_id": "not recorded",
            "revision": "not recorded",
            "license": "not recorded",
            "status": "missing",
            "weight_integrity": [],
            "weight_files_count": 0,
            "weight_total_bytes": 0,
            "note": (
                "download_provenance.json not found in active model directory; "
                "model provenance is unverified (metadata not recorded)."
            ),
        }

    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "complete": False,
            "verified": False,
            "label": "unverified",
            "model_dir": dir_display,
            "provenance_path": str(provenance_path),
            "repo_id": "not recorded",
            "revision": "not recorded",
            "license": "not recorded",
            "status": "invalid_json",
            "weight_integrity": [],
            "weight_files_count": 0,
            "weight_total_bytes": 0,
            "note": "download_provenance.json could not be parsed as JSON; model provenance is unverified.",
        }

    if not isinstance(payload, Mapping):
        return {
            "complete": False,
            "verified": False,
            "label": "unverified",
            "model_dir": dir_display,
            "provenance_path": str(provenance_path),
            "repo_id": "not recorded",
            "revision": "not recorded",
            "license": "not recorded",
            "status": "invalid_payload",
            "weight_integrity": [],
            "weight_files_count": 0,
            "weight_total_bytes": 0,
            "note": "download_provenance.json root is not an object; model provenance is unverified.",
        }

    repo_id_raw = payload.get("repo_id")
    revision_raw = payload.get("revision")
    license_raw = payload.get("license")
    status_raw = payload.get("status")

    repo_id = str(repo_id_raw).strip() if isinstance(repo_id_raw, str) and repo_id_raw.strip() else "not recorded"
    revision = str(revision_raw).strip() if isinstance(revision_raw, str) and revision_raw.strip() else "not recorded"
    license_str = str(license_raw).strip() if isinstance(license_raw, str) and license_raw.strip() else "not recorded"
    status = str(status_raw).strip() if isinstance(status_raw, str) and status_raw.strip() else "not recorded"

    weights_ok, weights_list, total_bytes = _validate_weight_integrity_list(
        payload.get("weight_integrity")
    )

    is_complete = (
        status == "complete"
        and repo_id != "not recorded"
        and revision != "not recorded"
        and license_str != "not recorded"
        and not _looks_like_placeholder(repo_id)
        and not _looks_like_placeholder(revision)
        and not _looks_like_placeholder(license_str)
        and weights_ok
    )

    if is_complete:
        note = (
            "Downloader provenance metadata is complete (metadata-only: weight files on disk "
            "were NOT cryptographically re-hashed by the package generator, and no new model "
            "inference was executed)."
        )
        label = "complete (metadata-only; weight files not re-hashed)"
    else:
        reasons: list[str] = []
        if status != "complete":
            reasons.append(f"status={status!r} (expected 'complete')")
        if repo_id == "not recorded" or _looks_like_placeholder(repo_id):
            reasons.append("missing/invalid repo_id")
        if revision == "not recorded" or _looks_like_placeholder(revision):
            reasons.append("missing/invalid revision")
        if license_str == "not recorded" or _looks_like_placeholder(license_str):
            reasons.append("missing/invalid license")
        if not weights_ok:
            reasons.append("missing or malformed weight_integrity list")
        note = (
            "Incomplete or malformed download_provenance.json ("
            + ", ".join(reasons)
            + "); labeled unverified."
        )
        label = "unverified"

    return {
        "complete": is_complete,
        "verified": False,
        "label": label,
        "model_dir": dir_display,
        "provenance_path": str(provenance_path),
        "repo_id": repo_id,
        "revision": revision,
        "license": license_str,
        "status": status,
        "weight_integrity": weights_list,
        "weight_files_count": len(weights_list),
        "weight_total_bytes": total_bytes,
        "note": note,
    }


def _canonical(value: Any) -> Any:
    """Canonical serializer matching scripts/evaluate.py provenance digest logic."""
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _digest(value: Any) -> str:
    """Canonical SHA-256 digest matching scripts/evaluate.py."""
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sanitize_numeric_value(field: str, value: Any) -> int | float | None:
    """Strictly validate allowlisted summary metrics.

    Rejects None, bool, NaN/Inf, negative values, non-integral counts, and
    probabilities outside [0.0, 1.0].
    """
    if isinstance(value, bool) or value is None:
        return None

    if field in INTEGER_SUMMARY_FIELDS:
        if isinstance(value, int):
            return int(value) if value >= 0 else None
        return None

    if not isinstance(value, (int, float)):
        return None

    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        return None

    if field in PROBABILITY_SUMMARY_FIELDS:
        if numeric > 1.0:
            return None
        return numeric

    if field in DURATION_SUMMARY_FIELDS:
        return numeric

    return None


def _verify_evaluation_provenance(
    payload: Mapping[str, Any],
    *,
    split: str,
    model: str,
) -> dict[str, Any]:
    """Verify internal consistency of evaluation provenance metadata via _digest.

    Checks:
    1. provenance.schema_version == 1
    2. provenance.identity is a dict and _digest(identity) == provenance.identity_sha256
    3. payload['split'] == provenance['identity']['split']
    4. payload['model'] == Path(provenance['identity']['config']['model_path']).name
    Never returns or exposes the contents of identity, selected_ids, or selected_records.
    """
    prov_obj = payload.get("provenance")
    if not isinstance(prov_obj, Mapping):
        return {
            "identity_sha256": None,
            "provenance_status": "unverified",
            "provenance_reason": "missing provenance object",
            "identity_consistent": False,
        }

    raw_sha = prov_obj.get("identity_sha256")
    sha_str = str(raw_sha).strip().lower() if _is_sha256_hex(raw_sha) else None

    schema_ver = prov_obj.get("schema_version")
    if isinstance(schema_ver, bool) or schema_ver != 1:
        return {
            "identity_sha256": sha_str,
            "provenance_status": "unverified",
            "provenance_reason": "missing or invalid provenance.schema_version (expected 1)",
            "identity_consistent": False,
        }

    if sha_str is None:
        return {
            "identity_sha256": None,
            "provenance_status": "unverified",
            "provenance_reason": "missing or non-hex provenance.identity_sha256",
            "identity_consistent": False,
        }

    identity_obj = prov_obj.get("identity")
    if not isinstance(identity_obj, Mapping):
        return {
            "identity_sha256": sha_str,
            "provenance_status": "unverified",
            "provenance_reason": "missing provenance.identity object for canonical digest check",
            "identity_consistent": False,
        }

    try:
        computed_sha = _digest(dict(identity_obj))
    except (TypeError, ValueError):
        return {
            "identity_sha256": sha_str,
            "provenance_status": "unverified",
            "provenance_reason": "provenance.identity contains non-canonical or NaN/Inf values",
            "identity_consistent": False,
        }

    if computed_sha != sha_str:
        return {
            "identity_sha256": sha_str,
            "provenance_status": "unverified",
            "provenance_reason": "canonical digest mismatch between provenance.identity and identity_sha256",
            "identity_consistent": False,
        }

    identity_split_raw = identity_obj.get("split")
    identity_split = (
        str(identity_split_raw).strip().lower()
        if isinstance(identity_split_raw, str)
        else ""
    )
    if not identity_split or identity_split != split:
        return {
            "identity_sha256": sha_str,
            "provenance_status": "unverified",
            "provenance_reason": "payload split does not match provenance.identity.split",
            "identity_consistent": False,
        }

    config_obj = identity_obj.get("config")
    model_path_raw = config_obj.get("model_path") if isinstance(config_obj, Mapping) else None
    if not isinstance(model_path_raw, str) or not model_path_raw.strip():
        return {
            "identity_sha256": sha_str,
            "provenance_status": "unverified",
            "provenance_reason": "missing provenance.identity.config.model_path",
            "identity_consistent": False,
        }

    expected_model_name = Path(model_path_raw.strip().replace("\\", "/")).name
    if expected_model_name != model:
        return {
            "identity_sha256": sha_str,
            "provenance_status": "unverified",
            "provenance_reason": (
                f"payload model ({model!r}) does not match "
                f"Path(identity.config.model_path).name ({expected_model_name!r})"
            ),
            "identity_consistent": False,
        }

    identity_model_raw = identity_obj.get("model")
    if isinstance(identity_model_raw, str) and identity_model_raw.strip():
        id_model_clean = identity_model_raw.strip()
        if id_model_clean != model and Path(id_model_clean.replace("\\", "/")).name != model:
            return {
                "identity_sha256": sha_str,
                "provenance_status": "unverified",
                "provenance_reason": "payload model does not match provenance.identity.model",
                "identity_consistent": False,
            }

    return {
        "identity_sha256": sha_str,
        "provenance_status": "identity_hash_consistent (metadata-only)",
        "provenance_reason": (
            "canonical _digest(identity) matches identity_sha256 and split/model metadata "
            "(proves internal metadata consistency only; does not re-run evaluation or verify weight files)"
        ),
        "identity_consistent": True,
    }


def load_evaluation_summary(eval_path: Path) -> dict[str, Any]:
    """Parse an evaluation JSON container and extract only allowlisted aggregate metrics.

    Discards ``records``, ``provenance.identity``, ``selected_ids``, ``selected_records``,
    and any non-allowlisted fields so they are never rendered.
    """
    try:
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to read evaluation JSON: {eval_path.name}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError(f"Evaluation JSON root must be an object: {eval_path.name}")

    split_raw = payload.get("split")
    split = (
        str(split_raw).strip().lower()
        if isinstance(split_raw, str) and str(split_raw).strip()
        else "unspecified"
    )
    model_raw = payload.get("model")
    model = (
        str(model_raw).strip()
        if isinstance(model_raw, str) and str(model_raw).strip()
        else "unspecified_model"
    )

    prov_check = _verify_evaluation_provenance(payload, split=split, model=model)
    identity_sha256 = prov_check["identity_sha256"]
    provenance_status = prov_check["provenance_status"]
    fingerprint_suffix = (
        identity_sha256
        if prov_check["identity_consistent"] and identity_sha256
        else f"unverified:{eval_path.name}"
    )

    raw_summary = payload.get("summary")
    if not isinstance(raw_summary, Mapping):
        raw_summary = {}

    sanitized_summary: dict[str, int | float | None] = {}
    for field in ALLOWLISTED_SUMMARY_FIELDS:
        sanitized_summary[field] = _sanitize_numeric_value(field, raw_summary.get(field))

    return {
        "source_name": eval_path.name,
        "split": split,
        "model": model,
        "identity_sha256": identity_sha256,
        "provenance_status": provenance_status,
        "provenance_reason": prov_check["provenance_reason"],
        "identity_consistent": prov_check["identity_consistent"],
        "fingerprint": f"{model}@{fingerprint_suffix}",
        "summary": sanitized_summary,
    }


def format_metric_value(field: str, value: int | float | None) -> str:
    """Format a single validated metric or return 'not measured' if missing/invalid."""
    if value is None:
        return "not measured"
    if field in INTEGER_SUMMARY_FIELDS and isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if field in PROBABILITY_SUMMARY_FIELDS:
            return f"{value:.4f}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def load_normative_registry(repo_root: Path) -> dict[str, Any]:
    """Load config/normative_rules_rk.json at runtime (with fallback to repo default for synthetic tests)."""
    candidates = [
        (repo_root.resolve() / "config" / "normative_rules_rk.json").resolve(),
        (DEFAULT_REPO_ROOT / "config" / "normative_rules_rk.json").resolve(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, Mapping) and isinstance(data.get("rules"), list):
                    return dict(data)
            except (OSError, json.JSONDecodeError):
                continue
    return {"rules": [], "primary_source": {}, "decision_policy": {}}


def _render_normative_registry_section(registry: Mapping[str, Any]) -> list[str]:
    rules = registry.get("rules") if isinstance(registry.get("rules"), list) else []
    supported_rules = [
        r for r in rules if isinstance(r, Mapping) and r.get("supported") is True
    ]
    unsupported_rules = [
        r for r in rules if isinstance(r, Mapping) and r.get("supported") is not True
    ]

    lines: list[str] = [
        "## 6. Нормативный реестр РК (`config/normative_rules_rk.json`)",
        "",
        "- **Основной нормативный акт:** Постановление Правления Национального Банка Республики Казахстан от 29 ноября 2019 года № 231 (регистрация МЮ РК № 19680 от 03.12.2019, `https://adilet.zan.kz/rus/docs/V1900019680`, индексированный текст `https://old.adilet.zan.kz/rus/docs/V1900019680`).",
        "- **Контекстный акт по устройству помещений:** Постановление Правления НБРК от 24 февраля 2020 года № 14 (`https://adilet.zan.kz/rus/docs/V2000020075`).",
        "- **Семантика поля `violations[].type` в HTTP API:** в ответах `GET /api/v1/jobs/{job_id}` поле `violations[].type` содержит **человекочитаемое название правила `rule[\"title\"]`**, а НЕ внутренний идентификатор `rule[\"id\"]`. Внутренний `rule[\"id\"]` используется локальным классификатором и сервером для маппинга и подстановки проверенной ссылки `source`.",
        "- **Политика доказательности:** все условия `visual_conditions` соответствующего правила обязательны одновременно (`all_visual_conditions_required=true`). Для правил с `requires_complete_action_sequence=true` требуется непрерывная видимость всей операции. Неизвестное не трактуется как нарушение (`unknown_is_not_evidence=true`). Один эпизод не дублируется под общим и частным классом.",
        "",
        f"### 6.1. Поддерживаемые правила (`supported=true`, всего: {len(supported_rules)})",
        "",
        "| Поле `violations[].type` (`rule.title`) | Внутренний `rule.id` | Требуемые визуальные условия (`visual_conditions`) | Недостаточные доказательства и исключения (`insufficient_evidence` / `exceptions`) | Нормативный источник (`source`) |",
        "| --- | --- | --- | --- | --- |",
    ]

    for rule in supported_rules:
        r_id = str(rule.get("id") or "")
        r_title = str(rule.get("title") or "")
        vis_list = rule.get("visual_conditions") if isinstance(rule.get("visual_conditions"), list) else []
        insuff_list = rule.get("insufficient_evidence") if isinstance(rule.get("insufficient_evidence"), list) else []
        exc_list = rule.get("exceptions") if isinstance(rule.get("exceptions"), list) else []
        src = rule.get("source") if isinstance(rule.get("source"), Mapping) else {}

        vis_text = "<br>".join(f"{i}. {str(v)}" for i, v in enumerate(vis_list, start=1))
        lim_parts: list[str] = []
        if insuff_list:
            lim_parts.append("Недостаточно: " + "; ".join(str(x) for x in insuff_list))
        if exc_list:
            lim_parts.append("Исключения: " + "; ".join(str(x) for x in exc_list))
        if rule.get("evidence_policy"):
            lim_parts.append("Политика: " + str(rule.get("evidence_policy")))
        lim_text = "<br>".join(lim_parts)

        src_parts = [
            f"{src.get('title', '')} ({src.get('paragraph', '')})",
            f"URL: {src.get('url', '')}",
        ]
        if src.get("quote"):
            src_parts.append(f"Цитата: «{src.get('quote')}»")
        if src.get("amendment_url"):
            src_parts.append(f"Поправка: {src.get('amendment_url')} (проверенный текст: {src.get('verified_text_url', '')})")
        if isinstance(src.get("context_source"), Mapping):
            ctx = src["context_source"]
            src_parts.append(
                f"Контекст: {ctx.get('title', '')} ({ctx.get('paragraph', '')}, {ctx.get('url', '')})"
            )
        if isinstance(src.get("known_caveats"), list) and src["known_caveats"]:
            src_parts.append("Оговорки: " + " ".join(str(c) for c in src["known_caveats"]))
        src_text = "<br>".join(src_parts)

        lines.append(
            f"| **{r_title}** | `{r_id}` | {vis_text} | {lim_text} | {src_text} |"
        )

    lines.extend(
        [
            "",
            f"### 6.2. Неподдерживаемые правила и ограничения видеооценки (`supported=false`, всего: {len(unsupported_rules)})",
            "",
            "Правила со статусом `supported=false` **запрещено** использовать для самостоятельного положительного вердикта:",
            "",
        ]
    )
    for rule in unsupported_rules:
        r_id = str(rule.get("id") or "")
        r_title = str(rule.get("title") or "")
        r_desc = str(rule.get("description") or "")
        insuff = "; ".join(str(x) for x in (rule.get("insufficient_evidence") or []))
        exc = "; ".join(str(x) for x in (rule.get("exceptions") or []))
        lines.append(
            f"- **`{r_id}` (`{r_title}`):** {r_desc} Недостаточность доказательств: {insuff} Исключения: {exc}"
        )
    lines.append("")
    return lines


def _render_evaluation_section(
    evaluations: Sequence[dict[str, Any]],
) -> list[str]:
    lines: list[str] = [
        "## 7. Результаты локальной оценки (Evaluation Summary)",
        "",
        "- **Статус целевой точности (Target Accuracy):** целевая точность НЕ продемонстрирована (`target accuracy has NOT been demonstrated`).",
        "- **Описание набора данных:** в исходной таблице 42 уникальных видеоролика (22 положительных с одним точечным таймкодом и 20 отрицательных без таймкода; разметка типов нарушений и интервалов начала/конца в исходном XLSX отсутствует). Разбиение фиксировано (`seed=26`): 34 обучающих ролика (`train`) и 8 контрольных (`holdout`: 4 положительных, 4 отрицательных). Выборка `holdout` не используется для подбора порогов или few-shot примеров.",
        "- **Политика честности и фильтрации полей:** при разборе JSON-контейнера оценки (`json.loads`) в пакет извлекаются и рендерятся **только 18 разрешённых агрегированных числовых полей** `summary`. Поля `records`, `provenance.identity`, `provenance.selected_ids`, `provenance.selected_records` и любые данные по отдельным видеороликам отбрасываются и никогда не выводятся. Некорректные числовые значения (`NaN`, `Inf`, `bool`, отрицательные числа, дробные счётчики и вероятности вне диапазона `0..1`) отклоняются и помечаются как `not measured`. Завершившиеся с ошибкой или таймаутом видеоролики (`failed`, `failed_positive`) остаются ошибками.",
        "- **Проверка внутренней согласованности метаданных (`identity_sha256`):** статус `identity_hash_consistent (metadata-only)` присваивается только тогда, когда `provenance.schema_version == 1`, канонический SHA-256 дайджест `_digest(provenance.identity)` совпадает с `provenance.identity_sha256`, `split == provenance.identity.split` и `model == Path(provenance.identity.config.model_path).name`. Эта проверка подтверждает только внутреннюю согласованность метаданных JSON-файла оценки и **не означает** повторного запуска инференса или перехеширования файлов весов модели. Произвольная строка `identity_sha256` без валидного объекта `identity` помечается как `unverified`.",
    ]

    if not evaluations:
        lines.extend(
            [
                "- **Предоставленные файлы оценки:** не предоставлены (`not measured`).",
                "- **Train split:** `not measured`",
                "- **Holdout split:** `not measured`",
                "",
            ]
        )
        return lines

    fingerprints = {item["fingerprint"] for item in evaluations}
    unverified_count = sum(
        1 for item in evaluations if not item["identity_consistent"]
    )
    if len(fingerprints) > 1:
        lines.append(
            "- **Предупреждение о совместимости прогонов:** обнаружены различные профили/отпечатки прогонов (`model` / `provenance.identity_sha256`). Прогоны приведены строго раздельно без объединения (no pooling) и без выбора лучших показателей (no cherry-picking)."
        )
    if unverified_count > 0:
        lines.append(
            f"- **Предупреждение о прослеживаемости:** {unverified_count} прогон(ов) не прошли каноническую проверку согласованности `provenance.identity` и помечены как `unverified`."
        )

    splits_order: list[str] = []
    for canonical in ("train", "holdout"):
        if any(item["split"] == canonical for item in evaluations):
            splits_order.append(canonical)
    for item in evaluations:
        if item["split"] not in splits_order:
            splits_order.append(item["split"])

    for split_name in ("train", "holdout"):
        if split_name not in splits_order:
            lines.append(f"- **Выборка `{split_name}`:** `not measured` (данные не переданы).")

    lines.append("")
    for split_name in splits_order:
        split_runs = [item for item in evaluations if item["split"] == split_name]
        lines.append(f"### Выборка (Split): `{split_name}` (прогонов: {len(split_runs)})")
        lines.append("")
        for idx, run in enumerate(split_runs, start=1):
            sha_display = (
                run["identity_sha256"]
                if run["identity_consistent"] and run["identity_sha256"]
                else f"unverified ({run['provenance_reason']})"
            )
            lines.extend(
                [
                    f"#### Прогон {idx} — `{run['source_name']}` (статус provenance: `{run['provenance_status']}`)",
                    "",
                    f"- **Split:** `{run['split']}`",
                    f"- **Model:** `{run['model']}`",
                    f"- **Provenance (`identity_sha256`):** `{sha_display}`",
                    f"- **Проверка метаданных прогона:** {run['provenance_reason']}",
                    "",
                    "| Поле `summary` | Значение |",
                    "| --- | --- |",
                ]
            )
            summary_map = run["summary"]
            for field in ALLOWLISTED_SUMMARY_FIELDS:
                formatted = format_metric_value(field, summary_map.get(field))
                lines.append(f"| `{field}` | `{formatted}` |")
            lines.append("")

    return lines


def build_submission_package(
    repo_root: Path,
    output_path: Path | str | None = None,
    secret_path: Path | None = None,
    endpoint_path: Path | None = None,
    model_provenance_path: Path | str | None = None,
    eval_paths: Sequence[Path] | None = None,
    *,
    reference_time: datetime | None = None,
    max_endpoint_age_sec: float = DEFAULT_MAX_ENDPOINT_AGE_SEC,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Generate runtime/private_submission/SUBMISSION_PACKAGE.md and return metadata."""
    root = repo_root.resolve()
    resolved_output = resolve_private_output_path(root, output_path)
    resolved_secret = (
        secret_path.resolve()
        if secret_path is not None
        else (root / "runtime" / "secrets" / "api_key.txt").resolve()
    )
    resolved_endpoint = (
        endpoint_path.resolve()
        if endpoint_path is not None
        else (root / "runtime" / "public_endpoint.json").resolve()
    )
    active_model_dir, resolved_provenance = resolve_model_provenance_path(
        root,
        model_provenance_path=model_provenance_path,
        env=env,
    )
    grounding_enabled, grounding_model_dir = resolve_grounding_model_dir(root, env=env)
    grounding_prov_path = (grounding_model_dir / "download_provenance.json").resolve()

    api_key_state = read_api_key_state(resolved_secret)
    endpoint_state = read_public_endpoint_state(
        resolved_endpoint,
        reference_time=reference_time,
        max_age_sec=max_endpoint_age_sec,
    )
    provenance_state = read_model_provenance(
        resolved_provenance,
        model_dir=active_model_dir,
    )
    grounding_provenance_state = read_model_provenance(
        grounding_prov_path if grounding_prov_path.is_file() else None,
        model_dir=grounding_model_dir,
    )
    normative_registry = load_normative_registry(root)

    loaded_evals: list[dict[str, Any]] = []
    for path in eval_paths or ():
        loaded_evals.append(load_evaluation_summary(Path(path).resolve()))

    overall_ready = bool(api_key_state["ready"] and endpoint_state["ready"])
    overall_status = "ready" if overall_ready else "not_ready"
    local_metadata_status = (
        "LOCAL_METADATA_READY" if overall_ready else "LOCAL_METADATA_NOT_READY"
    )
    external_reachability_status = "EXTERNAL_REACHABILITY_NOT_TESTED"

    blocking_reasons: list[str] = []
    if not api_key_state["ready"]:
        blocking_reasons.append(f"API key not ready: {api_key_state['reason']}")
    if not endpoint_state["ready"]:
        for issue in endpoint_state["issues"]:
            blocking_reasons.append(f"Endpoint metadata not ready: {issue}")

    url_display = (
        endpoint_state["url"]
        if endpoint_state["url"]
        else "not available (missing or not configured)"
    )
    curl_base_url = (
        endpoint_state["url"]
        if endpoint_state["ready"] and endpoint_state["url"]
        else "<PUBLIC_HTTPS_ENDPOINT_NOT_READY>"
    )
    key_display = (
        api_key_state["key"]
        if api_key_state["ready"] and api_key_state["key"]
        else "NOT_AVAILABLE (missing or invalid runtime/secrets/api_key.txt)"
    )
    age_display = (
        f"{endpoint_state['age_seconds']:.1f}s (max allowed: {endpoint_state['max_age_seconds']:.1f}s)"
        if isinstance(endpoint_state.get("age_seconds"), (int, float))
        else f"not available (max allowed: {endpoint_state['max_age_seconds']:.1f}s)"
    )

    weight_lines: list[str] = []
    if provenance_state["weight_integrity"]:
        weight_lines.extend(
            [
                "",
                "| Файл весов (`filename`) | Размер (`bytes`) | Записанный SHA-256 в `download_provenance.json` |",
                "| --- | --- | --- |",
            ]
        )
        for item in provenance_state["weight_integrity"]:
            weight_lines.append(
                f"| `{item['filename']}` | `{item['bytes']}` | `{item['sha256']}` |"
            )
    else:
        weight_lines.append(
            "- **Список `weight_integrity`:** не записан или не прошёл валидацию (`unverified`)."
        )

    md_lines: list[str] = [
        "# ПРИВАТНЫЙ ПАКЕТ ДЛЯ ЖЮРИ (SUBMISSION PACKAGE)",
        "",
        "> **ВНИМАНИЕ:** Этот файл сгенерирован локально в `runtime/private_submission/` и содержит приватные эксплуатационные реквизиты для проверки жюри. ЗАПРЕЩЕНО коммитить этот файл в систему контроля версий или публиковать в открытых каталогах (`docs/`, stdout, логах).",
        "",
        "## 1. Статус готовности пакета (Readiness Status: Local Metadata vs External Reachability)",
        "",
        f"- **Итоговый статус пакета:** `{overall_status.upper()}`",
        f"- **Готовность локальных метаданных (Local Metadata Readiness):** `{local_metadata_status}`",
        f"- **Внешняя сетевая достижимость (External Reachability):** `{external_reachability_status}` — генератор пакета работает полностью офлайн, не выполняет сетевых запросов и НЕ проверяет внешнюю доступность туннеля из интернета.",
        f"- **Локальные метаданные туннеля (`runtime/public_endpoint.json`):** `{'READY' if endpoint_state['ready'] else 'NOT_READY'}` (`status={endpoint_state['status']}`, `api_ready={endpoint_state['api_ready']}`, `tunnel_connected={endpoint_state['tunnel_connected']}`, `temporary={endpoint_state['temporary']}`)",
        f"- **Ключ доступа (`runtime/secrets/api_key.txt`):** `{'READY' if api_key_state['ready'] else 'NOT_READY'}` ({api_key_state['reason']})",
        f"- **Метаданные активной модели (`download_provenance.json`):** `{provenance_state['label']}`",
        "- **Демонстрация целевой точности (Target Accuracy):** `NOT DEMONSTRATED` (целевая точность НЕ продемонстрирована)",
    ]

    if blocking_reasons:
        md_lines.extend(
            [
                "",
                "### Причины статуса `NOT_READY` (`LOCAL_METADATA_NOT_READY`):",
                *[f"- {reason}" for reason in blocking_reasons],
            ]
        )

    md_lines.extend(
        [
            "",
            "## 2. Архитектура решения и поток обработки видео (Architecture & Data Flow)",
            "",
            "1. **Приём и очередь заданий (`FastAPI` + `SQLite` + постоянный воркер модели):** сервис принимает видеофайлы `.mov` (до 90 MiB через публичный туннель) по `POST /api/v1/analyze`, сохраняет метаданные задания в SQLite и передаёт обработку отдельному постоянному процессу модели. Модель загружается в память один раз при старте сервиса и обрабатывает задания последовательно. Общий лимит времени (ожидание в очереди + анализ) составляет **не более 300 секунд** (`<= 300s`). Любая ошибка модели или превышение 300 секунд переводит задание в `status: \"failed\"`, а не в ложный `no_violation`. Результаты завершённых заданий сохраняются в SQLite при перезапуске.",
            "2. **Декодирование видеопотока (`PyAV 16.1.0`):** декодируется только видеопоток без использования аудио и без биометрии/распознавания лиц. Выборка с частотой `scan_fps = 2.0` FPS сохраняет реальные временные метки (`PTS`), учитывает метаданные поворота контейнера и оценивает межкадровое движение.",
            "3. **Предложение областей сцены (`IDEA-Research/grounding-dino-tiny` на CPU):** опциональный детектор (`grounding_enabled=True`) на CPU определяет ориентировочные области сцены на первом, среднем и последнем кадрах ролика. Выполнение на CPU освобождает всю видеопамять GPU для VLM. Детекции используются только для дополнительного кадрирования и никогда не являются самостоятельным основанием нарушения.",
            "4. **Обзорный проход VLM (`Qwen3-VL`):** локальная визуально-языковая модель просматривает весь ролик по 32 равномерным кадрам (`overview_frames=32`, `overview_pixels=393472`, соответствует детализации `464x848`) и пикам движения и предлагает до `max_candidates=5` кандидатных временных окон.",
            "5. **Детальная проверка кандидатных окон:** каждое окно проверяется отдельно по более плотной последовательности (`detail_frames=16`, `detail_pixels=393472`, `timeout_sec=245.0`, `confidence_threshold=0.62`), сочетающей полные кадры для сохранения контекста и крупные планы из областей Grounding DINO.",
            "6. **Серверная верификация и привязка к нормативному справочнику:** сервер проверяет структуру ответа, допустимость правила (`supported=true` в `config/normative_rules_rk.json`) и таймкода, преобразует внутренний идентификатор `rule[\"id\"]` в человекочитаемое название `rule[\"title\"]` для поля `violations[].type`, автоматически добавляет проверенную нормативную ссылку из `config/normative_rules_rk.json` (модель не сочиняет названия актов и пункты) и извлекает 1–3 реальных JPEG-кадра (`evidence_frames`).",
            "",
            "## 3. Среда выполнения, воспроизводимая подготовка (Provisioning) и офлайн-инференс (OFFLINE Launch)",
            "",
            "### 3.1. Проверенная аппаратная и программная среда (`docs/environment.md`)",
            "- **Основной хост:** Windows 10 Pro x64, AMD Ryzen 7 5700X (8 ядер / 16 потоков), ~48 GiB RAM, NVIDIA GeForce RTX 3080 (10 GiB VRAM, драйвер `596.49`).",
            "- **Изолированный интерпретатор:** `.venv\\Scripts\\python.exe` на базе Python `3.10.11`; `PyTorch 2.10.0+cu128`, `torchvision 0.25.0+cu128` (CUDA 12.8 встроена в колёса), `PyAV 16.1.0`, `Pillow 12.3.0`. Точные версии зафиксированы в `requirements-runtime.lock.txt`. Нативная поддержка `bfloat16` проверена.",
            "- **Вторичный компьютер (только read-only осмотр):** Windows 11 Pro, Intel Core i7-13650HX, ~31.7 GiB RAM, RTX 4060 Laptop GPU (8188 MiB VRAM, драйвер `551.52`). Среда и инференс на нём не устанавливались и не валидировались.",
            "- **Опциональный FFmpeg:** переносимый архив Gyan FFmpeg `9.0.2` (`GPLv3`, SHA-256 `4705843ccaaf54257c16ad90f3e952ece33c17df964ecf7bfdbb0f49c7171077`) не обязателен для работы конвейера, так как `PyAV 16.1.0` напрямую читает все 42 контейнера `.mov`.",
            "",
            "### 3.2. Разделение онлайн-подготовки (Provisioning) и полностью офлайнового инференса (OFFLINE Runtime)",
            "- **Этап подготовки (Provisioning — единственная стадия с сетевым скачиванием публичных весов и пакетов):**",
            "  ```powershell",
            "  .\\scripts\\bootstrap.ps1",
            "  . .\\scripts\\runtime_env.ps1",
            "  .venv\\Scripts\\python.exe scripts\\download_models.py --repo Qwen/Qwen3-VL-4B-Instruct IDEA-Research/grounding-dino-tiny",
            "  .venv\\Scripts\\python.exe scripts\\install_cloudflared.py",
            "  ```",
            "  Скрипт `scripts/runtime_env.ps1` удерживает кэши UV, pip, Hugging Face, Torch и CUDA внутри `.cache/` проекта. Загрузчик моделей сверяет лицензию `apache-2.0`, фиксирует commit SHA из `data/model_sources.json`, скачивает веса диапазонами по 32 MiB с проверкой официального LFS SHA-256, не разрешает удалённый код (`trust_remote_code` не используется) и записывает `download_provenance.json` в каталог модели.",
            "- **Этап исполнения и анализа видео (OFFLINE Inference Launch):**",
            "  Перед импортом `transformers` / Hugging Face функция `configure_offline_environment()` в `forte/analysis/config.py` принудительно устанавливает:",
            "  - `HF_HOME = <repo_root>/.cache/huggingface`",
            "  - `TORCH_HOME = <repo_root>/.cache/torch`",
            "  - `HF_HUB_OFFLINE = 1`",
            "  - `TRANSFORMERS_OFFLINE = 1`",
            "  - `HF_HUB_DISABLE_TELEMETRY = 1`",
            "  - `DO_NOT_TRACK = 1`",
            "  - `TOKENIZERS_PARALLELISM = false`",
            "  Модель и процессор загружаются исключительно с локального диска с `local_files_only=True`. Во время анализа видео обращения к внешним AI API или серверам Hugging Face полностью исключены; видеоролики, кадры и тексты анализа не покидают локальную машину.",
            "",
            "## 4. Модели, происхождение весов (`download_provenance.json`) и лицензии (`docs/THIRD_PARTY.md`)",
            "",
            "### 4.1. Активная визуально-языковая модель (Active VLM)",
            f"- **Активный каталог модели (`FORTE_MODEL_PATH` / `AnalysisConfig.from_env().model_path`):** `{provenance_state['model_dir']}`",
            f"- **Файл метаданных происхождения:** `{provenance_state['provenance_path']}`",
            f"- **Статус метаданных `download_provenance.json`:** `{provenance_state['label']}`",
            f"- **Репозиторий (`repo_id`):** `{provenance_state['repo_id']}`",
            f"- **Ревизия (`revision`):** `{provenance_state['revision']}`",
            f"- **Лицензия в метаданных (`license`):** `{provenance_state['license']}` (официальная карточка модели декларирует `Apache-2.0`)",
            f"- **Статус загрузчика (`status`):** `{provenance_state['status']}`",
            f"- **Файлов весов в `weight_integrity`:** `{provenance_state['weight_files_count']}` (суммарно байт: `{provenance_state['weight_total_bytes']}`)",
            f"- **Примечание о проверке:** {provenance_state['note']}",
            *weight_lines,
            "",
            "### 4.2. Опциональные и вспомогательные модели",
            "- **Основная VLM по умолчанию:** `Qwen/Qwen3-VL-4B-Instruct` (`https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct`, квантование `NF4` 4-bit с double quantization, вычисления `bfloat16`, `PyTorch SDPA`, лицензия `Apache-2.0` по официальной карточке модели).",
            "- **Опциональная кандидатная VLM (8B):** `Qwen/Qwen3-VL-8B-Instruct` (`https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct`, лицензия `Apache-2.0` по официальной карточке модели; подключается через переменную окружения `FORTE_MODEL_PATH` при наличии подготовленного локального каталога `models/Qwen3-VL-8B-Instruct`; неизмеренные результаты для 8B не предполагаются и не заявляются).",
            f"- **Опциональный детектор областей сцены (CPU):** `IDEA-Research/grounding-dino-tiny` (`https://huggingface.co/IDEA-Research/grounding-dino-tiny`, лицензия `Apache-2.0` по официальной карточке модели; `grounding_enabled={grounding_enabled}`, локальный каталог `{grounding_provenance_state['model_dir']}`, статус локального `download_provenance.json`: `{grounding_provenance_state['label']}`, ревизия: `{grounding_provenance_state['revision']}`, лицензия: `{grounding_provenance_state['license']}`).",
            "",
            "### 4.3. Сводка лицензий сторонних компонентов (`docs/THIRD_PARTY.md`)",
            "- **Собственный код репозитория:** `MIT`.",
            "- **Модели (`Qwen3-VL-4B-Instruct`, `Qwen3-VL-8B-Instruct`, `Grounding DINO tiny`):** `Apache-2.0` (согласно официальным карточкам Hugging Face; ревизии и SHA-256 весов фиксируются в `download_provenance.json`). Лицензия на модель не даёт права на публикацию клиентских видеозаписей или персональных данных.",
            "- **Библиотеки времени выполнения:** `Transformers` / `Accelerate` (`Apache-2.0`), `bitsandbytes` (`MIT`), `PyTorch` / `torchvision` (`BSD-style`, компоненты CUDA — по условиям NVIDIA), `FastAPI` (`MIT`), `Uvicorn` (`BSD-3-Clause`), `Pydantic` (`MIT`), `PyAV 16.1.0` (`BSD-3-Clause`; компонуемые библиотеки FFmpeg имеют собственные условия лицензирования), `OpenCV` (`Apache-2.0`), `Pillow 12.3.0` (`MIT-CMU`), `NumPy` / `SciPy` (`BSD-3-Clause`), `cloudflared 2026.10.1` (`Apache-2.0`).",
            "- **Опциональная сборка Gyan FFmpeg 9.0.2:** `GPLv3` (`https://www.gyan.dev/ffmpeg/builds/`, требует сохранения текста лицензии GPLv3, сведений о сборке и ссылок на исходный код при распространении бинарных файлов; не требуется для работы основного пайплайна на `PyAV`).",
            "",
            "## 5. Подключение, контракт HTTP API и эксплуатация (`Runtime Endpoint & API Contract`)",
            "",
            "### 5.1. Параметры подключения и локальные метаданные туннеля (`runtime/public_endpoint.json`)",
            f"- **Статус локальных метаданных:** `{local_metadata_status}`",
            f"- **Статус проверки внешней достижимости:** `{external_reachability_status}` (сетевые запросы генератором не выполняются)",
            f"- **Версия схемы (`schema_version`):** `{endpoint_state['schema_version']}`",
            f"- **Базовый публичный HTTPS URL (`url`):** `{url_display}`",
            f"- **Статус туннеля (`status`):** `{endpoint_state['status']}`",
            f"- **Готовность локального API (`api_ready`):** `{endpoint_state['api_ready']}`",
            f"- **Подключение туннеля (`tunnel_connected`):** `{endpoint_state['tunnel_connected']}`",
            f"- **Временный туннель (`temporary`):** `{endpoint_state['temporary']}`",
            f"- **Экземпляр супервизора (`instance`):** `{endpoint_state['instance']}`",
            f"- **Поколение туннеля (`generation`):** `{endpoint_state['generation']}`",
            f"- **Время обновления метаданных (`updated_at`):** `{endpoint_state['updated_at']}` (возраст записи: `{age_display}`)",
            f"- **Заголовок аутентификации:** `X-API-Key: {key_display}`",
            "",
            "### 5.2. Контракт HTTP API",
            "- **`POST /api/v1/analyze`** — отправка видеофайла `multipart/form-data` в поле `file` (`.mov`, размер **до 90 MiB** через публичный туннель) с обязательным заголовком `X-API-Key`. Возвращает HTTP `202 Accepted` и JSON `{\"job_id\": \"<id>\"}`.",
            "- **`GET /api/v1/jobs/{job_id}`** — опрос состояния задачи с обязательным заголовком `X-API-Key`. Поле `status` принимает значения `processing`, `done` или `failed`.",
            "  - При `status: \"done\"` ответ содержит **ровно четыре верхнеуровневых поля**: `job_id`, `status`, `verdict` (`\"violation\"` или `\"no_violation\"`), `violations` (при `no_violation` — пустой список `[]`).",
            "  - Каждый элемент массива `violations` содержит **ровно четыре поля**:",
            "    - `type` — **человекочитаемое название правила `rule[\"title\"]`** из `config/normative_rules_rk.json` (например, `\"Приём наличных без наблюдаемого пересчёта\"`), а **НЕ** внутренний идентификатор `rule[\"id\"]`; только для правил с `supported=true`;",
            "    - `timecode_sec` — таймкод наблюдаемого признака нарушения в секундах (`float` / `int`);",
            "    - `description` — русскоязычное описание наблюдаемого признака с автоматически добавленной ссылкой на проверенный нормативный источник;",
            "    - `evidence_frames` — массив из 1–3 объектов `[{\"timestamp_sec\": <float>, \"image_base64\": \"<Base64 JPEG>\"}]`, где `image_base64` содержит чистый Base64 JPEG без префикса `data:`.",
            "- **Дополнительные сервисные маршруты:** `GET /api/v1/jobs/{job_id}/video` (защищённое браузерное MP4-превью с заголовком `X-API-Key`) и `GET /healthz` (проверка доступности процесса и поля `ready`, подтверждающего загрузку модели).",
            "",
            "#### Пример вызова (`curl`):",
            "```bash",
            f'curl -X POST "{curl_base_url}/api/v1/analyze" \\',
            f'  -H "X-API-Key: {key_display}" \\',
            '  -F "file=@sample.mov;type=video/quicktime"',
            "",
            f'curl -X GET "{curl_base_url}/api/v1/jobs/<job_id>" \\',
            f'  -H "X-API-Key: {key_display}"',
            "```",
            "",
            "#### Пример структуры успешного ответа `GET /api/v1/jobs/{job_id}` (`status: \"done\"`):",
            "```json",
            "{",
            '  "job_id": "0199f7e0-0000-7000-8000-000000000001",',
            '  "status": "done",',
            '  "verdict": "violation",',
            '  "violations": [',
            "    {",
            '      "type": "Приём наличных без наблюдаемого пересчёта",',
            '      "timecode_sec": 14.5,',
            '      "description": "Обнаружены признаки возможного нарушения: приём наличных завершён без наблюдаемого полистного пересчёта (Правила № 231 НБРК, п. 19, подп. 2; п. 20, абз. 1).",',
            '      "evidence_frames": [',
            "        {",
            '          "timestamp_sec": 14.5,',
            '          "image_base64": "<base64_jpeg_without_data_prefix>"',
            "        }",
            "      ]",
            "    }",
            "  ]",
            "}",
            "```",
            "",
            "### 5.3. Эксплуатационный регламент и временный туннель до конца 18 октября 2026 (UTC+05)",
            "- **Временный адрес Cloudflare Quick Tunnel (`temporary=true`):** публичный HTTPS-адрес предоставляется бесплатным сервисом Cloudflare Quick Tunnels без SLA и **меняется при каждом перезапуске процесса туннеля**. Локальный супервизор (`scripts/start_service.ps1` / `scripts/serve.py`) отслеживает дочерние процессы, перезапускает их при сбое и обновляет `runtime/public_endpoint.json` (увеличивая `generation` и обновляя `updated_at`). При смене адреса необходимо повторно запустить `scripts/build_submission_package.py` и передать обновлённый пакет жюри.",
            "- **Непрерывная работа хоста:** компьютер под управлением Windows, сетевое соединение и процесс супервизора должны оставаться включёнными без перехода в спящий режим **до конца 18 октября 2026 года (UTC+05)** (`through end of 18 Oct 2026 UTC+05`).",
            "",
        ]
    )

    md_lines.extend(_render_normative_registry_section(normative_registry))
    md_lines.extend(_render_evaluation_section(loaded_evals))

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "status": overall_status,
        "ready": overall_ready,
        "local_metadata_status": local_metadata_status,
        "external_reachability_status": external_reachability_status,
        "output_path": resolved_output,
        "endpoint_ready": endpoint_state["ready"],
        "api_key_ready": api_key_state["ready"],
        "active_model_dir": active_model_dir,
        "model_provenance_status": provenance_state["label"],
        "model_provenance_complete": provenance_state["complete"],
        "evaluations_loaded": len(loaded_evals),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the private submission package inside runtime/private_submission/."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root directory (default: parent of scripts/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (must be inside runtime/private_submission/).",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
        help="Path to runtime/secrets/api_key.txt.",
    )
    parser.add_argument(
        "--public-endpoint-file",
        type=Path,
        default=None,
        help="Path to runtime/public_endpoint.json.",
    )
    parser.add_argument(
        "--model-provenance",
        type=Path,
        default=None,
        help=(
            "Optional path to local model download_provenance.json or model directory "
            "(defaults to FORTE_MODEL_PATH or models/Qwen3-VL-4B-Instruct)."
        ),
    )
    parser.add_argument(
        "--eval-json",
        type=Path,
        action="append",
        default=[],
        help="Explicit evaluation summary JSON path (may be specified multiple times).",
    )
    parser.add_argument(
        "--max-endpoint-age-sec",
        type=float,
        default=DEFAULT_MAX_ENDPOINT_AGE_SEC,
        help=f"Maximum allowed age in seconds for runtime/public_endpoint.json updated_at (default: {DEFAULT_MAX_ENDPOINT_AGE_SEC:.0f}).",
    )
    parser.add_argument(
        "--reference-time",
        type=str,
        default=None,
        help="Optional ISO-8601 timezone-aware reference timestamp for endpoint freshness checks.",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return exit code 2 if the generated package status is not_ready.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ref_time: datetime | None = None
    if args.reference_time is not None:
        ref_time = _parse_aware_iso_timestamp(args.reference_time)
        if ref_time is None:
            print(
                "status=error reason=reference-time must be a valid timezone-aware ISO-8601 timestamp",
                file=sys.stdout,
            )
            return 1

    try:
        result = build_submission_package(
            repo_root=args.repo_root,
            output_path=args.output,
            secret_path=args.api_key_file,
            endpoint_path=args.public_endpoint_file,
            model_provenance_path=args.model_provenance,
            eval_paths=args.eval_json,
            reference_time=ref_time,
            max_endpoint_age_sec=args.max_endpoint_age_sec,
        )
    except ValueError as exc:
        print(f"status=error reason={exc}", file=sys.stdout)
        return 1

    print(
        f"status={result['status']} "
        f"local_metadata={result['local_metadata_status']} "
        f"external_reachability={result['external_reachability_status']} "
        f"output={result['output_path']}",
        file=sys.stdout,
    )
    if args.require_ready and not result["ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
