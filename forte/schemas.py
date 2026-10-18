"""Pydantic schemas and strict output validation for ForteBank video audit API."""

from __future__ import annotations

import math
import base64
import binascii
from io import BytesIO
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from PIL import Image

JobStatus = Literal["processing", "done", "failed"]
VerdictType = Literal["violation", "no_violation"]


def _validate_finite_non_negative_seconds(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a numeric value in seconds")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return numeric


class EvidenceFrame(BaseModel):
    """Single evidence frame attached to a detected violation."""

    model_config = ConfigDict(extra="forbid")

    timestamp_sec: float
    image_base64: str = Field(min_length=1)

    @field_validator("timestamp_sec", mode="before")
    @classmethod
    def _check_timestamp(cls, value: Any) -> float:
        return _validate_finite_non_negative_seconds(value, "timestamp_sec")

    @field_validator("image_base64", mode="before")
    @classmethod
    def _check_base64(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("image_base64 must be a non-empty string")
        value = value.strip()
        try:
            decoded = base64.b64decode(value, validate=True)
            if not decoded.startswith(b'\xff\xd8\xff'):
                raise ValueError('Evidence must be JPEG.')
            with Image.open(BytesIO(decoded)) as image:
                if image.format != 'JPEG':
                    raise ValueError('Evidence must be JPEG.')
                image.verify()
        except (binascii.Error, ValueError, OSError) as exc:
            raise ValueError('image_base64 must contain a valid JPEG encoded as pure base64.') from exc
        return value


class ViolationItem(BaseModel):
    """Single policy/safety violation item."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(min_length=1)
    timecode_sec: float
    description: str = Field(min_length=1)
    evidence_frames: list[EvidenceFrame] = Field(min_length=1, max_length=3)

    @field_validator("type", "description", mode="before")
    @classmethod
    def _check_non_empty_text(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Field must be a non-empty string")
        return value.strip()

    @field_validator("timecode_sec", mode="before")
    @classmethod
    def _check_timecode(cls, value: Any) -> float:
        return _validate_finite_non_negative_seconds(value, "timecode_sec")


class AnalyzerOutput(BaseModel):
    """Strict validation model for raw output returned by VideoAnalyzer.analyze."""

    model_config = ConfigDict(extra="forbid")

    verdict: VerdictType
    violations: list[ViolationItem]

    @model_validator(mode="after")
    def _check_verdict_consistency(self) -> AnalyzerOutput:
        if self.verdict == "no_violation" and len(self.violations) != 0:
            raise ValueError("verdict 'no_violation' requires an empty violations list")
        if self.verdict == "violation" and len(self.violations) == 0:
            raise ValueError("verdict 'violation' requires at least one violation entry")
        return self


class AnalyzeAcceptedResponse(BaseModel):
    """Response payload for POST /api/v1/analyze (HTTP 202)."""

    model_config = ConfigDict(extra="forbid")

    job_id: str


class JobProcessingResponse(BaseModel):
    """Response payload for GET /api/v1/jobs/{job_id} when processing."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Literal["processing"] = "processing"
    stage: str | None = None
    progress: float | None = Field(default=None, ge=0.0, le=1.0)


class JobDoneResponse(BaseModel):
    """Response payload for GET /api/v1/jobs/{job_id} when done.

    Contains strictly job_id, status, verdict, and violations.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Literal["done"] = "done"
    verdict: VerdictType
    violations: list[ViolationItem]


class JobFailedResponse(BaseModel):
    """Response payload for GET /api/v1/jobs/{job_id} when failed."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: Literal["failed"] = "failed"
    error_code: str
    error: str


class ErrorResponse(BaseModel):
    """Standardized error body for HTTP 4xx/5xx responses."""

    model_config = ConfigDict(extra="forbid")

    detail: str
    error_code: str


class HealthResponse(BaseModel):
    """Response payload for GET /healthz."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    ready: bool
    worker_alive: bool
    api_key_configured: bool
    queue_depth: int
    queue_capacity: int


class PublicConfigResponse(BaseModel):
    """Response payload for GET /api/v1/config (never exposes secrets)."""

    model_config = ConfigDict(extra="forbid")

    max_upload_bytes: int
    max_upload_mib: int
    allowed_extensions: list[str]
    max_queue_size: int
    job_deadline_seconds: float
    auth_header: str = "X-API-Key"
    requires_api_key: bool = True
    video_preview_supported: bool
