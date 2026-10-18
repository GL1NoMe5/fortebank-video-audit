"""Project-local paths and bounded inference settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def configure_offline_environment() -> None:
    # Set before importing Transformers/Hugging Face. Downloads are a separate,
    # explicit provisioning step and are never attempted while analyzing footage.
    for key, value in {
        "HF_HOME": str(ROOT / ".cache" / "huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TORCH_HOME": str(ROOT / ".cache" / "torch"),
    }.items():
        os.environ[key] = value


@dataclass(frozen=True)
class AnalysisConfig:
    model_path: Path
    rules_path: Path
    private_dir: Path
    quantization: str = "nf4"
    scan_fps: float = 2.0
    overview_frames: int = 32
    overview_pixels: int = 393472  # Retain native detail of a 464 x 848 frame.
    detail_pixels: int = 393472
    detail_frames: int = 16
    max_candidates: int = 5
    confidence_threshold: float = 0.62
    timeout_sec: float = 285.0
    generation_tokens: int = 1200
    grounding_enabled: bool = False
    grounding_model_path: Path = ROOT / "models" / "grounding-dino-tiny"

    @classmethod
    def from_env(cls) -> "AnalysisConfig":
        return cls(
            model_path=Path(os.environ.get("FORTE_MODEL_PATH", str(ROOT / "models" / "Qwen3-VL-8B-Instruct"))),
            rules_path=Path(os.environ.get("FORTE_RULES_PATH", str(ROOT / "config" / "normative_rules_rk.json"))),
            private_dir=Path(os.environ.get("FORTE_PRIVATE_DIR", str(ROOT / "runtime" / "private"))),
            quantization=os.environ.get("FORTE_QUANTIZATION", "nf4"),
            scan_fps=float(os.environ.get("FORTE_SCAN_FPS", "2")),
            overview_frames=int(os.environ.get("FORTE_OVERVIEW_FRAMES", "32")),
            overview_pixels=int(os.environ.get("FORTE_OVERVIEW_PIXELS", "393472")),
            detail_pixels=int(os.environ.get("FORTE_DETAIL_PIXELS", "393472")),
            detail_frames=int(os.environ.get("FORTE_DETAIL_FRAMES", "16")),
            max_candidates=int(os.environ.get("FORTE_MAX_CANDIDATES", "5")),
            confidence_threshold=float(os.environ.get("FORTE_CONFIDENCE_THRESHOLD", "0.62")),
            timeout_sec=float(os.environ.get("FORTE_ANALYSIS_TIMEOUT", "285")),
            generation_tokens=int(os.environ.get("FORTE_GENERATION_TOKENS", "1200")),
            grounding_enabled=os.environ.get("FORTE_GROUNDING_ENABLED", "0") == "1",
            grounding_model_path=Path(os.environ.get("FORTE_GROUNDING_MODEL_PATH", str(ROOT / "models" / "grounding-dino-tiny"))),
        )


