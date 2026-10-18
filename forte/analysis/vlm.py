"""Local Hugging Face VLM runner. All image inputs stay in this process."""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from .config import AnalysisConfig, configure_offline_environment
from .video import VideoFrame, resize_pixels


VISUAL_TASK_HEADER = (
    "Inspect the following chronological cashier workstation frames. "
    "Focus on visible people, hand interactions, banknotes/cash, counting machines, "
    "collection bags/seals, handheld phones/cameras or portable devices in the cash area, "
    "items on the desk surface, and whether the cashier stays at or leaves the desk."
)


def decode_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    index = text.find("{")
    if index >= 0:
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError as exc:
            # Never salvage an inner object from a truncated outer response.
            err = ValueError("Local model returned an incomplete JSON object")
            err.error_reason = "incomplete_json"  # type: ignore[attr-defined]
            raise err from exc
        if isinstance(value, dict):
            return value
    err = ValueError("Local model did not return a complete JSON object")
    err.error_reason = "no_json_object"  # type: ignore[attr-defined]
    raise err


class LocalVLM:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.lock = threading.RLock()
        self.load_seconds = 0.0

    def load(self) -> None:
        with self.lock:
            if self.model is not None:
                return
            configure_offline_environment()
            if not (self.config.model_path / "config.json").exists():
                raise RuntimeError("Local model is not provisioned. Run scripts/download_models.py first.")
            import torch
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is required for the configured analysis deadline")
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = True
            if not torch.backends.cuda.is_flash_attention_available():
                # Windows CUDA wheels do not compile Flash Attention. HF's GQA
                # optimization otherwise selects quadratic math SDPA, spilling
                # a multi-image prefill into shared memory. Repeating K/V heads
                # is mathematically equivalent and enables efficient CUDA SDPA.
                from transformers.integrations import sdpa_attention
                sdpa_attention.use_gqa_in_sdpa = lambda attention_mask, key, value: False
            start = time.monotonic()
            load_options: dict[str, Any] = {
                "local_files_only": True,
                "trust_remote_code": False,
                "torch_dtype": torch.bfloat16,
                "device_map": {"": 0},
                "attn_implementation": "sdpa",
                "low_cpu_mem_usage": True,
            }
            if self.config.quantization == "nf4":
                load_options["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    # Qwen ties the output head to token embeddings. Explicitly
                    # preserving it avoids an uninitialized 4-bit tied weight.
                    llm_int8_skip_modules=["visual", "lm_head"],
                )
            elif self.config.quantization != "none":
                raise ValueError("Unsupported quantization mode")
            self.processor = AutoProcessor.from_pretrained(str(self.config.model_path), local_files_only=True, trust_remote_code=False)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(str(self.config.model_path), **load_options).eval()
            self.load_seconds = round(time.monotonic() - start, 3)

    def generate(self, frames: list[VideoFrame], prompt: str, *, max_pixels: int, max_tokens: int, deadline: float) -> tuple[dict, dict, str]:
        raise RuntimeError('Neural JSON generation is disabled. Use full_timeline.VideoAnalyzer; Python constructs the response.')
