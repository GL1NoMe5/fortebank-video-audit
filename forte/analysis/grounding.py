"""Optional local object grounding; boxes guide views, never determine verdicts.

The detector runs on CPU to reserve VRAM for the VLM. Calibration is rebuilt for
each upload, with normalized boxes rather than fixed camera coordinates.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass

from .config import AnalysisConfig, configure_offline_environment
from .video import SampledVideo, VideoFrame


@dataclass
class SceneRegion:
    timestamp_sec: float
    label: str
    confidence: float
    box: tuple[float, float, float, float]


class SceneGrounder:
    LABELS = ["person", "table", "banknotes", "cash counting machine", "cash drawer", "safe", "bag"]

    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.model = None
        self.processor = None

    @property
    def available(self) -> bool:
        return self.config.grounding_enabled and (self.config.grounding_model_path / "config.json").is_file()

    def load(self) -> None:
        if self.model is not None or not self.available:
            return
        configure_offline_environment()
        from transformers import AutoConfig, AutoModelForZeroShotObjectDetection, AutoProcessor
        path = str(self.config.grounding_model_path)
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        config.disable_custom_kernels = True
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path, config=config, local_files_only=True).eval()

    def calibrate(self, video: SampledVideo, deadline: float) -> tuple[list[SceneRegion], dict]:
        if not self.available:
            return [], {"enabled": False, "reason": "weights_not_provisioned_or_disabled"}
        import torch
        self.load()
        start = time.monotonic()
        regions = []
        indices = sorted({0, len(video.frames) // 2, len(video.frames) - 1})
        for index in indices:
            if time.monotonic() + 5 >= deadline:
                break
            frame = video.frames[index]
            inputs = self.processor(images=frame.image, text=[self.LABELS], return_tensors="pt", size={"shortest_edge": 480, "longest_edge": 800})
            with torch.inference_mode():
                outputs = self.model(**inputs)
            post = self.processor.post_process_grounded_object_detection
            threshold_key = "threshold" if "threshold" in inspect.signature(post).parameters else "box_threshold"
            result = post(outputs, inputs.input_ids, **{threshold_key: .28}, text_threshold=.24, target_sizes=[frame.image.size[::-1]])[0]
            labels = result.get("text_labels", result.get("labels", []))
            for label, score, box in zip(labels, result["scores"], result["boxes"]):
                x0, y0, x1, y1 = box.tolist()
                width, height = frame.image.size
                normalized = (max(0., x0 / width), max(0., y0 / height), min(1., x1 / width), min(1., y1 / height))
                if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
                    continue
                regions.append(SceneRegion(frame.timestamp_sec, str(label), float(score), normalized))
        return regions, {"enabled": True, "seconds": round(time.monotonic() - start, 3), "calibration_frames": len(indices), "regions": len(regions)}


def detail_crops(frames: list[VideoFrame], regions: list[SceneRegion], limit: int = 2) -> list[VideoFrame]:
    """Append context crops near an event, retaining all original full frames."""
    if not frames or not regions or limit <= 0:
        return []
    output = []
    selected = [frames[len(frames) // 2], frames[-1]][:limit]
    for frame in selected:
        calibration_time = min({r.timestamp_sec for r in regions}, key=lambda t: abs(t - frame.timestamp_sec))
        relevant = [r for r in regions if r.timestamp_sec == calibration_time and any(word in r.label for word in ("table", "banknote", "counting", "drawer", "safe", "bag"))]
        if not relevant:
            continue
        # Prefer a table (context) or the most confident work object. A detector
        # miss leaves the full-frame path intact and cannot suppress an event.
        region = max(relevant, key=lambda r: r.confidence + (.2 if "table" in r.label else 0))
        x0, y0, x1, y1 = region.box
        pad_x, pad_y = (x1 - x0) * .15, (y1 - y0) * .15
        x0, y0, x1, y1 = max(0., x0-pad_x), max(0., y0-pad_y), min(1., x1+pad_x), min(1., y1+pad_y)
        area = (x1-x0) * (y1-y0)
        if not .025 < area < .75:
            continue
        w, h = frame.image.size
        crop = frame.image.crop((int(x0*w), int(y0*h), int(x1*w), int(y1*h)))
        output.append(VideoFrame(frame.timestamp_sec, crop, view="detail crop; use the same-time full frame for context"))
    return output
