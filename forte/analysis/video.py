"""Video sampling with actual presentation timestamps and motion coverage."""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import av
import cv2
import numpy as np
from PIL import Image


@dataclass
class VideoFrame:
    timestamp_sec: float
    image: Image.Image
    motion: float = 0.0
    sharpness: float = 0.0
    view: str = "full frame"


@dataclass
class SampledVideo:
    frames: list[VideoFrame]
    duration_sec: float
    width: int
    height: int


def resize_pixels(image: Image.Image, max_pixels: int, max_edge: int = 1280) -> Image.Image:
    ratio = min(1.0, math.sqrt(max_pixels / (image.width * image.height)), max_edge / max(image.size))
    if ratio >= 1:
        return image
    return image.resize((max(32, round(image.width * ratio)), max(32, round(image.height * ratio))), Image.Resampling.LANCZOS)


def sample_video(path: Path, fps: float = 2.0, max_duration: float = 180.0) -> SampledVideo:
    if fps <= 0:
        raise ValueError("Sampling fps must be positive")
    frames: list[VideoFrame] = []
    previous = None
    next_time = 0.0
    last_time = 0.0
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("No video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        width, height = stream.codec_context.width, stream.codec_context.height
        if width <= 0 or height <= 0 or width * height > 4096 * 4096:
            raise ValueError("Unsupported video dimensions")
        declared_duration = float(stream.duration * stream.time_base) if stream.duration is not None else (float(container.duration / av.time_base) if container.duration else 0.0)
        if declared_duration > max_duration + 1:
            raise ValueError("Video exceeds supported duration")
        origin = None
        for index, frame in enumerate(container.decode(stream)):
            raw_time = float(frame.time) if frame.time is not None else index / float(stream.average_rate or 30)
            if origin is None:
                origin = raw_time
            timestamp = max(0.0, raw_time - origin)
            last_time = timestamp
            if timestamp > max_duration:
                raise ValueError("Video exceeds supported duration")
            if timestamp + 1e-6 < next_time:
                continue
            next_time = timestamp + 1.0 / fps
            image = frame.to_image().convert("RGB")
            rotation = int(frame.rotation or 0)
            if rotation:
                image = image.rotate(rotation, expand=True)
            image = resize_pixels(image, 1280 * 720)
            small = np.asarray(image.resize((160, 96)).convert("L"), dtype=np.float32)
            # Subtract median intensity to reduce exposure-driven false peaks.
            normalized = small - np.median(small)
            motion = float(np.mean(np.abs(normalized - previous)) / 255) if previous is not None else 0.0
            previous = normalized
            sharpness = float(cv2.Laplacian(small.astype(np.uint8), cv2.CV_32F).var())
            frames.append(VideoFrame(round(timestamp, 4), image, motion, sharpness))
    if len(frames) < 2:
        raise ValueError("Video cannot be decoded or is too short")
    duration = max(last_time, declared_duration, frames[-1].timestamp_sec)
    return SampledVideo(frames, duration, width, height)


def overview_frames(video: SampledVideo, count: int) -> list[VideoFrame]:
    """Cover the whole timeline, adding diverse motion peaks without fixed zones."""
    n = len(video.frames)
    if n <= count:
        return video.frames[:]
    uniform_count = max(2, int(count * 0.75))
    selected = set(np.linspace(0, n - 1, uniform_count, dtype=int).tolist())
    min_gap = max(1, n // (count * 2))
    ranked = sorted(range(n), key=lambda i: video.frames[i].motion, reverse=True)
    for i in ranked:
        if all(abs(i - j) >= min_gap for j in selected):
            selected.add(i)
        if len(selected) >= count:
            break
    if len(selected) < count:
        for i in np.linspace(0, n - 1, count, dtype=int):
            selected.add(int(i))
            if len(selected) >= count:
                break
    return [video.frames[i] for i in sorted(selected)]


def window_frames(video: SampledVideo, start: float, end: float, count: int, center: float | None = None) -> list[VideoFrame]:
    indices = [i for i, frame in enumerate(video.frames) if start <= frame.timestamp_sec <= end]
    if not indices:
        target = (start + end) / 2
        return [min(video.frames, key=lambda f: abs(f.timestamp_sec - target))]
    selected = {indices[int(i)] for i in np.linspace(0, len(indices) - 1, min(count, len(indices)), dtype=int)}
    if center is not None:
        center_idx = min(indices, key=lambda i: abs(video.frames[i].timestamp_sec - center))
        if center_idx not in selected and len(selected) >= count:
            replaceable = selected - {min(selected), max(selected)}
            if replaceable:
                selected.remove(min(replaceable, key=lambda i: abs(video.frames[i].timestamp_sec - center)))
        selected.add(center_idx)
    return [video.frames[i] for i in sorted(selected)]


def evidence_frames(video: SampledVideo, timestamps: Iterable[float], count: int = 3) -> list[dict]:
    selected: dict[float, VideoFrame] = {}
    for timestamp in timestamps:
        frame = min(video.frames, key=lambda f: abs(f.timestamp_sec - timestamp))
        selected.setdefault(frame.timestamp_sec, frame)
        if len(selected) >= count:
            break
    output = []
    for timestamp, frame in sorted(selected.items()):
        image = resize_pixels(frame.image, 960 * 720, max_edge=1280)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=83, optimize=True)
        output.append({"timestamp_sec": round(timestamp, 3), "image_base64": base64.b64encode(buf.getvalue()).decode("ascii")})
    return output
