"""Small synthetic, local-only multimodal smoke test (no dataset content)."""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from PIL import Image, ImageDraw
    from forte.analysis.config import AnalysisConfig
    from forte.analysis.video import VideoFrame
    from forte.analysis.vlm import LocalVLM

    original_connect = socket.socket.connect

    def offline_connect(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "localhost", "::1"):
            raise RuntimeError("Offline smoke test blocked outbound network")
        return original_connect(sock, address)

    socket.socket.connect = offline_connect
    img = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((30, 60, 130, 160), fill="red")
    draw.ellipse((190, 60, 290, 160), fill="blue")
    runner = LocalVLM(AnalysisConfig.from_env())
    runner.load()
    answer, stats, _ = runner.generate(
        [VideoFrame(0.0, img)],
        'Return JSON only: {"left_shape":"...","left_color":"...","right_shape":"...","right_color":"..."}. Identify the two colored shapes.',
        max_pixels=112896,
        max_tokens=100,
        deadline=time.monotonic() + 60,
    )
    assert answer.get("left_color", "").lower() == "red", "Synthetic vision test failed"
    assert answer.get("right_color", "").lower() == "blue", "Synthetic vision test failed"
    print(json.dumps({"passed": True, "offline": True, "model": runner.config.model_path.name, "load_seconds": runner.load_seconds, "stats": stats, "answer": answer}))


if __name__ == "__main__":
    main()

