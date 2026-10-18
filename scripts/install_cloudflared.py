"""Install a pinned, SHA-256 verified official cloudflared Windows binary."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/cloudflare/cloudflared/releases/download/2026.10.1/cloudflared-windows-amd64.exe"
SIZE = 55325120
SHA256 = "4d2490e74082c5d48dbfedaf92576fa5bd1702379ab88eb7faa0cfbe8dc891b6"


def main():
    target = ROOT / "tools/cloudflared/cloudflared.exe"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == SIZE and hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
        print("cloudflared 2026.10.1 already verified")
        return
    cache = ROOT / ".cache/cloudflared-2026.10.1"
    cache.mkdir(parents=True, exist_ok=True)
    chunk = 2 * 1024 * 1024
    segments = [(i, start, min(start+chunk, SIZE)-1) for i, start in enumerate(range(0, SIZE, chunk))]
    def fetch(segment):
        index, start, end = segment
        path = cache / f"{index:04}.part"
        if path.exists() and path.stat().st_size == end-start+1:
            return path
        for attempt in range(4):
            try:
                response = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=(15, 90))
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                    raise RuntimeError("Unexpected binary response range")
                if len(response.content) != end-start+1:
                    raise RuntimeError("Truncated binary segment")
                path.write_bytes(response.content)
                return path
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(fetch, segments))
    pending = target.with_suffix(".pending")
    with pending.open("wb") as handle:
        for path in paths:
            handle.write(path.read_bytes())
    if pending.stat().st_size != SIZE or hashlib.sha256(pending.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("cloudflared SHA-256 mismatch")
    pending.replace(target)
    (target.parent / "provenance.json").write_text(json.dumps({"version":"2026.10.1", "url":URL,"bytes":SIZE,"sha256":SHA256,"license":"Apache-2.0","license_url":"https://github.com/cloudflare/cloudflared/blob/2026.10.1/LICENSE"}, indent=2), encoding="utf-8")
    print("cloudflared 2026.10.1 downloaded and SHA-256 verified")


if __name__ == "__main__":
    main()
