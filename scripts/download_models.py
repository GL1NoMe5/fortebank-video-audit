"""Download approved public model artifacts; never reads local dataset media."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCES_PATH = ROOT / "data" / "model_sources.json"
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(ROOT / ".cache" / "huggingface" / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(ROOT / ".cache" / "huggingface" / "xet"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "0")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

APPROVED = ("Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct", "IDEA-Research/grounding-dino-tiny")
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja", "*.jinja2", "LICENSE*", "NOTICE*", "README.md"]


def segmented_weights(repo: str, revision: str, filename: str, destination: Path, workers: int, recover_integrity: bool = True) -> dict:
    """Resume exact byte ranges from the official CDN and verify LFS SHA-256."""
    import requests
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    metadata = get_hf_file_metadata(hf_hub_url(repo, filename, revision=revision))
    expected_size = int(metadata.size)
    expected_sha = metadata.etag.strip('"')
    if len(expected_sha) != 64:
        raise RuntimeError("Expected a SHA-256 LFS object identifier")
    target = destination / filename
    def sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if target.is_file() and target.stat().st_size == expected_size and sha256(target) == expected_sha:
        return {"filename": filename, "bytes": expected_size, "sha256": expected_sha}
    cache = ROOT / ".cache" / "model-downloads" / repo.replace("/", "--") / revision / filename
    cache.mkdir(parents=True, exist_ok=True)
    chunk_size = 32 * 1024 * 1024
    segments = [(index, start, min(start + chunk_size, expected_size) - 1) for index, start in enumerate(range(0, expected_size, chunk_size))]
    started = time.monotonic()
    def fetch(segment):
        index, start, end = segment
        size = end - start + 1
        part = cache / f"{index:05d}.part"
        if part.exists() and part.stat().st_size == size:
            return size
        pending = part.with_suffix(".pending")
        for attempt in range(5):
            try:
                with requests.get(metadata.location, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(15, 90)) as response:
                    response.raise_for_status()
                    expected_range = f"bytes {start}-{end}/{expected_size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError("CDN did not honor the exact requested byte range")
                    written = 0
                    with pending.open("wb") as handle:
                        for block in response.iter_content(1024 * 1024):
                            written += len(block)
                            if written > size:
                                raise RuntimeError("CDN returned more bytes than requested")
                            handle.write(block)
                    if written != size:
                        raise RuntimeError("Incomplete byte range")
                pending.replace(part)
                return size
            except Exception as exc:
                if attempt == 4:
                    raise RuntimeError(f"Byte-range download failed for {repo}/{filename}, segment {index}: {type(exc).__name__}") from None
                time.sleep(min(2 ** attempt, 16))
    done = 0
    last_report = 0.0
    print(json.dumps({"event": "weights_start", "repo": repo, "file": filename, "bytes": expected_size, "segments": len(segments), "workers": workers}), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, segment) for segment in segments]
        for future in as_completed(futures):
            done += future.result()
            elapsed = time.monotonic() - started
            if elapsed - last_report >= 20 or done == expected_size:
                print(json.dumps({"event": "weights_progress", "repo": repo, "file": filename, "completed_bytes": done, "total_bytes": expected_size, "elapsed_sec": round(elapsed, 1)}), flush=True)
                last_report = elapsed
    assembling = target.with_suffix(".assembling")
    digest = hashlib.sha256()
    with assembling.open("wb") as output:
        for index, _, _ in segments:
            with (cache / f"{index:05d}.part").open("rb") as part:
                for block in iter(lambda: part.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    output.write(block)
    if assembling.stat().st_size != expected_size or digest.hexdigest() != expected_sha:
        # A full-sized corrupt cached chunk cannot be diagnosed by length alone.
        # Invalidate this weight's own parts so a retry never repeats bad bytes.
        for part in cache.glob("*.part"):
            part.unlink()
        assembling.unlink(missing_ok=True)
        if recover_integrity:
            print(json.dumps({"event": "weights_integrity_retry", "repo": repo, "file": filename}), flush=True)
            return segmented_weights(repo, revision, filename, destination, workers, recover_integrity=False)
        raise RuntimeError(f"Weight integrity check failed for {repo}/{filename}")
    assembling.replace(target)
    print(json.dumps({"event": "weights_verified", "repo": repo, "file": filename, "bytes": expected_size, "sha256": expected_sha}), flush=True)
    return {"filename": filename, "bytes": expected_size, "sha256": expected_sha}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", choices=APPROVED, nargs="+", default=list(APPROVED))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--refresh-revision", action="store_true", help="Explicitly use the current upstream revision instead of the project or saved revision")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    registered_models = json.loads(MODEL_SOURCES_PATH.read_text(encoding="utf-8"))["models"]
    for repo in args.repo:
        destination = ROOT / "models" / repo.split("/")[-1]
        record = destination / "download_provenance.json"
        pinned_revision = None if args.refresh_revision else registered_models[repo]["revision"]
        if record.is_file() and not args.refresh_revision:
            previous = json.loads(record.read_text(encoding="utf-8"))
            pinned_revision = previous.get("revision")
            if previous.get("repo_id") != repo or not isinstance(pinned_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", pinned_revision):
                raise RuntimeError(f"Invalid saved model provenance for {repo}; refusing to silently change revision")
        if pinned_revision is not None and (not isinstance(pinned_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", pinned_revision)):
            raise RuntimeError(f"Invalid project model revision for {repo}")
        info = api.model_info(repo, revision=pinned_revision)
        card = info.card_data.to_dict() if info.card_data is not None else {}
        license_name = card.get("license")
        if license_name != "apache-2.0":
            raise RuntimeError(f"Unapproved license metadata for {repo}: {license_name!r}")
        destination.mkdir(parents=True, exist_ok=True)
        provenance = {"repo_id": repo, "revision": info.sha, "license": license_name, "license_evidence": f"https://huggingface.co/{repo}/blob/{info.sha}/README.md", "download_started_at": datetime.now(timezone.utc).isoformat(), "status": "downloading", "allow_patterns": PATTERNS}
        record.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"event": "download_start", "repo": repo, "revision": info.sha, "license": license_name}), flush=True)
        snapshot_download(repo, revision=info.sha, local_dir=str(destination), allow_patterns=[p for p in PATTERNS if p != "*.safetensors"], max_workers=args.workers)
        provenance["weight_integrity"] = []
        for entry in info.siblings:
            if entry.rfilename.endswith(".safetensors"):
                provenance["weight_integrity"].append(segmented_weights(repo, info.sha, entry.rfilename, destination, args.workers))
        licenses = [p for p in destination.glob("LICENSE*") if p.is_file()]
        if not licenses:
            license_url = "https://www.apache.org/licenses/LICENSE-2.0.txt"
            text = urllib.request.urlopen(license_url, timeout=60).read()
            (destination / "LICENSE-APACHE-2.0.txt").write_bytes(text)
            provenance["license_text_source"] = license_url
            provenance["license_text_note"] = "Canonical license text; repository applicability is declared by the model card metadata linked above."
        provenance["status"] = "complete"
        provenance["download_completed_at"] = datetime.now(timezone.utc).isoformat()
        provenance["files"] = [{"name": path.relative_to(destination).as_posix(), "bytes": path.stat().st_size} for path in sorted(destination.iterdir()) if path.is_file() and path != record]
        record.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"event": "download_complete", "repo": repo, "revision": info.sha, "files": len(provenance["files"]), "bytes": sum(item["bytes"] for item in provenance["files"])}), flush=True)


if __name__ == "__main__":
    main()
