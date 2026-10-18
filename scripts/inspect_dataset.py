"""Create a content-free video/label manifest using container metadata only.

Never decodes frames, extracts audio, or prints video-derived descriptions.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def parse_timestamp(value: str | None) -> float | None:
    """Parse the supplied Russian textual timestamps without guessing missing data."""
    if value is None or not str(value).strip():
        return None
    value = str(value).strip().lower().replace(",", ".")
    match = re.fullmatch(r"(?:(\d+(?:\.\d+)?)\s*минут\w*\s*)?(?:(\d+(?:\.\d+)?)\s*секунд\w*\s*)?", value)
    if match and any(match.groups()):
        return float(match.group(1) or 0) * 60 + float(match.group(2) or 0)
    if re.fullmatch(r"\d+(?::\d{1,2}){1,2}(?:\.\d+)?", value):
        total = 0.0
        for part in value.split(":"):
            total = total * 60 + float(part)
        return total
    raise ValueError(f"Unsupported timestamp format: {value!r}")


def read_labels(path: Path) -> list[dict]:
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(e.itertext()) for e in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si", NS)]
        def cell_value(cell):
            if cell.get("t") == "inlineStr":
                return "".join(cell.find("m:is", NS).itertext())
            element = cell.find("m:v", NS)
            if element is None:
                return None
            return strings[int(element.text)] if cell.get("t") == "s" else element.text
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = [{re.sub(r"\d", "", cell.get("r")): cell_value(cell) for cell in row} for row in sheet.findall("m:sheetData/m:row", NS)]
    expected = {"A": "наименование файла", "B": "таймстемп", "C": "нарушение / без нарушения"}
    if rows[0] != expected:
        raise ValueError("Unexpected label headers; refusing to reinterpret the sheet")
    records = []
    for row in rows[1:]:
        label_text = row.get("C")
        if label_text not in {"нарушение", "без нарушения"}:
            raise ValueError("Unrecognized binary label")
        label = int(label_text == "нарушение")
        annotated_sec = parse_timestamp(row.get("B"))
        if bool(label) != (annotated_sec is not None):
            raise ValueError("Unexpected label/timestamp completeness")
        records.append({"id": row["A"], "label": label, "label_text": label_text, "annotated_sec": annotated_sec})
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate video identifiers")
    return records


def probe_av(path: Path) -> dict:
    """Read PyAV container/stream headers; deliberately never call decode()."""
    import av
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream: {path.name}")
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else float(container.duration / av.time_base)
        context = stream.codec_context
        return {"duration_sec": duration, "width": context.width, "height": context.height, "fps": float(stream.average_rate) if stream.average_rate else None, "frame_count": stream.frames or None, "video_codec": context.name, "pixel_format": context.format.name if context.format else None, "has_audio": bool(container.streams.audio), "file_size_bytes": path.stat().st_size}


def probe(path: Path, ffprobe: Path | None) -> dict:
    if ffprobe is None:
        return probe_av(path)
    args = [str(ffprobe), "-v", "error", "-show_entries", "format=duration,size:stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,pix_fmt,duration", "-of", "json", str(path)]
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", check=True, timeout=30)
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if not videos:
        raise ValueError(f"No video stream: {path.name}")
    video = videos[0]
    duration = float(video.get("duration") or info.get("format", {}).get("duration"))
    rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    fps = float(Fraction(rate)) if rate != "0/0" else None
    frames = video.get("nb_frames")
    return {"duration_sec": duration, "width": int(video["width"]), "height": int(video["height"]), "fps": fps, "frame_count": int(frames) if frames and frames != "N/A" else None, "video_codec": video.get("codec_name"), "pixel_format": video.get("pix_fmt"), "has_audio": any(s.get("codec_type") == "audio" for s in streams), "file_size_bytes": path.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "Dataset", help="Dataset directory inside the project root; manifest paths remain project-relative")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "dataset_manifest.json")
    parser.add_argument("--ffprobe", type=Path, default=ROOT / "tools" / "ffmpeg" / "bin" / "ffprobe.exe")
    args = parser.parse_args()
    try:
        args.dataset.resolve().relative_to(ROOT)
    except ValueError:
        parser.error("--dataset must be inside the project root because manifest paths are project-relative")
    ffprobe = args.ffprobe if args.ffprobe.is_file() else Path(shutil.which("ffprobe") or "")
    if not ffprobe.is_file():
        ffprobe = None
    labels_path = args.dataset / "open_labels.xlsx"
    records = read_labels(labels_path)
    normalize = lambda value: unicodedata.normalize("NFC", value).casefold()
    paths = list((args.dataset / "videos").glob("*.mov"))
    by_stem = {normalize(path.stem): path for path in paths}
    if len(by_stem) != len(paths):
        raise ValueError("Filename collision after NFC/casefold normalization")
    for record in records:
        path = by_stem.pop(normalize(record["id"]), None)
        if path is None:
            raise ValueError(f"No exact Unicode filename match for {record['id']!r}")
        record["path"] = path.resolve().relative_to(ROOT).as_posix()
        record["metadata"] = probe(path, ffprobe)
        if record["annotated_sec"] is not None and record["annotated_sec"] > record["metadata"]["duration_sec"]:
            raise ValueError(f"Annotation lies beyond video duration: {record['id']}")
    if by_stem:
        raise ValueError("Unlabelled video files remain")
    durations = [record["metadata"]["duration_sec"] for record in records]
    summary = {"video_count": len(records), "positive_count": sum(r["label"] for r in records), "negative_count": sum(not r["label"] for r in records), "total_duration_sec": sum(durations), "min_duration_sec": min(durations), "max_duration_sec": max(durations), "total_bytes": sum(r["metadata"]["file_size_bytes"] for r in records), "codecs": dict(Counter(r["metadata"]["video_codec"] for r in records)), "resolutions": dict(Counter(f"{r['metadata']['width']}x{r['metadata']['height']}" for r in records)), "audio_video_count": sum(r["metadata"]["has_audio"] for r in records)}
    manifest = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(), "dataset_root": args.dataset.resolve().relative_to(ROOT).as_posix(), "labels_path": labels_path.resolve().relative_to(ROOT).as_posix(), "label_semantics": "Binary video label plus one point timestamp for positive videos. No violation-type or interval ground truth is supplied.", "records": records, "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_suffix(args.output.suffix + ".tmp")
    pending.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(args.output)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
