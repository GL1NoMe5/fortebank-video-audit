"""Local synthetic camera-perturbation checks; output contains metrics only.

Image-space transforms are a limited stress test, not evidence of performance on
a genuinely new camera with different occlusions or view geometry.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate import splits
from forte.analysis.config import AnalysisConfig
from forte.analysis.pipeline import VideoAnalyzer
from forte.analysis.video import SampledVideo, VideoFrame, sample_video
import forte.analysis.pipeline as pipeline


def transformed(video, variant):
    frames = []
    for frame in video.frames:
        if variant == "mirror":
            image = frame.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif variant == "rotate_12":
            image = frame.image.rotate(12, expand=False, fillcolor=(32,32,32))
        elif variant == "perspective":
            array = np.asarray(frame.image)
            h,w = array.shape[:2]
            source = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
            destination = np.float32([[.12*w,.04*h],[.96*w,.12*h],[.88*w,.96*h],[.04*w,.9*h]])
            matrix = cv2.getPerspectiveTransform(source,destination)
            image = Image.fromarray(cv2.warpPerspective(array,matrix,(w,h),borderValue=(32,32,32)))
        else:
            image = frame.image
        frames.append(replace(frame,image=image))
    return replace(video,frames=frames)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--limit",type=int,default=4)
    parser.add_argument("--output",type=Path,default=ROOT/"runtime/robustness.json")
    args=parser.parse_args()
    records=splits(json.loads((ROOT/"data/dataset_manifest.json").read_text(encoding="utf-8-sig"))["records"])["holdout"][:args.limit]
    config=AnalysisConfig.from_env()
    config=replace(config,private_dir=config.private_dir/"robustness")
    analyzer=VideoAnalyzer(config)
    analyzer.warmup()
    results=[]
    original=pipeline.sample_video
    try:
        for record in records:
            video=sample_video(ROOT/record["path"],fps=config.scan_fps)
            baseline=None
            baseline_times=[]
            for variant in ("original","mirror","rotate_12","perspective"):
                altered=transformed(video,variant)
                pipeline.sample_video=lambda *a,**k: altered
                started=time.monotonic()
                row={"id":record["id"],"variant":variant,"label":record["label"]}
                try:
                    result=analyzer.analyze(ROOT/record["path"])
                    prediction=int(result["verdict"]=="violation")
                    times=[v["timecode_sec"] for v in result["violations"]]
                    if variant=="original":
                        baseline,baseline_times=prediction,times
                    row.update(predicted=prediction,event_count=len(times))
                    if variant != "original" and baseline is not None:
                        row["consistent_with_original"] = prediction == baseline
                    if times and baseline_times:
                        row["nearest_event_shift_sec"]=max(min(abs(t-b) for b in baseline_times) for t in times)
                except Exception as exc:
                    row["error_type"]=type(exc).__name__
                row["seconds"]=round(time.monotonic()-started,3)
                results.append(row)
                comparisons=[r for r in results if r["variant"]!="original" and "consistent_with_original" in r]
                summary={"runs":len(results),"failed":sum("error_type" in r for r in results),"comparison_count":len(comparisons),"consistent":sum(r["consistent_with_original"] for r in comparisons),"consistency_rate":sum(r["consistent_with_original"] for r in comparisons)/len(comparisons) if comparisons else None,"max_seconds":max(r["seconds"] for r in results)}
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps({"records":results,"summary":summary,"limitation":"Image transforms do not reproduce unseen camera viewpoints."},ensure_ascii=False,indent=2),encoding="utf-8")
                print(json.dumps(summary),flush=True)
    finally:
        pipeline.sample_video=original


if __name__=="__main__":
    main()
