"""Promote human-confirmed training examples into a local-only prompt cache.

No images or observations are printed. Holdout and unsupported rules are always
excluded. Source filenames and weak binary labels are not included in prompts.
"""

import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate import splits


def main():
    source=ROOT/"runtime/review_labels.json"
    if not source.exists():
        print(json.dumps({"examples":0,"reason":"no_human_reviews"}))
        return
    manifest=json.loads((ROOT/"data/dataset_manifest.json").read_text(encoding="utf-8-sig"))
    allowed_ids={r["id"] for r in splits(manifest["records"])["train"]}
    rules=json.loads((ROOT/"config/normative_rules_rk.json").read_text(encoding="utf-8"))["rules"]
    known={r["id"] for r in rules if r.get("supported") is True}
    reviews=json.loads(source.read_text(encoding="utf-8"))["records"]
    examples=[]
    counts={}
    for video_id,record in reviews.items():
        if video_id not in allowed_ids or record.get("calibration_allowed") is not True or not record.get("review_complete"):
            continue
        for event in record.get("events",[]):
            rule_id=event.get("rule_id")
            description=event.get("description","").strip()
            if event.get("confirmed") is True and rule_id in known and len(description)>=8 and counts.get(rule_id,0)<2:
                examples.append({"rule_id":rule_id,"observed_example":description[:600],"human_validated":True})
                counts[rule_id]=counts.get(rule_id,0)+1
    target=ROOT/"runtime/private/fewshot_examples.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps({"schema_version":1,"training_only":True,"examples":examples},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"examples":len(examples),"classes":len(counts),"human_validated":True,"holdout_excluded":True}))


if __name__=="__main__":
    main()
