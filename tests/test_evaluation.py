"""Synthetic evaluation provenance checks; no model or dataset is loaded."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from forte.analysis.config import AnalysisConfig
import scripts.evaluate as evaluation
from scripts.evaluate import build_run_provenance, splits, summarize, validate_resume


@pytest.fixture
def evaluation_run(tmp_path):
    model = tmp_path / "models" / "synthetic-vlm"
    grounding_model = tmp_path / "models" / "synthetic-grounder"
    for directory in (model, grounding_model):
        directory.mkdir(parents=True)
        (directory / "config.json").write_text('{"synthetic": true}', encoding="utf-8")
        (directory / "model.safetensors").write_bytes(b"synthetic weights")
    rules = tmp_path / "config" / "rules.json"
    rules.parent.mkdir()
    rules.write_text('{"rules": []}', encoding="utf-8")
    private_dir = tmp_path / "synthetic_reports"
    private_dir.mkdir()
    (private_dir / "fewshot_examples.json").write_text(
        '{"training_only": true, "examples": []}', encoding="utf-8"
    )
    analysis_dir = tmp_path / "forte" / "analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "pipeline.py").write_text("# synthetic pipeline v1\n", encoding="utf-8")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "evaluate.py").write_text("# synthetic evaluator v1\n", encoding="utf-8")
    config = AnalysisConfig(
        model_path=model,
        grounding_model_path=grounding_model,
        rules_path=rules,
        private_dir=private_dir,
    )
    records = [
        {"id": "synthetic-negative", "path": "synthetic/negative.mp4", "label": 0, "annotated_sec": None},
        {"id": "synthetic-positive", "path": "synthetic/positive.mp4", "label": 1, "annotated_sec": 4.5},
        {"id": "synthetic-extra", "path": "synthetic/extra.mp4", "label": 1, "annotated_sec": 9.0},
    ]
    return {"root": tmp_path, "config": config, "manifest": {"records": records}, "split": "holdout", "selected": records}


def saved_evaluation(run, completed=1):
    provenance = build_run_provenance(**run)
    rows = [
        {
            "id": record["id"],
            "label": record["label"],
            "annotated_sec": record.get("annotated_sec"),
            "predicted": record["label"],
            "seconds": 0.01,
        }
        for record in run["selected"][:completed]
    ]
    return {"provenance": provenance, "split": run["split"], "model": run["config"].model_path.name, "records": rows}


def test_provenance_is_stable_and_json_serializable(evaluation_run):
    first = build_run_provenance(**evaluation_run)
    assert json.loads(json.dumps(first)) == first
    assert first == build_run_provenance(**evaluation_run)
    assert first["schema_version"] == 1
    assert first["selected_ids"] == [r["id"] for r in evaluation_run["selected"]]
    assert len(first["identity_sha256"]) == 64


def test_resume_accepts_same_run_partial_results(evaluation_run):
    payload = saved_evaluation(evaluation_run)
    assert validate_resume(payload, build_run_provenance(**evaluation_run)) == payload["records"]


def test_resume_accepts_ordered_limit_extension(evaluation_run):
    limited = {**evaluation_run, "selected": evaluation_run["selected"][:1]}
    payload = saved_evaluation(limited)
    assert validate_resume(payload, build_run_provenance(**evaluation_run)) == payload["records"]


@pytest.mark.parametrize("change", ["split", "config", "rules", "code", "evaluator", "model", "grounding_model", "manifest", "examples"])
def test_resume_rejects_changed_analysis_inputs(evaluation_run, change):
    payload = saved_evaluation(evaluation_run)
    current_run = {**evaluation_run}
    config = evaluation_run["config"]
    if change == "split":
        current_run["split"] = "train"
    elif change == "config":
        current_run["config"] = replace(config, confidence_threshold=0.9)
    elif change == "manifest":
        current_run["manifest"] = deepcopy(evaluation_run["manifest"])
        current_run["manifest"]["records"][2]["annotated_sec"] = 12.0
    else:
        path = {
            "rules": config.rules_path,
            "code": evaluation_run["root"] / "forte" / "analysis" / "pipeline.py",
            "evaluator": evaluation_run["root"] / "scripts" / "evaluate.py",
            "model": config.model_path / "model.safetensors",
            "grounding_model": config.grounding_model_path / "model.safetensors",
            "examples": config.private_dir / "fewshot_examples.json",
        }[change]
        # Same-length content changes must invalidate a resume too.
        content = path.read_bytes()
        path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    with pytest.raises(ValueError):
        validate_resume(payload, build_run_provenance(**current_run))


def test_resume_rejects_legacy_output(evaluation_run):
    payload = saved_evaluation(evaluation_run)
    del payload["provenance"]
    with pytest.raises(ValueError):
        validate_resume(payload, build_run_provenance(**evaluation_run))


@pytest.mark.parametrize("mutation", ["payload_list", "selected_record_nonobject", "selected_id_nonstring"])
def test_resume_rejects_malformed_payload_shapes(evaluation_run, mutation):
    payload = saved_evaluation(evaluation_run)
    if mutation == "payload_list":
        payload = []
    elif mutation == "selected_record_nonobject":
        payload["provenance"]["selected_records"][0] = "not a record object"
    else:
        payload["provenance"]["selected_ids"][0] = ["not a string ID"]
    with pytest.raises(ValueError):
        validate_resume(payload, build_run_provenance(**evaluation_run))


@pytest.mark.parametrize("seconds", [float("nan"), -1.0, "1.0"])
def test_resume_rejects_invalid_timing_values(evaluation_run, seconds):
    payload = saved_evaluation(evaluation_run)
    payload["records"][0]["seconds"] = seconds
    with pytest.raises(ValueError):
        validate_resume(payload, build_run_provenance(**evaluation_run))


@pytest.mark.parametrize("side", ["saved", "current"])
def test_resume_rejects_tampered_identity_hash(evaluation_run, side):
    payload = saved_evaluation(evaluation_run)
    current = build_run_provenance(**evaluation_run)
    target = payload["provenance"] if side == "saved" else current
    target["identity_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_resume(payload, current)


def test_resume_rejects_modified_identity_even_if_digest_unchanged(evaluation_run):
    payload = saved_evaluation(evaluation_run)
    payload["provenance"]["identity"]["split"] = "train"
    with pytest.raises(ValueError):
        validate_resume(payload, build_run_provenance(**evaluation_run))


@pytest.mark.parametrize("selection", ["shorter", "reordered"])
def test_resume_rejects_nonextension_selection(evaluation_run, selection):
    payload = saved_evaluation(evaluation_run)
    selected = evaluation_run["selected"]
    changed = selected[:2] if selection == "shorter" else list(reversed(selected))
    current = build_run_provenance(**{**evaluation_run, "selected": changed})
    with pytest.raises(ValueError):
        validate_resume(payload, current)


@pytest.mark.parametrize("mutation", ["foreign_id", "duplicate", "label", "annotation"])
def test_resume_rejects_contaminated_rows(evaluation_run, mutation):
    payload = saved_evaluation(evaluation_run, completed=2)
    if mutation == "foreign_id":
        payload["records"][0]["id"] = "outside-selected-split"
    elif mutation == "duplicate":
        payload["records"].append(deepcopy(payload["records"][0]))
    elif mutation == "label":
        payload["records"][0]["label"] = 1
    else:
        payload["records"][1]["annotated_sec"] = 123.0
    with pytest.raises(ValueError):
        validate_resume(payload, build_run_provenance(**evaluation_run))


def test_resume_cannot_claim_unselected_extension_records(evaluation_run):
    limited = {**evaluation_run, "selected": evaluation_run["selected"][:1]}
    payload = saved_evaluation(limited)
    payload["records"].append(saved_evaluation(evaluation_run, completed=2)["records"][1])
    with pytest.raises(ValueError):
        validate_resume(payload, build_run_provenance(**evaluation_run))


def test_save_rejects_mismatched_run_without_changing_output(evaluation_run):
    output = evaluation_run["root"] / "evaluation.json"
    previous = saved_evaluation(evaluation_run)
    old_bytes = json.dumps(previous, indent=2).encode("utf-8")
    output.write_bytes(old_bytes)
    changed_run = {**evaluation_run, "split": "train"}
    with pytest.raises(ValueError):
        evaluation._save_payload(output, saved_evaluation(changed_run, completed=2))
    assert output.read_bytes() == old_bytes
    assert not list(output.parent.glob("*.tmp"))


def test_save_replaces_valid_extension_atomically(evaluation_run, monkeypatch):
    output = evaluation_run["root"] / "evaluation.json"
    limited = {**evaluation_run, "selected": evaluation_run["selected"][:1]}
    previous = saved_evaluation(limited)
    output.write_text(json.dumps(previous), encoding="utf-8")
    extended = saved_evaluation(evaluation_run, completed=2)
    original_replace = evaluation.os.replace
    replacements = []

    def observe_replace(source, destination):
        # The previous document stays intact until the complete new document
        # exists beside it and is handed to the atomic replacement primitive.
        assert json.loads(output.read_text(encoding="utf-8")) == previous
        assert source.parent == output.parent
        assert json.loads(source.read_text(encoding="utf-8")) == extended
        replacements.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr(evaluation.os, "replace", observe_replace)
    evaluation._save_payload(output, extended)
    assert len(replacements) == 1
    assert json.loads(output.read_text(encoding="utf-8")) == extended
    assert not list(output.parent.glob("*.tmp"))


@pytest.mark.parametrize("mutation", ["shorter", "changed_prediction"])
def test_save_rejects_regressive_or_changed_completed_rows(evaluation_run, mutation):
    output = evaluation_run["root"] / "evaluation.json"
    previous = saved_evaluation(evaluation_run, completed=2)
    old_bytes = json.dumps(previous).encode("utf-8")
    output.write_bytes(old_bytes)
    incoming = deepcopy(previous)
    if mutation == "shorter":
        incoming["records"].pop()
    else:
        incoming["records"][0]["predicted"] = 1
    with pytest.raises(ValueError):
        evaluation._save_payload(output, incoming)
    assert output.read_bytes() == old_bytes
    assert not list(output.parent.glob("*.tmp"))


def test_existing_output_without_resume_fails_before_manifest_read(tmp_path, monkeypatch):
    output = tmp_path / "existing-evaluation.json"
    old_bytes = b"preserve this synthetic output exactly"
    output.write_bytes(old_bytes)
    monkeypatch.setattr(evaluation.sys, "argv", [
        "evaluate.py", "--manifest", str(tmp_path / "nonexistent-manifest.json"),
        "--output", str(output),
    ])
    with pytest.raises(SystemExit) as failure:
        evaluation.main()
    assert failure.value.code == 2
    assert output.read_bytes() == old_bytes


def test_completed_resume_does_not_construct_analyzer(evaluation_run, monkeypatch, capsys):
    run = {**evaluation_run, "selected": splits(evaluation_run["manifest"]["records"])["holdout"]}
    output = run["root"] / "evaluation.json"
    manifest_path = run["root"] / "synthetic-manifest.json"
    manifest_path.write_text(json.dumps(run["manifest"]), encoding="utf-8")
    previous = saved_evaluation(run, completed=len(run["selected"]))
    output.write_text(json.dumps(previous), encoding="utf-8")

    def forbidden_analyzer(*args, **kwargs):
        pytest.fail("A completed resume must not construct or load an analyzer")

    monkeypatch.setitem(evaluation.sys.modules, "forte.analysis.pipeline", SimpleNamespace(VideoAnalyzer=forbidden_analyzer))
    monkeypatch.setattr(AnalysisConfig, "from_env", classmethod(lambda cls: run["config"]))
    original_provenance_builder = evaluation.build_run_provenance
    monkeypatch.setattr(evaluation, "build_run_provenance", lambda **kwargs: original_provenance_builder(**kwargs, root=run["root"]))
    monkeypatch.setattr(evaluation.sys, "argv", [
        "evaluate.py", "--manifest", str(manifest_path), "--output", str(output), "--resume",
    ])
    evaluation.main()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["event"] == "already_complete"
    assert json.loads(output.read_text(encoding="utf-8"))["records"] == previous["records"]


def test_failed_positives_remain_false_negatives():
    result = summarize([
        {"label": 1, "predicted": 1, "seconds": 1.0},
        {"label": 1, "predicted": None, "seconds": 2.0, "error_type": "TimeoutError"},
        {"label": 0, "predicted": 0, "seconds": 1.0},
        {"label": 0, "predicted": None, "seconds": 2.0, "error_type": "ValueError"},
    ])
    assert result["recall"] == 0.5
    assert result["accuracy"] == 0.5
    assert result["completed_accuracy"] == 1.0
    assert result["fn"] == result["failed_positive"] == 1
    assert result["failed"] == 2


def test_splits_are_stable_disjoint_and_interleaved():
    records = [{"id": f"synthetic-{label}-{i}", "label": label} for label in (0, 1) for i in range(7)]
    result = splits(records)
    assert result == splits(list(reversed(records)))
    assert [r["label"] for r in result["holdout"]] == [0, 1] * 4
    assert [r["label"] for r in result["train"]] == [0, 1] * 3
    assert {r["id"] for r in result["train"]}.isdisjoint(r["id"] for r in result["holdout"])
    assert len(result["all"]) == len(records)
