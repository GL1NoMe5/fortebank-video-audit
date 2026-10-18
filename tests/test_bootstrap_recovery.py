"""Recovery tests use tiny public-weight stand-ins, never dataset media or GPU."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("has_provenance,refresh", [(True, False), (False, False), (True, True)])
def test_model_revision_is_reproducible(monkeypatch, tmp_path, has_provenance, refresh):
    import huggingface_hub
    module = load_script("download_models")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    repo = "Qwen/Qwen3-VL-4B-Instruct"
    saved_revision = "a" * 40
    registered_revision = json.loads(module.MODEL_SOURCES_PATH.read_text())["models"][repo]["revision"]
    requested_revision = None if refresh else (saved_revision if has_provenance else registered_revision)
    resolved_revision = requested_revision or "b" * 40
    destination = tmp_path / "models" / "Qwen3-VL-4B-Instruct"
    destination.mkdir(parents=True)
    (destination / "LICENSE").write_text("test license")
    if has_provenance:
        (destination / "download_provenance.json").write_text(json.dumps({"repo_id": repo, "revision": saved_revision}))
    requested = []
    class API:
        def model_info(self, repo_id, revision=None):
            requested.append((repo_id, revision))
            return SimpleNamespace(sha=resolved_revision, card_data=SimpleNamespace(to_dict=lambda: {"license": "apache-2.0"}), siblings=[])
    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["download_models.py", "--repo", repo] + (["--refresh-revision"] if refresh else []))
    module.main()
    assert requested == [(repo, requested_revision)]
    assert json.loads((destination / "download_provenance.json").read_text())["revision"] == resolved_revision


@pytest.mark.parametrize("server_bytes,cached,expected_requests", [(b"good", True, 1), (b"evil", True, 1), (b"evil", False, 2)])
def test_same_size_corrupt_chunks_are_recoverable(monkeypatch, tmp_path, server_bytes, cached, expected_requests):
    import huggingface_hub
    import requests
    module = load_script("download_models")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    repo, revision, filename = "Qwen/Qwen3-VL-4B-Instruct", "a" * 40, "model.safetensors"
    good_sha = hashlib.sha256(b"good").hexdigest()
    metadata = SimpleNamespace(size=4, etag=good_sha, location="https://example.invalid/weights")
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(huggingface_hub, "hf_hub_url", lambda *args, **kwargs: "https://example.invalid/metadata")
    destination = tmp_path / "models" / "test"
    destination.mkdir(parents=True)
    cache = tmp_path / ".cache" / "model-downloads" / repo.replace("/", "--") / revision / filename
    cache.mkdir(parents=True)
    if cached:
        (cache / "00000.part").write_bytes(b"evil")
    requests_made = []
    class Response:
        status_code = 206
        headers = {"Content-Range": "bytes 0-3/4"}
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def raise_for_status(self): pass
        def iter_content(self, size): yield server_bytes
    def get(*args, **kwargs):
        requests_made.append(kwargs["headers"]["Range"])
        return Response()
    monkeypatch.setattr(requests, "get", get)
    if server_bytes == b"good":
        result = module.segmented_weights(repo, revision, filename, destination, workers=1)
        assert result["sha256"] == good_sha
        assert (destination / filename).read_bytes() == b"good"
    else:
        with pytest.raises(RuntimeError, match="Weight integrity check failed"):
            module.segmented_weights(repo, revision, filename, destination, workers=1)
        assert not list(cache.glob("*.part"))
        assert not (destination / "model.assembling").exists()
    assert len(requests_made) == expected_requests


def test_external_dataset_is_rejected_before_io(monkeypatch, tmp_path, capsys):
    module = load_script("inspect_dataset")
    monkeypatch.setattr(module, "ROOT", tmp_path / "project")
    monkeypatch.setattr(sys, "argv", ["inspect_dataset.py", "--dataset", str(tmp_path / "external")])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert "project-relative" in capsys.readouterr().err
