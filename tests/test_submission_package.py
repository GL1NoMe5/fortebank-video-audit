"""Synthetic tests for normative registry integration and submission package builder."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.build_submission_package import (
    ALLOWLISTED_SUMMARY_FIELDS,
    _digest,
    build_submission_package,
    main,
    resolve_private_output_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class NormativeRegistryIntegrationTests(unittest.TestCase):
    """Verify normative_rules_rk.json, normative_rule_proposals.json, and docs."""

    def test_camera_capable_device_rule_integrated_and_existing_preserved(self) -> None:
        rules_path = REPO_ROOT / "config" / "normative_rules_rk.json"
        proposals_path = REPO_ROOT / "config" / "normative_rule_proposals.json"
        docs_path = REPO_ROOT / "docs" / "normative_sources.md"

        rules_data = json.loads(rules_path.read_text(encoding="utf-8"))
        proposals_data = json.loads(proposals_path.read_text(encoding="utf-8"))
        docs_text = docs_path.read_text(encoding="utf-8")

        rules_by_id = {rule["id"]: rule for rule in rules_data["rules"]}
        expected_ids = {
            "cash_recount_on_receipt",
            "cash_recount_on_delivery",
            "cash_separation_by_transaction",
            "cashier_leaves_during_service",
            "collection_bag_seal_check",
            "damaged_collection_bag_not_recounted",
            "camera_capable_device_in_cash_area",
            "valuables_unattended_accessible",
            "secure_storage_left_open",
            "unauthorized_person_access",
            "mobile_phone_use",
            "equipment_authenticity_or_abs_compliance",
        }
        self.assertEqual(set(rules_by_id.keys()), expected_ids)

        cam_rule = rules_by_id["camera_capable_device_in_cash_area"]
        self.assertTrue(cam_rule["supported"])
        self.assertFalse(cam_rule["requires_complete_action_sequence"])
        self.assertEqual(len(cam_rule["visual_conditions"]), 4)
        self.assertEqual(len(cam_rule["insufficient_evidence"]), 6)
        self.assertEqual(len(cam_rule["exceptions"]), 5)
        self.assertEqual(
            cam_rule["source"]["amendment_url"],
            "https://adilet.zan.kz/rus/docs/V2500037429",
        )
        self.assertEqual(
            cam_rule["source"]["context_source"]["url"],
            "https://adilet.zan.kz/rus/docs/V2000020075",
        )

        mobile_rule = rules_by_id["mobile_phone_use"]
        self.assertFalse(mobile_rule["supported"])

        self.assertEqual(proposals_data["status"], "integrated_into_main_registry")
        proposal_item = proposals_data["proposals"][0]
        self.assertEqual(proposal_item["id"], "camera_capable_device_in_cash_area")
        self.assertTrue(proposal_item["supported"])
        self.assertTrue(proposal_item["active"])
        self.assertEqual(proposal_item["visual_conditions"], cam_rule["visual_conditions"])

        self.assertIn("camera_capable_device_in_cash_area", docs_text)
        self.assertIn("V2500037429", docs_text)
        self.assertIn("mobile_phone_use", docs_text)


class SubmissionPackageBuilderTests(unittest.TestCase):
    """Verify security, path confinement, real schemas, and honesty of build_submission_package.py."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "runtime" / "secrets").mkdir(parents=True, exist_ok=True)
        (self.root / "runtime" / "private_submission").mkdir(parents=True, exist_ok=True)
        self.ref_time = datetime.fromisoformat("2026-10-17T17:02:00+05:00")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_valid_credentials_and_endpoint(
        self,
        secret: str = "sk_live_9876543210abcdef",
        **endpoint_overrides: object,
    ) -> None:
        (self.root / "runtime" / "secrets" / "api_key.txt").write_text(
            secret + "\n",
            encoding="utf-8",
        )
        payload: dict[str, object] = {
            "schema_version": 1,
            "url": "https://cash-audit-tunnel.trycloudflare.com",
            "status": "connected",
            "updated_at": "2026-10-17T17:00:00+05:00",
            "instance": "supervisor-win10-rtx3080",
            "generation": 4,
            "api_ready": True,
            "tunnel_connected": True,
            "temporary": True,
        }
        payload.update(endpoint_overrides)
        (self.root / "runtime" / "public_endpoint.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def _write_complete_model_provenance(
        self,
        model_dir: Path,
        repo_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        revision: str = "b7e4c9a1234567890abcdef1234567890abcdef1",
        license_id: str = "apache-2.0",
    ) -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        prov_path = model_dir / "download_provenance.json"
        prov_path.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "repo_id": repo_id,
                    "revision": revision,
                    "license": license_id,
                    "weight_integrity": [
                        {
                            "filename": "model-00001-of-00002.safetensors",
                            "bytes": 2147483648,
                            "sha256": "a" * 64,
                        },
                        {
                            "filename": "model-00002-of-00002.safetensors",
                            "bytes": 1825361100,
                            "sha256": "b" * 64,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return prov_path

    def _make_canonical_eval_payload(
        self,
        *,
        split: str = "train",
        model: str = "Qwen3-VL-4B-Instruct",
        model_path_str: str = "D:/CodexProjects/Datathon/models/Qwen3-VL-4B-Instruct",
        summary: dict[str, object] | None = None,
    ) -> dict[str, object]:
        identity = {
            "split": split,
            "config": {
                "model_path": model_path_str,
                "quantization": "nf4",
                "scan_fps": 2.0,
            },
            "model": model,
            "grounding_model": "IDEA-Research/grounding-dino-tiny",
            "rules_sha256": "c" * 64,
            "code": {"git_head": "1234567890abcdef"},
            "manifest_sha256": "d" * 64,
            "examples_sha256": "e" * 64,
            "environment": {"python": "3.10.11", "torch": "2.10.0+cu128"},
        }
        identity_sha256 = _digest(identity)
        return {
            "split": split,
            "model": model,
            "provenance": {
                "schema_version": 1,
                "identity": identity,
                "identity_sha256": identity_sha256,
                "selected_ids": ["SECRET_SELECTED_ID_001"],
                "selected_records": [{"video_id": "SECRET_SELECTED_RECORD_001"}],
            },
            "records": [
                {
                    "video": "PRIVATE_CLIENT_VIDEO_DO_NOT_LEAK.mov",
                    "notes": "RAW_RECORD_SECRET_TEXT_999",
                }
            ],
            "private_report": "FORBIDDEN_RAW_REPORT_CONTENT",
            "summary": summary or {},
        }

    def test_output_path_confinement_rejects_outside_paths(self) -> None:
        default_out = resolve_private_output_path(self.root)
        self.assertEqual(
            default_out,
            (self.root / "runtime" / "private_submission" / "SUBMISSION_PACKAGE.md").resolve(),
        )
        custom_valid = resolve_private_output_path(
            self.root,
            "runtime/private_submission/custom_jury_pack.md",
        )
        self.assertTrue(
            str(custom_valid).startswith(
                str((self.root / "runtime" / "private_submission").resolve())
            )
        )

        for forbidden in (
            "docs/SUBMISSION_PACKAGE.md",
            "runtime/SUBMISSION_PACKAGE.md",
            "runtime/private_submission/../leaked.md",
            "../outside.md",
            "runtime/private_submission",
        ):
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ValueError):
                    resolve_private_output_path(self.root, forbidden)

    def test_endpoint_flag_combinations_and_timestamp_freshness_validation(self) -> None:
        result = build_submission_package(repo_root=self.root, reference_time=self.ref_time)
        self.assertEqual(result["status"], "not_ready")
        self.assertEqual(result["local_metadata_status"], "LOCAL_METADATA_NOT_READY")
        self.assertEqual(result["external_reachability_status"], "EXTERNAL_REACHABILITY_NOT_TESTED")
        content = result["output_path"].read_text(encoding="utf-8")
        self.assertIn("LOCAL_METADATA_NOT_READY", content)
        self.assertIn("EXTERNAL_REACHABILITY_NOT_TESTED", content)
        self.assertIn("missing runtime/secrets/api_key.txt", content)
        self.assertIn("missing runtime/public_endpoint.json", content)
        self.assertIn("target accuracy has NOT been demonstrated", content)

        invalid_cases: list[tuple[str, dict[str, object], str]] = [
            ("degraded_status", {"status": "degraded"}, "endpoint status is not ready (degraded"),
            ("reconnecting_status", {"status": "reconnecting"}, "endpoint status is not ready (reconnecting"),
            ("api_ready_false", {"api_ready": False}, "api_ready flag must be strictly true"),
            ("api_ready_int", {"api_ready": 1}, "api_ready flag must be strictly true"),
            ("tunnel_connected_false", {"tunnel_connected": False}, "tunnel_connected flag must be strictly true"),
            ("temporary_false", {"temporary": False}, "temporary flag must be strictly true"),
            ("bad_schema_version", {"schema_version": 2}, "missing or invalid schema_version"),
            ("missing_instance", {"instance": ""}, "missing or invalid instance identifier"),
            ("bool_generation", {"generation": True}, "missing or invalid generation integer"),
            ("stale_timestamp", {"updated_at": "2026-10-17T16:50:00+05:00"}, "endpoint metadata is stale"),
            ("future_timestamp", {"updated_at": "2026-10-17T17:05:00+05:00"}, "updated_at timestamp is in the future"),
            ("naive_timestamp", {"updated_at": "2026-10-17T17:00:00"}, "invalid or timezone-naive updated_at"),
        ]

        for label, overrides, expected_msg in invalid_cases:
            with self.subTest(case=label):
                self._write_valid_credentials_and_endpoint(**overrides)
                res = build_submission_package(
                    repo_root=self.root,
                    reference_time=self.ref_time,
                    max_endpoint_age_sec=300.0,
                )
                self.assertEqual(res["status"], "not_ready")
                self.assertEqual(res["local_metadata_status"], "LOCAL_METADATA_NOT_READY")
                pkg_text = res["output_path"].read_text(encoding="utf-8")
                self.assertIn(expected_msg, pkg_text)

    def test_model_provenance_default_and_forte_model_path_override_and_malformed_rejection(self) -> None:
        secret_value = "super_secret_jury_api_key_2026_rk"
        self._write_valid_credentials_and_endpoint(secret=secret_value)

        default_model_dir = self.root / "models" / "Qwen3-VL-4B-Instruct"
        self._write_complete_model_provenance(
            default_model_dir,
            repo_id="Qwen/Qwen3-VL-4B-Instruct",
            revision="rev_4b_default_12345",
            license_id="apache-2.0",
        )
        dino_dir = self.root / "models" / "grounding-dino-tiny"
        self._write_complete_model_provenance(
            dino_dir,
            repo_id="IDEA-Research/grounding-dino-tiny",
            revision="rev_dino_67890",
            license_id="apache-2.0",
        )

        stdout_buf = io.StringIO()
        with patch("sys.stdout", stdout_buf):
            exit_code = main(
                [
                    "--repo-root",
                    str(self.root),
                    "--reference-time",
                    "2026-10-17T17:02:00+05:00",
                    "--require-ready",
                ]
            )

        self.assertEqual(exit_code, 0)
        stdout_text = stdout_buf.getvalue().strip()
        self.assertIn("status=ready", stdout_text)
        self.assertIn("local_metadata=LOCAL_METADATA_READY", stdout_text)
        self.assertIn("external_reachability=EXTERNAL_REACHABILITY_NOT_TESTED", stdout_text)
        self.assertNotIn(secret_value, stdout_text)
        self.assertNotIn("trycloudflare.com", stdout_text)

        pkg_path = self.root / "runtime" / "private_submission" / "SUBMISSION_PACKAGE.md"
        pkg_text = pkg_path.read_text(encoding="utf-8")
        self.assertIn(secret_value, pkg_text)
        self.assertIn("https://cash-audit-tunnel.trycloudflare.com", pkg_text)
        self.assertIn("Qwen/Qwen3-VL-4B-Instruct", pkg_text)
        self.assertIn("rev_4b_default_12345", pkg_text)
        self.assertIn("apache-2.0", pkg_text)
        self.assertIn("model-00001-of-00002.safetensors", pkg_text)
        self.assertIn("complete (metadata-only; weight files not re-hashed)", pkg_text)
        self.assertIn("weight files on disk were NOT cryptographically re-hashed", pkg_text)
        self.assertIn("IDEA-Research/grounding-dino-tiny", pkg_text)
        self.assertIn("rev_dino_67890", pkg_text)
        self.assertIn("Qwen/Qwen3-VL-8B-Instruct", pkg_text)
        self.assertIn("GPLv3", pkg_text)
        self.assertIn("HF_HUB_OFFLINE = 1", pkg_text)
        self.assertIn("local_files_only=True", pkg_text)
        self.assertIn("Приём наличных без наблюдаемого пересчёта", pkg_text)
        self.assertIn("Средство связи с фото- или видеофункцией у работника в кассовом узле", pkg_text)
        self.assertIn("18 октября 2026", pkg_text)

        candidate_8b_dir = self.root / "models" / "Qwen3-VL-8B-Instruct"
        self._write_complete_model_provenance(
            candidate_8b_dir,
            repo_id="Qwen/Qwen3-VL-8B-Instruct",
            revision="rev_8b_override_99999",
            license_id="apache-2.0",
        )
        res_8b = build_submission_package(
            repo_root=self.root,
            reference_time=self.ref_time,
            env={"FORTE_MODEL_PATH": str(candidate_8b_dir)},
        )
        self.assertTrue(res_8b["model_provenance_complete"])
        self.assertEqual(res_8b["active_model_dir"], candidate_8b_dir.resolve())
        pkg_8b_text = res_8b["output_path"].read_text(encoding="utf-8")
        self.assertIn("rev_8b_override_99999", pkg_8b_text)

        malformed_dir = self.root / "models" / "malformed-model"
        malformed_dir.mkdir(parents=True, exist_ok=True)
        (malformed_dir / "download_provenance.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
                    "revision": "rev_bad",
                    "license": "apache-2.0",
                    "weight_integrity": [{"filename": "bad.safetensors", "bytes": -5, "sha256": "not_hex"}],
                }
            ),
            encoding="utf-8",
        )
        res_malformed = build_submission_package(
            repo_root=self.root,
            model_provenance_path=malformed_dir,
            reference_time=self.ref_time,
        )
        self.assertFalse(res_malformed["model_provenance_complete"])
        self.assertEqual(res_malformed["model_provenance_status"], "unverified")

    def test_evaluation_canonical_digest_verification_and_record_filtering(self) -> None:
        self._write_valid_credentials_and_endpoint()

        train_eval_path = self.root / "eval_train_valid.json"
        bad_hash_eval_path = self.root / "eval_holdout_bad_hash.json"
        mismatch_model_eval_path = self.root / "eval_holdout_mismatch_model.json"

        train_summary: dict[str, object] = {
            "videos": 34,
            "completed": 32,
            "failed": 2,
            "tp": 14,
            "fp": 3,
            "tn": 12,
            "fn": 3,
            "precision": 0.8235,
            "recall": 0.8235,
            "accuracy": 0.7647,
            "completed_accuracy": 0.8125,
            "failed_positive": 1,
            "mean_seconds": 94.25,
            "max_seconds": 240.5,
            "median_localization_error_sec": 1.8,
            "localized_positive_count": 14,
            "localized_within_3_sec": 11,
            "over_300_seconds": 0,
            "UnlistedSecretMetric": 999999,
        }
        valid_payload = self._make_canonical_eval_payload(
            split="train",
            model="Qwen3-VL-4B-Instruct",
            summary=train_summary,
        )
        train_eval_path.write_text(json.dumps(valid_payload), encoding="utf-8")

        bad_hash_payload = self._make_canonical_eval_payload(
            split="holdout",
            model="Qwen3-VL-4B-Instruct",
            summary={
                "videos": 8,
                "completed": 6,
                "failed": 2,
                "failed_positive": 1,
                "accuracy": 0.5,
                "completed_accuracy": 0.6667,
            },
        )
        bad_hash_payload["provenance"]["identity_sha256"] = "f" * 64  # type: ignore[index]
        bad_hash_eval_path.write_text(json.dumps(bad_hash_payload), encoding="utf-8")

        mismatch_payload = self._make_canonical_eval_payload(
            split="holdout",
            model="Qwen3-VL-8B-Instruct",
            model_path_str="D:/CodexProjects/Datathon/models/Qwen3-VL-4B-Instruct",
            summary={"videos": 8, "completed": 8, "failed": 0},
        )
        mismatch_model_eval_path.write_text(json.dumps(mismatch_payload), encoding="utf-8")

        result = build_submission_package(
            repo_root=self.root,
            eval_paths=[train_eval_path, bad_hash_eval_path, mismatch_model_eval_path],
            reference_time=self.ref_time,
        )
        pkg_text = result["output_path"].read_text(encoding="utf-8")

        self.assertNotIn("PRIVATE_CLIENT_VIDEO_DO_NOT_LEAK.mov", pkg_text)
        self.assertNotIn("RAW_RECORD_SECRET_TEXT_999", pkg_text)
        self.assertNotIn("FORBIDDEN_RAW_REPORT_CONTENT", pkg_text)
        self.assertNotIn("SECRET_SELECTED_ID_001", pkg_text)
        self.assertNotIn("SECRET_SELECTED_RECORD_001", pkg_text)
        self.assertNotIn("UnlistedSecretMetric", pkg_text)
        self.assertNotIn("999999", pkg_text)

        self.assertIn("Выборка (Split): `train`", pkg_text)
        self.assertIn("Выборка (Split): `holdout`", pkg_text)
        self.assertIn("identity_hash_consistent (metadata-only)", pkg_text)
        self.assertIn("canonical digest mismatch", pkg_text)
        self.assertIn("does not match Path(identity.config.model_path).name", pkg_text)
        self.assertIn("| `failed` | `2` |", pkg_text)
        self.assertIn("| `failed_positive` | `1` |", pkg_text)

    def test_evaluation_numeric_bounds_reject_nan_inf_bool_negative_fractional_and_out_of_range(self) -> None:
        self._write_valid_credentials_and_endpoint()
        bounds_eval_path = self.root / "eval_bounds.json"
        payload = self._make_canonical_eval_payload(
            split="train",
            model="Qwen3-VL-4B-Instruct",
            summary={
                "videos": 10.5,
                "completed": -1,
                "failed": True,
                "tp": 4,
                "fp": 0,
                "tn": 4,
                "fn": 2,
                "precision": 1.25,
                "recall": -0.1,
                "accuracy": float("nan"),
                "completed_accuracy": 0.8,
                "failed_positive": 1.7,
                "mean_seconds": float("inf"),
                "max_seconds": -12.0,
                "median_localization_error_sec": 2.25,
                "localized_positive_count": 4,
                "localized_within_3_sec": 3,
                "over_300_seconds": 0,
            },
        )
        bounds_eval_path.write_text(
            json.dumps(payload, allow_nan=True),
            encoding="utf-8",
        )

        result = build_submission_package(
            repo_root=self.root,
            eval_paths=[bounds_eval_path],
            reference_time=self.ref_time,
        )
        pkg_text = result["output_path"].read_text(encoding="utf-8")

        for invalid_field in (
            "videos",
            "completed",
            "failed",
            "precision",
            "recall",
            "accuracy",
            "failed_positive",
            "mean_seconds",
            "max_seconds",
        ):
            with self.subTest(field=invalid_field):
                self.assertIn(f"| `{invalid_field}` | `not measured` |", pkg_text)

        self.assertIn("| `tp` | `4` |", pkg_text)
        self.assertIn("| `completed_accuracy` | `0.8000` |", pkg_text)
        self.assertIn("| `median_localization_error_sec` | `2.25` |", pkg_text)


class SupervisorEndpointHeartbeatTests(unittest.TestCase):
    """Verify Supervisor.state() refreshes connected public_endpoint.json without masking degraded/reconnecting/shutdown."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "runtime" / "secrets").mkdir(parents=True, exist_ok=True)
        (self.root / "runtime" / "private_submission").mkdir(parents=True, exist_ok=True)
        (self.root / "runtime" / "secrets" / "api_key.txt").write_text(
            "sk_live_9876543210abcdef\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _make_stub_supervisor(self):
        import threading
        from types import SimpleNamespace
        from scripts.serve import Supervisor

        sup = Supervisor.__new__(Supervisor)
        sup.root = self.root
        sup.runtime = self.root / "runtime"
        sup.instance = "sup-instance-001"
        sup.me = {"pid": 1234}
        sup.stop_event = threading.Event()
        sup.awake = SimpleNamespace(active=True)
        sup.api_ready = True
        sup.api_http_responsive = True
        sup.api_unresponsive_since = None
        sup.url = "https://stable-tunnel.trycloudflare.com"
        sup.tunnel_connected = True
        sup.endpoint_generation = 2
        sup.state_status = "running"
        sup.children = {
            "api": {"identity": {"pid": 11}, "process": None, "failures": 0},
            "cloudflared": {"identity": {"pid": 22}, "process": None, "failures": 0},
        }
        return sup

    def test_stable_connection_over_300s_stays_ready_with_heartbeat_and_stale_when_stopped(self) -> None:
        sup = self._make_stub_supervisor()
        with patch("scripts.serve.timestamp", return_value="2026-10-17T17:00:00+05:00"):
            sup.endpoint("connected")

        with patch("scripts.serve.timestamp", return_value="2026-10-17T17:10:00+05:00"):
            sup.state()

        res_live = build_submission_package(
            repo_root=self.root,
            reference_time=datetime.fromisoformat("2026-10-17T17:10:03+05:00"),
            max_endpoint_age_sec=300.0,
        )
        self.assertTrue(res_live["ready"])
        self.assertEqual(res_live["local_metadata_status"], "LOCAL_METADATA_READY")
        self.assertEqual(res_live["external_reachability_status"], "EXTERNAL_REACHABILITY_NOT_TESTED")

        res_stale = build_submission_package(
            repo_root=self.root,
            reference_time=datetime.fromisoformat("2026-10-17T17:16:00+05:00"),
            max_endpoint_age_sec=300.0,
        )
        self.assertFalse(res_stale["ready"])
        self.assertEqual(res_stale["local_metadata_status"], "LOCAL_METADATA_NOT_READY")

    def test_heartbeat_preserves_degraded_reconnecting_and_shutdown_states(self) -> None:
        sup = self._make_stub_supervisor()
        ep_path = self.root / "runtime" / "public_endpoint.json"

        for non_ready_status in ("degraded", "reconnecting", "stopped"):
            with self.subTest(status=non_ready_status):
                with patch("scripts.serve.timestamp", return_value="2026-10-17T17:00:00+05:00"):
                    sup.endpoint(non_ready_status)
                with patch("scripts.serve.timestamp", return_value="2026-10-17T17:05:00+05:00"):
                    sup.state()
                ep_data = json.loads(ep_path.read_text(encoding="utf-8"))
                self.assertEqual(ep_data["status"], non_ready_status)
                self.assertEqual(ep_data["updated_at"], "2026-10-17T17:00:00+05:00")

        with patch("scripts.serve.timestamp", return_value="2026-10-17T17:00:00+05:00"):
            sup.endpoint("connected")
        sup.stop_event.set()
        with patch("scripts.serve.timestamp", return_value="2026-10-17T17:05:00+05:00"):
            sup.state()
        ep_stopped = json.loads(ep_path.read_text(encoding="utf-8"))
        self.assertEqual(ep_stopped["updated_at"], "2026-10-17T17:00:00+05:00")


if __name__ == "__main__":
    unittest.main()
