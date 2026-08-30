# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import json
import tempfile
import unittest
from pathlib import Path

from chai_lab.data.io.prepared_input import (
    PreparedInput,
    PreparedInputError,
    load_prepared_input,
    prepared_input_path,
    write_prepared_input,
)
from chai_lab.workflow import build_workflow_plan


def _manifest_dict() -> dict:
    return {
        "version": 1,
        "name": "seq",
        "fasta_path": "seq.fasta",
        "use_esm_embeddings": True,
        "fasta_names_as_cif_chains": False,
        "msas": [
            {
                "chains": ["A", "B"],
                "sequence_hash": (
                    "63c1dd951ffedf6f7fd968ad4efa39b8ed584f162f46e715114ee184f8de9201"
                ),
                "paired_msa": "msas/seq_A_paired.a3m.zst",
                "unpaired_msa": "msas/seq_A_unpaired.a3m.zst",
                "unpaired_source_database": "auto",
            }
        ],
        "templates": {
            "hits_path": "templates/all_chain_templates.m8",
            "cif_directory": "templates/cifs",
            "query_id_mode": "sequence_hash",
        },
        "constraint_path": "constraints/seq.restraints.csv",
    }


class PreparedInputTest(unittest.TestCase):
    def test_round_trip_and_relative_path_resolution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            job_dir = Path(temporary_directory) / "seq"
            (job_dir / "msas").mkdir(parents=True)
            (job_dir / "templates" / "cifs").mkdir(parents=True)
            (job_dir / "constraints").mkdir()
            for path in (
                job_dir / "seq.fasta",
                job_dir / "msas" / "seq_A_paired.a3m.zst",
                job_dir / "msas" / "seq_A_unpaired.a3m.zst",
                job_dir / "templates" / "all_chain_templates.m8",
                job_dir / "constraints" / "seq.restraints.csv",
            ):
                path.write_text("test\n", encoding="utf-8")

            prepared = PreparedInput.from_dict(_manifest_dict())
            manifest_path = write_prepared_input(
                prepared, job_dir / "renamed_data.json"
            )
            loaded = load_prepared_input(manifest_path)
            resolved = loaded.validate_resources(manifest_path)

            self.assertEqual(loaded, prepared)
            self.assertEqual(resolved.name, "seq")
            self.assertEqual(resolved.fasta_path, (job_dir / "seq.fasta").resolve())
            self.assertEqual(
                resolved.msas[0].unpaired_msa,
                (job_dir / "msas" / "seq_A_unpaired.a3m.zst").resolve(),
            )
            self.assertTrue(resolved.templates.cif_directory.is_absolute())

    def test_empty_optional_sections_are_valid(self):
        manifest = _manifest_dict()
        manifest["msas"] = []
        manifest["templates"] = None
        manifest["constraint_path"] = None
        prepared = PreparedInput.from_dict(manifest)
        self.assertEqual(prepared.msas, ())
        self.assertIsNone(prepared.templates)
        self.assertIsNone(prepared.constraint_path)

    def test_schema_rejects_unknown_or_missing_fields(self):
        manifest = _manifest_dict()
        manifest["typo"] = True
        with self.assertRaisesRegex(PreparedInputError, "unknown fields: typo"):
            PreparedInput.from_dict(manifest)

        manifest = _manifest_dict()
        del manifest["fasta_path"]
        with self.assertRaisesRegex(PreparedInputError, "missing fields: fasta_path"):
            PreparedInput.from_dict(manifest)

    def test_schema_rejects_unsafe_name_and_duplicate_msa_mapping(self):
        manifest = _manifest_dict()
        manifest["name"] = "../seq"
        with self.assertRaisesRegex(PreparedInputError, "name must contain"):
            PreparedInput.from_dict(manifest)

        manifest = _manifest_dict()
        duplicate = dict(manifest["msas"][0])
        duplicate["chains"] = ["B", "C"]
        duplicate["sequence_hash"] = "f" * 64
        manifest["msas"].append(duplicate)
        with self.assertRaisesRegex(PreparedInputError, "reuses chains: B"):
            PreparedInput.from_dict(manifest)

    def test_schema_rejects_unknown_source_and_template_mode(self):
        manifest = _manifest_dict()
        manifest["msas"][0]["unpaired_source_database"] = "mystery"
        with self.assertRaisesRegex(PreparedInputError, "Chai MSA data source"):
            PreparedInput.from_dict(manifest)

        manifest = _manifest_dict()
        manifest["templates"]["query_id_mode"] = "guess"
        with self.assertRaisesRegex(PreparedInputError, "entity_name.*sequence_hash"):
            PreparedInput.from_dict(manifest)

    def test_schema_rejects_non_sha256_sequence_hash(self):
        manifest = _manifest_dict()
        manifest["msas"][0]["sequence_hash"] = "abc123"
        with self.assertRaisesRegex(PreparedInputError, "lowercase SHA-256"):
            PreparedInput.from_dict(manifest)

    def test_missing_declared_resource_is_an_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "seq_data.json"
            manifest_path.write_text(json.dumps(_manifest_dict()), encoding="utf-8")
            prepared = load_prepared_input(manifest_path)
            with self.assertRaisesRegex(
                PreparedInputError, "fasta_path does not exist"
            ):
                prepared.validate_resources(manifest_path)

    def test_prepared_output_path(self):
        path = prepared_input_path("output", "seq")
        self.assertEqual(path.name, "seq_data.json")
        self.assertEqual(path.parent.name, "seq")


class WorkflowPlanTest(unittest.TestCase):
    def test_data_plan_uses_fasta_stem(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fasta_path = root / "seq.fasta"
            fasta_path.write_text(">protein|name=A\nAAAA\n", encoding="utf-8")
            for run_inference in (False, True):
                with self.subTest(run_inference=run_inference):
                    plan = build_workflow_plan(
                        fasta_path,
                        root / "result",
                        run_data_pipeline=True,
                        run_inference=run_inference,
                    )
                    self.assertEqual(plan.name, "seq")
                    self.assertEqual(plan.prepared_path.name, "seq_data.json")
                    self.assertEqual(plan.job_dir, (root / "result" / "seq").resolve())
                    self.assertIsNone(plan.prepared_input)

    def test_inference_plan_uses_json_name_not_filename(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = _manifest_dict()
            manifest["msas"] = []
            manifest["templates"] = None
            manifest["constraint_path"] = None
            (root / "seq.fasta").write_text(">protein|name=A\nAAAA\n", encoding="utf-8")
            manifest_path = root / "anything.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            plan = build_workflow_plan(
                manifest_path,
                root / "result",
                run_data_pipeline=False,
                run_inference=True,
            )
            self.assertEqual(plan.name, "seq")
            self.assertEqual(plan.prepared_path, manifest_path.resolve())
            self.assertEqual(plan.job_dir, (root / "result" / "seq").resolve())
            self.assertTrue(plan.prepared_input.fasta_path.is_absolute())

    def test_stage_and_input_kind_validation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fasta_path = root / "seq.fasta"
            fasta_path.write_text(">protein|name=A\nAAAA\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "At least one"):
                build_workflow_plan(
                    fasta_path,
                    root / "result",
                    run_data_pipeline=False,
                    run_inference=False,
                )
            with self.assertRaisesRegex(PreparedInputError, "prepared.*json"):
                build_workflow_plan(
                    fasta_path,
                    root / "result",
                    run_data_pipeline=False,
                    run_inference=True,
                )


if __name__ == "__main__":
    unittest.main()
