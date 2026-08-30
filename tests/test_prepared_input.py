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
from chai_lab.data.parsing.structure.entity_type import EntityType
from chai_lab.workflow import build_workflow_plan


def _manifest_dict() -> dict:
    return {
        "version": 1,
        "name": "seq",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "AAAA",
                    "paired_msa": "msas/seq_A_paired.a3m.zst",
                    "unpaired_msa": "msas/seq_A_unpaired.a3m.zst",
                    "unpaired_msa_fallback_source": "auto",
                }
            },
            {"rna": {"id": ["R"], "sequence": "AUGC"}},
            {"dna": {"id": ["D"], "sequence": "ATGC"}},
            {"ligand": {"id": ["L"], "smiles": "CC(=O)O"}},
            {"glycan": {"id": ["G"], "sequence": "MAN(6-1 MAN)"}},
        ],
        "use_esm_embeddings": True,
        "entity_ids_as_cif_chains": False,
        "templates": None,
        "constraint_path": "constraints/seq.restraints.csv",
    }


class PreparedInputTest(unittest.TestCase):
    def test_round_trip_and_relative_path_resolution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            job_dir = Path(temporary_directory) / "seq"
            (job_dir / "msas").mkdir(parents=True)
            (job_dir / "constraints").mkdir()
            for path in (
                job_dir / "msas" / "seq_A_paired.a3m.zst",
                job_dir / "msas" / "seq_A_unpaired.a3m.zst",
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
            self.assertEqual(
                resolved.sequences[0].unpaired_msa,
                (job_dir / "msas" / "seq_A_unpaired.a3m.zst").resolve(),
            )

    def test_initial_json_can_have_no_external_resources(self):
        manifest = _manifest_dict()
        protein = manifest["sequences"][0]["protein"]
        protein["paired_msa"] = None
        protein["unpaired_msa"] = None
        del protein["unpaired_msa_fallback_source"]
        manifest["constraint_path"] = None

        prepared = PreparedInput.from_dict(manifest)
        self.assertIsNone(prepared.sequences[0].paired_msa)
        self.assertIsNone(prepared.sequences[0].unpaired_msa)
        self.assertEqual(prepared.sequences[0].unpaired_msa_fallback_source, "auto")

    def test_json_entities_expand_to_native_chai_inputs(self):
        prepared = PreparedInput.from_dict(_manifest_dict())
        inputs = prepared.to_chai_inputs()

        self.assertEqual(
            [(item.entity_name, item.sequence, item.entity_type) for item in inputs],
            [
                ("A", "AAAA", EntityType.PROTEIN.value),
                ("B", "AAAA", EntityType.PROTEIN.value),
                ("R", "AUGC", EntityType.RNA.value),
                ("D", "ATGC", EntityType.DNA.value),
                ("L", "CC(=O)O", EntityType.LIGAND.value),
                ("G", "MAN(6-1 MAN)", EntityType.MANUAL_GLYCAN.value),
            ],
        )

    def test_schema_rejects_unknown_or_missing_fields(self):
        manifest = _manifest_dict()
        manifest["fasta_path"] = "seq.fasta"
        with self.assertRaisesRegex(PreparedInputError, "unknown fields: fasta_path"):
            PreparedInput.from_dict(manifest)

        manifest = _manifest_dict()
        del manifest["sequences"]
        with self.assertRaisesRegex(PreparedInputError, "missing fields: sequences"):
            PreparedInput.from_dict(manifest)

    def test_schema_rejects_unsafe_or_duplicate_entity_ids(self):
        manifest = _manifest_dict()
        manifest["name"] = "../seq"
        with self.assertRaisesRegex(PreparedInputError, "name must contain"):
            PreparedInput.from_dict(manifest)

        manifest = _manifest_dict()
        manifest["sequences"][0]["protein"]["id"] = ["A", "../B"]
        with self.assertRaisesRegex(PreparedInputError, "ASCII letters"):
            PreparedInput.from_dict(manifest)

        manifest = _manifest_dict()
        manifest["sequences"][1]["rna"]["id"] = ["A"]
        with self.assertRaisesRegex(PreparedInputError, "reuses entity IDs: A"):
            PreparedInput.from_dict(manifest)

    def test_duplicate_protein_sequences_must_share_one_entry(self):
        manifest = _manifest_dict()
        manifest["sequences"].append(
            {
                "protein": {
                    "id": ["C"],
                    "sequence": "aaaa",
                    "paired_msa": None,
                    "unpaired_msa": None,
                }
            }
        )
        with self.assertRaisesRegex(PreparedInputError, "merge their id lists"):
            PreparedInput.from_dict(manifest)

    def test_schema_rejects_unknown_fallback_source(self):
        manifest = _manifest_dict()
        protein = manifest["sequences"][0]["protein"]
        protein["unpaired_msa_fallback_source"] = "mystery"
        with self.assertRaisesRegex(PreparedInputError, "supported Chai MSA source"):
            PreparedInput.from_dict(manifest)

    def test_missing_declared_resource_is_an_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "seq_data.json"
            manifest_path.write_text(json.dumps(_manifest_dict()), encoding="utf-8")
            prepared = load_prepared_input(manifest_path)
            with self.assertRaisesRegex(
                PreparedInputError, "paired_msa does not exist"
            ):
                prepared.validate_resources(manifest_path)

    def test_prepared_output_path(self):
        path = prepared_input_path("output", "seq")
        self.assertEqual(path.name, "seq_data.json")
        self.assertEqual(path.parent.name, "seq")


class WorkflowPlanTest(unittest.TestCase):
    def test_data_and_combined_plans_use_json_name(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = _manifest_dict()
            protein = manifest["sequences"][0]["protein"]
            protein["paired_msa"] = None
            protein["unpaired_msa"] = None
            manifest["constraint_path"] = None
            input_path = root / "arbitrary_name.json"
            input_path.write_text(json.dumps(manifest), encoding="utf-8")

            for run_inference in (False, True):
                with self.subTest(run_inference=run_inference):
                    plan = build_workflow_plan(
                        input_path,
                        root / "result",
                        run_data_pipeline=True,
                        run_inference=run_inference,
                    )
                    self.assertEqual(plan.name, "seq")
                    self.assertEqual(plan.prepared_path.name, "seq_data.json")
                    self.assertEqual(plan.job_dir, (root / "result" / "seq").resolve())
                    self.assertIsNotNone(plan.prepared_input)

    def test_inference_plan_uses_json_name_not_filename(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = _manifest_dict()
            protein = manifest["sequences"][0]["protein"]
            protein["paired_msa"] = None
            protein["unpaired_msa"] = None
            manifest["constraint_path"] = None
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
            self.assertEqual(plan.prepared_input.sequences[0].ids, ("A", "B"))

    def test_stage_and_input_kind_validation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            text_path = root / "seq.fasta"
            text_path.write_text(">protein|name=A\nAAAA\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "At least one"):
                build_workflow_plan(
                    text_path,
                    root / "result",
                    run_data_pipeline=False,
                    run_inference=False,
                )
            with self.assertRaisesRegex(PreparedInputError, "self-contained JSON"):
                build_workflow_plan(
                    text_path,
                    root / "result",
                    run_data_pipeline=True,
                    run_inference=False,
                )


if __name__ == "__main__":
    unittest.main()
