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
                    "pairedMsaPath": "msas/seq__A_pairedmsa.a3m.zst",
                    "unpairedMsaPath": "msas/seq__A_unpairedmsa.a3m.zst",
                    "unpairedMsaFallbackSource": "auto",
                    "templates": [],
                }
            },
            {"rna": {"id": ["R"], "sequence": "AUGC"}},
            {"dna": {"id": ["D"], "sequence": "ATGC"}},
            {"ligand": {"id": ["L"], "smiles": "CC(=O)O"}},
            {"glycan": {"id": ["G"], "sequence": "MAN(6-1 MAN)"}},
        ],
    }


class PreparedInputTest(unittest.TestCase):
    def test_runtime_options_are_rejected_from_json(self):
        manifest = _manifest_dict()
        protein = manifest["sequences"][0]["protein"]
        protein["pairedMsa"] = ""
        protein.pop("pairedMsaPath")
        protein["unpairedMsa"] = ">query\nAAAA\n"
        protein.pop("unpairedMsaPath")
        prepared = PreparedInput.from_dict(manifest)
        serialized = prepared.to_dict()
        self.assertNotIn("use_esm_embeddings", serialized)
        self.assertNotIn("constraint_path", serialized)

        for runtime_field, value in (
            ("use_esm_embeddings", False),
            ("constraint_path", "constraints.csv"),
        ):
            configured = dict(manifest)
            configured[runtime_field] = value
            with self.subTest(runtime_field=runtime_field), self.assertRaisesRegex(
                PreparedInputError, f"unknown fields: {runtime_field}"
            ):
                PreparedInput.from_dict(configured)

    def test_round_trip_and_relative_path_resolution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            job_dir = Path(temporary_directory) / "seq"
            (job_dir / "msas").mkdir(parents=True)
            for path in (
                job_dir / "msas" / "seq__A_pairedmsa.a3m.zst",
                job_dir / "msas" / "seq__A_unpairedmsa.a3m.zst",
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
                resolved.sequences[0].unpaired_msa_path,
                (job_dir / "msas" / "seq__A_unpairedmsa.a3m.zst").resolve(),
            )

    def test_initial_json_can_have_no_external_resources(self):
        manifest = _manifest_dict()
        protein = manifest["sequences"][0]["protein"]
        del protein["pairedMsaPath"]
        del protein["unpairedMsaPath"]
        del protein["unpairedMsaFallbackSource"]
        protein["templates"] = None

        prepared = PreparedInput.from_dict(manifest)
        self.assertIsNone(prepared.sequences[0].paired_msa)
        self.assertIsNone(prepared.sequences[0].paired_msa_path)
        self.assertIsNone(prepared.sequences[0].unpaired_msa)
        self.assertIsNone(prepared.sequences[0].unpaired_msa_path)
        self.assertEqual(prepared.sequences[0].unpaired_msa_fallback_source, "auto")

    def test_inline_msa_and_path_are_alternatives(self):
        manifest = _manifest_dict()
        protein = manifest["sequences"][0]["protein"]
        del protein["pairedMsaPath"]
        del protein["unpairedMsaPath"]
        protein["pairedMsa"] = ""
        protein["unpairedMsa"] = ">query\nAAAA\n"
        prepared = PreparedInput.from_dict(manifest)
        self.assertEqual(prepared.sequences[0].paired_msa, "")
        self.assertEqual(prepared.sequences[0].unpaired_msa, ">query\nAAAA\n")

        protein["unpairedMsaPath"] = "msa.a3m"
        with self.assertRaisesRegex(PreparedInputError, "only one of"):
            PreparedInput.from_dict(manifest)

    def test_af3_style_template_schema_and_mapping_validation(self):
        manifest = _manifest_dict()
        protein = manifest["sequences"][0]["protein"]
        protein["templates"] = [
            {
                "mmcif": "data_template\n#\n",
                "queryIndices": [0, 2, 3],
                "templateIndices": [1, 2, 4],
            }
        ]
        prepared = PreparedInput.from_dict(manifest)
        template = prepared.sequences[0].templates[0]
        self.assertEqual(template.query_indices, (0, 2, 3))
        self.assertEqual(template.template_indices, (1, 2, 4))

        protein["templates"][0]["mmcifPath"] = "template.cif"
        with self.assertRaisesRegex(PreparedInputError, "exactly one"):
            PreparedInput.from_dict(manifest)

        protein["templates"][0].pop("mmcifPath")
        protein["templates"][0]["templateIndices"] = [1, 2]
        with self.assertRaisesRegex(PreparedInputError, "equal length"):
            PreparedInput.from_dict(manifest)

        protein["templates"][0]["templateIndices"] = [1, 1, 4]
        with self.assertRaisesRegex(PreparedInputError, "strictly increasing"):
            PreparedInput.from_dict(manifest)

    def test_top_level_m8_template_block_is_rejected(self):
        manifest = _manifest_dict()
        manifest["templates"] = {
            "hits_path": "hits.m8",
            "cif_directory": "cifs",
            "query_id_mode": "entity_name",
        }
        with self.assertRaisesRegex(PreparedInputError, "unknown fields: templates"):
            PreparedInput.from_dict(manifest)

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

        manifest = _manifest_dict()
        manifest["entity_ids_as_cif_chains"] = True
        with self.assertRaisesRegex(
            PreparedInputError, "unknown fields: entity_ids_as_cif_chains"
        ):
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
                }
            }
        )
        with self.assertRaisesRegex(PreparedInputError, "merge their id lists"):
            PreparedInput.from_dict(manifest)

    def test_schema_rejects_unknown_fallback_source(self):
        manifest = _manifest_dict()
        protein = manifest["sequences"][0]["protein"]
        protein["unpairedMsaFallbackSource"] = "mystery"
        with self.assertRaisesRegex(PreparedInputError, "supported Chai MSA source"):
            PreparedInput.from_dict(manifest)

    def test_missing_declared_resource_is_an_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest_path = Path(temporary_directory) / "seq_data.json"
            manifest_path.write_text(json.dumps(_manifest_dict()), encoding="utf-8")
            prepared = load_prepared_input(manifest_path)
            with self.assertRaisesRegex(
                PreparedInputError, "pairedMsaPath does not exist"
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
            protein.pop("pairedMsaPath")
            protein.pop("unpairedMsaPath")
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
            protein.pop("pairedMsaPath")
            protein.pop("unpairedMsaPath")
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
