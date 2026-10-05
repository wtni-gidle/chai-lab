"""Regression tests for the 2026-10-04 Chai wrapper review fixes."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from chai_lab.data.io.prepared_input import (
    PreparedEntity,
    PreparedInputError,
    PreparedTemplate,
)
from chai_lab.data.io.prepared_msas import prepare_data_bundle
from chai_lab.data.io.prepared_outputs import (
    publish_structure_candidates,
    seed_outputs_complete,
)
from chai_lab.data.io.prepared_templates import (
    _validate_resource_names_against_directory,
    materialize_template_structures,
)
from chai_lab.workflow import build_workflow_plan, run_prepared_workflow


def request_dict():
    return {"version": 1, "name": "demo", "sequences": [{"protein": {
        "id": ["A"], "sequence": "AAAA", "pairedMsa": "",
        "unpairedMsa": "", "templates": [],
    }}]}


def old_results(root, *, compressed=False):
    suffix = "npz" if compressed else "json"
    paths = [root / "models/seed-7_sample-0_model.cif",
             root / "summary_confidences/seed-7_sample-0_summary_confidences.json"]
    paths += [root / "full_data" / f"{kind}_seed-7_sample-0.{suffix}"
              for kind in ("pae", "pde", "plddt")]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"OLD RESULT")


class TemplateStageRegressionTest(unittest.TestCase):
    def test_inference_only_rejects_null_or_omitted_templates_for_both_write_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.json"
            for omitted in (False, True):
                for write in (False, True):
                    with self.subTest(omitted=omitted, write=write):
                        request = request_dict()
                        protein = request["sequences"][0]["protein"]
                        if omitted:
                            del protein["templates"]
                        else:
                            protein["templates"] = None
                        source.write_text(json.dumps(request))
                        with self.assertRaisesRegex(PreparedInputError, "template preparation"):
                            build_workflow_plan(source, root / "out",
                                run_data_pipeline=False, run_inference=True,
                                write_input_json=write)
                        self.assertFalse((root / "out").exists())

    def test_skip_cannot_bypass_unprepared_template_rejection_or_publish_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.json"
            request = request_dict()
            request["sequences"][0]["protein"]["templates"] = None
            source.write_text(json.dumps(request))
            old_results(root / "out/demo")
            for write in (False, True):
                with self.subTest(write=write):
                    with self.assertRaisesRegex(PreparedInputError, "template preparation"):
                        run_prepared_workflow(source, root / "out",
                            run_data_pipeline=False, run_inference=True,
                            write_input_json=write, skip=True, seeds=7,
                            num_diffn_samples=1)
                    self.assertFalse((root / "out/demo/demo_data.json").exists())

    def test_explicit_empty_templates_and_data_preparation_remain_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.json"
            for write in (False, True):
                request = request_dict()
                source.write_text(json.dumps(request))
                plan = build_workflow_plan(source, root / "out",
                    run_data_pipeline=False, run_inference=True, write_input_json=write)
                self.assertEqual(plan.prepared_input.sequences[0].templates, ())
                request["sequences"][0]["protein"]["templates"] = None
                source.write_text(json.dumps(request))
                plan = build_workflow_plan(source, root / "out",
                    run_data_pipeline=True, run_inference=True, write_input_json=write)
                self.assertIsNone(plan.prepared_input.sequences[0].templates)


class PublicationRegressionTest(unittest.TestCase):
    def test_interrupted_overwrite_is_not_complete_and_successful_retry_recovers(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                old_results(root, compressed=compressed)
                self.assertTrue(seed_outputs_complete(root, seed=7, sample_count=1,
                    compress_full_confidence=compressed))
                source = root / "native.cif"
                source.write_text("NEW MODEL")
                candidates = SimpleNamespace(cif_paths=[source], ranking_data=[None],
                    pae=np.ones((1, 2, 2)), pde=np.ones((1, 2, 2)),
                    plddt=np.ones((1, 2)), msa_coverage_plot_path=None)
                with patch("chai_lab.data.io.prepared_outputs.get_scores", return_value={}), \
                     patch("chai_lab.data.io.prepared_outputs._atomic_json", side_effect=OSError("disk failure")):
                    with self.assertRaisesRegex(OSError, "disk failure"):
                        publish_structure_candidates(candidates, predictions_dir=root,
                            seed=7, compress_full_confidence=compressed)
                self.assertEqual((root / "models/seed-7_sample-0_model.cif").read_text(), "NEW MODEL")
                self.assertFalse(seed_outputs_complete(root, seed=7, sample_count=1,
                    compress_full_confidence=compressed))
                with patch("chai_lab.data.io.prepared_outputs.get_scores", return_value={}):
                    publish_structure_candidates(candidates, predictions_dir=root,
                        seed=7, compress_full_confidence=compressed)
                self.assertTrue(seed_outputs_complete(root, seed=7, sample_count=1,
                    compress_full_confidence=compressed))
                summary = json.loads((root / "summary_confidences/seed-7_sample-0_summary_confidences.json").read_text())
                self.assertEqual(summary, {"seed": 7, "sample": 0})


class ResourceValidationRegressionTest(unittest.TestCase):
    def test_existing_directory_is_scanned_once_for_many_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "other.cif").write_text("old")
            original = Path.iterdir
            scans = []
            def counted(path):
                scans.append(path)
                return original(path)
            with patch.object(Path, "iterdir", counted):
                _validate_resource_names_against_directory(
                    [f"demo__A_template_{i}.cif" for i in range(20)], directory)
            self.assertEqual(len(scans), 1)

    def test_prepared_bundle_checks_resource_directory_once_for_all_entities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = request_dict()
            request["sequences"].append({"protein": {
                "id": ["B"], "sequence": "CCCC", "pairedMsa": "",
                "unpairedMsa": "", "templates": [],
            }})
            for entity in request["sequences"]:
                entity["protein"]["templates"] = [{"mmcif": "data_example\n#\n",
                    "queryIndices": [0], "templateIndices": [0]}]
            source = root / "input.json"
            source.write_text(json.dumps(request))
            output = root / "out/demo_data.json"
            directory = output.parent / "msas"
            directory.mkdir(parents=True)
            original = Path.iterdir
            scans = []
            def counted(path):
                if path.resolve() == directory.resolve():
                    scans.append(path)
                return original(path)
            with patch.object(Path, "iterdir", counted):
                prepare_data_bundle(source, output, use_msa_server=False)
            self.assertEqual(len(scans), 1)
            saved = json.loads(output.read_text())
            self.assertEqual(len(saved["sequences"]), 2)
            for entry in saved["sequences"]:
                path = output.parent / entry["protein"]["templates"][0]["mmcifPath"]
                self.assertEqual(path.read_text(), "data_example\n#\n")

    def test_standalone_template_writer_still_rejects_existing_case_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "msas").mkdir()
            (root / "msas/demo__a_template_0.cif").write_text("KEEP")
            with self.assertRaisesRegex(ValueError, "case-insensitive"):
                materialize_template_structures(
                    entity=PreparedEntity(kind="protein", ids=("A",), sequence="AAAA"),
                    templates=(PreparedTemplate(mmcif="data_example\n#\n", mmcif_path=None,
                        query_indices=(0,), template_indices=(0,)),),
                    target_name="demo", output_manifest=root / "demo_data.json")
            self.assertEqual((root / "msas/demo__a_template_0.cif").read_text(), "KEEP")


if __name__ == "__main__":
    unittest.main()
