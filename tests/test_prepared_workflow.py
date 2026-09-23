# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import inspect
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import torch

from chai_lab.data.io.prepared_input import PreparedInput
from chai_lab.data.io.prepared_outputs import expected_seed_samples
from chai_lab.workflow import (
    build_workflow_plan,
    make_prepared_feature_context,
    normalize_seeds,
    run_prepared_workflow,
)


def _minimal_manifest() -> dict:
    return {
        "version": 1,
        "name": "seq",
        "sequences": [
            {
                "protein": {
                    "id": ["G"],
                    "sequence": "GGGG",
                    "pairedMsa": "",
                    "unpairedMsa": "",
                    "templates": [],
                }
            },
            {
                "protein": {
                    "id": ["H"],
                    "sequence": "HHHH",
                    "pairedMsa": "",
                    "unpairedMsa": "",
                    "templates": [],
                }
            },
        ],
    }


class SeedNormalizationTest(unittest.TestCase):
    def test_one_or_more_ordered_seeds(self):
        self.assertEqual(normalize_seeds(7), (7,))
        self.assertEqual(normalize_seeds("7, 9,11"), (7, 9, 11))
        self.assertEqual(normalize_seeds([9, 7]), (9, 7))

    def test_invalid_seeds_are_rejected(self):
        for value in ("1,,2", "x", "1,1", -1, 2**32, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_seeds(value)

    def test_missing_seed_is_materialized(self):
        seed = normalize_seeds(None)
        self.assertEqual(len(seed), 1)
        self.assertLess(seed[0], 2**32)

    def test_native_python_api_no_longer_accepts_num_trunk_samples(self):
        from chai_lab.chai1 import run_inference

        self.assertNotIn(
            "num_trunk_samples", inspect.signature(run_inference).parameters
        )


class PreparedFeatureContextTest(unittest.TestCase):
    def test_json_entities_and_native_restraint_parser_reach_feature_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            constraint = root / "constraints.csv"
            constraint.write_text(
                "chainA,res_idxA,chainB,res_idxB,connection_type,confidence,"
                "min_distance_angstrom,max_distance_angstrom,comment,restraint_id\n"
                "G,G1,H,H1,contact,1.0,0.0,8.0,test,contact_1\n",
                encoding="utf-8",
            )
            prepared = PreparedInput.from_dict(_minimal_manifest())
            msa_directory = root / "msas"
            msa_directory.mkdir()

            context = make_prepared_feature_context(
                prepared,
                msa_directory=msa_directory,
                esm_device=torch.device("cpu"),
                use_esm_embeddings=False,
                constraint_path=constraint,
                fasta_names_as_cif_chains=True,
            )

            self.assertEqual(
                [chain.entity_data.entity_name for chain in context.chains], ["G", "H"]
            )
            self.assertFalse(context.msa_context.mask.any())
            self.assertIsNotNone(context.restraint_context.contact_restraints)
            self.assertEqual(len(context.restraint_context.contact_restraints), 1)


class PreparedWorkflowExecutionTest(unittest.TestCase):
    def test_outputs_use_job_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.json"
            source.write_text(json.dumps(_minimal_manifest()), encoding="utf-8")

            plan = build_workflow_plan(
                source,
                root / "results",
                run_data_pipeline=False,
                run_inference=True,
            )

            job = (root / "results/seq").resolve()
            self.assertEqual(plan.predictions_dir, job)
            sample = expected_seed_samples(job, seed=7, sample_count=1)[0]
            self.assertEqual(
                sample.model_path,
                job / "models/seed-7_sample-0_model.cif",
            )
            self.assertFalse(job.exists())

    def test_feature_context_is_built_once_and_reused_for_all_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = root / "seq_data.json"
            request.write_text(json.dumps(_minimal_manifest()), encoding="utf-8")
            private_msa = root / "private_msa"
            feature_context = object()
            native_candidates = object()

            def fake_build_private(prepared, path):
                path.mkdir()
                return path

            def fake_publish(candidates, predictions_dir, seed, **kwargs):
                return (
                    SimpleNamespace(
                        model_path=Path(predictions_dir)
                        / "models"
                        / f"seed-{seed}_sample-0_model.cif"
                    ),
                )

            with (
                patch(
                    "chai_lab.data.io.prepared_msas.build_private_msa_directory",
                    side_effect=fake_build_private,
                ) as build_msa,
                patch(
                    "chai_lab.workflow.make_prepared_feature_context",
                    return_value=feature_context,
                ) as make_context,
                patch(
                    "chai_lab.workflow._run_folding",
                    return_value=native_candidates,
                ) as fold,
                patch(
                    "chai_lab.data.io.prepared_outputs.publish_structure_candidates",
                    side_effect=fake_publish,
                ) as publish,
            ):
                result = run_prepared_workflow(
                    request,
                    root / "result",
                    run_data_pipeline=False,
                    run_inference=True,
                    seeds="4,5",
                    num_diffn_samples=1,
                    device="cpu",
                    fasta_names_as_cif_chains=True,
                )

            self.assertEqual(result.seeds, (4, 5))
            self.assertEqual(len(result.prediction_paths), 2)
            self.assertEqual(build_msa.call_count, 1)
            self.assertEqual(make_context.call_count, 1)
            self.assertTrue(
                make_context.call_args.kwargs["fasta_names_as_cif_chains"]
            )
            self.assertEqual(fold.call_count, 2)
            self.assertTrue(
                fold.call_args_list[0].kwargs[
                    "entity_names_as_chain_names_in_output_cif"
                ]
            )
            self.assertEqual(
                [item.kwargs["seed"] for item in fold.call_args_list], [4, 5]
            )
            self.assertEqual(
                publish.call_args_list,
                [
                    call(
                        native_candidates,
                        predictions_dir=(root / "result/seq").resolve(),
                        seed=4,
                        compress_full_confidence=False,
                    ),
                    call(
                        native_candidates,
                        predictions_dir=(root / "result/seq").resolve(),
                        seed=5,
                        compress_full_confidence=False,
                    ),
                ],
            )
            self.assertNotEqual(private_msa, build_msa.call_args.args[1])

    def test_separate_stages_survive_manifest_rename_and_cwd_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input" / "request.json"
            source.parent.mkdir()
            manifest = _minimal_manifest()
            manifest["sequences"][0]["protein"].update(
                {
                    "pairedMsa": ">q\nGGGG\n",
                    "unpairedMsa": ">q\nGGGG\n",
                }
            )
            source.write_text(json.dumps(manifest), encoding="utf-8")
            source_bytes = source.read_bytes()
            data_root = root / "prepared"

            prepared_result = run_prepared_workflow(
                source,
                data_root,
                run_data_pipeline=True,
                run_inference=False,
                use_msa_server=False,
            )
            renamed = prepared_result.prepared_path.with_name("renamed.json")
            prepared_result.prepared_path.rename(renamed)
            other_cwd = root / "elsewhere"
            other_cwd.mkdir()

            def fake_publish(candidates, predictions_dir, seed, **kwargs):
                return (
                    SimpleNamespace(
                        model_path=Path(predictions_dir)
                        / "models"
                        / f"seed-{seed}_sample-0_model.cif"
                    ),
                )

            previous_cwd = Path.cwd()
            try:
                os.chdir(other_cwd)
                with (
                    patch(
                        "chai_lab.workflow.make_prepared_feature_context",
                        return_value=object(),
                    ),
                    patch("chai_lab.workflow._run_folding", return_value=object()),
                    patch(
                        "chai_lab.data.io.prepared_outputs.publish_structure_candidates",
                        side_effect=fake_publish,
                    ),
                ):
                    inferred = run_prepared_workflow(
                        renamed,
                        root / "predictions",
                        run_data_pipeline=False,
                        write_input_json=False,
                        run_inference=True,
                        seeds=7,
                        num_diffn_samples=1,
                        device="cpu",
                        use_esm_embeddings=False,
                    )
            finally:
                os.chdir(previous_cwd)

            self.assertEqual(inferred.prepared_path, renamed.resolve())
            self.assertEqual(
                inferred.prediction_paths,
                (
                    (root / "predictions/seq/models/seed-7_sample-0_model.cif").resolve(),
                ),
            )
            self.assertEqual(source.read_bytes(), source_bytes)

    def test_skip_avoids_feature_and_model_work_for_complete_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = root / "seq_data.json"
            request.write_text(json.dumps(_minimal_manifest()), encoding="utf-8")
            predictions = root / "result/seq"
            expected = expected_seed_samples(predictions, seed=4, sample_count=2)
            for sample in expected:
                for path in (
                    sample.model_path,
                    sample.summary_path,
                    sample.pae_path,
                    sample.pde_path,
                    sample.plddt_path,
                ):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"complete")

            with patch(
                "chai_lab.workflow.make_prepared_feature_context"
            ) as make_context:
                result = run_prepared_workflow(
                    request,
                    root / "result",
                    run_data_pipeline=False,
                    run_inference=True,
                    seeds=4,
                    num_diffn_samples=2,
                    device="cpu",
                    skip=True,
                )

            make_context.assert_not_called()
            self.assertEqual(result.seeds, (4,))
            self.assertEqual(
                result.prediction_paths,
                tuple(sample.model_path for sample in expected),
            )

    def test_skip_runs_only_incomplete_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = root / "seq_data.json"
            request.write_text(json.dumps(_minimal_manifest()), encoding="utf-8")
            predictions = root / "result/seq"
            complete_paths = []
            for seed in (4, 5):
                for sample in (0, 1):
                    prefix = f"seed-{seed}_sample-{sample}"
                    paths = [
                        predictions / "models" / f"{prefix}_model.cif",
                        predictions / "summary_confidences" / f"{prefix}_summary_confidences.json",
                        *(predictions / "full_data" / f"{kind}_{prefix}.json"
                          for kind in ("pae", "pde", "plddt")),
                    ]
                    for path in paths:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b"old condition result")
                    if seed == 4:
                        complete_paths.extend(paths)
            (predictions / "full_data/pde_seed-5_sample-1.json").write_bytes(b"")
            changed = _minimal_manifest()
            changed["sequences"][0]["protein"]["sequence"] = "AAAA"
            request.write_text(json.dumps(changed), encoding="utf-8")

            def fake_build_private(prepared, path):
                path.mkdir()
                return path

            def fake_publish(candidates, predictions_dir, seed, **kwargs):
                return tuple(
                    SimpleNamespace(
                        model_path=Path(predictions_dir)
                        / "models"
                        / f"seed-{seed}_sample-{sample}_model.cif"
                    )
                    for sample in range(2)
                )

            with (
                patch(
                    "chai_lab.data.io.prepared_msas.build_private_msa_directory",
                    side_effect=fake_build_private,
                ),
                patch(
                    "chai_lab.workflow.make_prepared_feature_context",
                    return_value=object(),
                ) as make_context,
                patch("chai_lab.workflow._run_folding", return_value=object()) as fold,
                patch(
                    "chai_lab.data.io.prepared_outputs.publish_structure_candidates",
                    side_effect=fake_publish,
                ),
            ):
                result = run_prepared_workflow(
                    request,
                    root / "result",
                    run_data_pipeline=False,
                    run_inference=True,
                    seeds="4,5",
                    num_diffn_samples=2,
                    num_trunk_recycles=9,
                    device="cpu",
                    skip=True,
                )

            self.assertEqual(result.seeds, (4, 5))
            make_context.assert_called_once()
            fold.assert_called_once()
            self.assertEqual(fold.call_args.kwargs["seed"], 5)
            self.assertEqual(fold.call_args.kwargs["num_diffn_samples"], 2)
            self.assertTrue(all(path.read_bytes() == b"old condition result" for path in complete_paths))


if __name__ == "__main__":
    unittest.main()
