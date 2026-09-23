# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from chai_lab.data.io.prepared_outputs import (
    PreparedOutputError,
    expected_seed_samples,
    publish_structure_candidates,
    seed_outputs_complete,
)


def _publish_seed_in_process(root_text: str, seed: int) -> None:
    import chai_lab.data.io.prepared_outputs as outputs

    root = Path(root_text)
    native = root / f"native-{seed}"
    native.mkdir()
    cif = native / "pred.model_idx_0.cif"
    cif.write_text(f"data_seed_{seed}\n", encoding="utf-8")
    plot = native / "msa_depth.pdf"
    plot.write_bytes(f"%PDF-{seed}".encode())
    candidates = SimpleNamespace(
        cif_paths=[cif],
        ranking_data=[None],
        msa_coverage_plot_path=plot,
        pae=np.full((1, 2, 2), seed, dtype=np.float32),
        pde=np.full((1, 2, 2), seed, dtype=np.float32),
        plddt=np.full((1, 2), seed, dtype=np.float32),
    )

    def fake_scores(_):
        return {
            "aggregate_score": np.array([seed / 100]),
            "ptm": np.array([0.5]),
            "iptm": np.array([0.5]),
        }

    outputs.get_scores = fake_scores
    outputs.publish_structure_candidates(
        candidates,
        predictions_dir=root,
        seed=seed,
    compress_full_confidence=True)


class PreparedOutputTest(unittest.TestCase):
    def test_resume_checks_only_file_metadata_for_every_required_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for sample in (0, 1):
                prefix = f"seed-7_sample-{sample}"
                paths.extend([
                    root / "models" / f"{prefix}_model.cif",
                    root / "summary_confidences" / f"{prefix}_summary_confidences.json",
                    *(root / "full_data" / f"{kind}_{prefix}.npz"
                      for kind in ("pae", "pde", "plddt")),
                ])
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"not a valid CIF, JSON or NPZ")
            with patch.object(Path, "open", side_effect=AssertionError("no content reads")):
                self.assertTrue(seed_outputs_complete(root, seed=7, sample_count=2, compress_full_confidence=True))
            for path in paths:
                with self.subTest(path=path.name):
                    path.write_bytes(b"")
                    self.assertFalse(seed_outputs_complete(root, seed=7, sample_count=2, compress_full_confidence=True))
                    path.unlink()
                    self.assertFalse(seed_outputs_complete(root, seed=7, sample_count=2, compress_full_confidence=True))
                    path.mkdir()
                    self.assertFalse(seed_outputs_complete(root, seed=7, sample_count=2, compress_full_confidence=True))
                    path.rmdir()
                    path.write_bytes(b"nonempty")

    def test_npz_is_deflated_and_lossless(self):
        from zipfile import ZIP_DEFLATED, ZipFile

        from chai_lab.data.io.prepared_outputs import _atomic_npz

        with tempfile.TemporaryDirectory() as temporary:
            for dtype in (np.float16, np.float32, np.float64):
                value = np.arange(9, dtype=dtype).reshape(3, 3)
                path = Path(temporary) / "pae.npz"
                _atomic_npz(path, "pae", value)
                with ZipFile(path) as archive:
                    self.assertEqual(archive.namelist(), ["pae.npy"])
                    self.assertTrue(
                        all(
                            item.compress_type == ZIP_DEFLATED
                            for item in archive.infolist()
                        )
                    )
                with np.load(path, allow_pickle=False) as archive:
                    self.assertEqual(archive["pae"].dtype, value.dtype)
                    np.testing.assert_array_equal(archive["pae"], value)

    def test_native_candidates_are_published_per_seed_and_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = root / "native"
            native.mkdir()
            cif_paths = []
            for sample in range(2):
                path = native / f"pred.model_idx_{sample}.cif"
                path.write_text(f"data_sample_{sample}\n", encoding="utf-8")
                cif_paths.append(path)
            msa_plot = native / "msa_depth.pdf"
            msa_plot.write_bytes(b"%PDF-test")
            candidates = SimpleNamespace(
                cif_paths=cif_paths,
                ranking_data=[object(), object()],
                msa_coverage_plot_path=msa_plot,
                pae=torch.arange(18).reshape(2, 3, 3),
                pde=torch.arange(18, 36).reshape(2, 3, 3),
                plddt=torch.arange(6).reshape(2, 3),
            )
            scores = {
                "aggregate_score": np.array([0.75]),
                "ptm": np.array([0.70]),
                "iptm": np.array([0.80]),
                "per_chain_ptm": np.array([[0.6, 0.7]]),
                "per_chain_pair_iptm": np.array([[[0.0, 0.8], [0.8, 0.0]]]),
                "has_inter_chain_clashes": np.array([False]),
                "chain_chain_clashes": np.zeros((1, 2, 2), dtype=bool),
            }

            with patch(
                "chai_lab.data.io.prepared_outputs.get_scores", return_value=scores
            ):
                published = publish_structure_candidates(
                    candidates, predictions_dir=root / "predictions", seed=42
                , compress_full_confidence=True)

            self.assertEqual(len(published), 2)
            self.assertEqual(published[1].model_path.name, "seed-42_sample-1_model.cif")
            self.assertEqual(
                published[0].summary_path.name,
                "seed-42_sample-0_summary_confidences.json",
            )
            summary = json.loads(published[0].summary_path.read_text())
            self.assertEqual(summary["seed"], 42)
            self.assertEqual(summary["sample"], 0)
            self.assertEqual(summary["aggregate_score"], 0.75)
            self.assertEqual(summary["per_chain_ptm"], [0.6, 0.7])
            for path, key in (
                (published[0].pae_path, "pae"),
                (published[0].pde_path, "pde"),
                (published[0].plddt_path, "plddt"),
            ):
                with np.load(path, allow_pickle=False) as archive:
                    self.assertEqual(archive.files, [key])
            self.assertFalse((root / "predictions" / "msa_depth.pdf").exists())
            self.assertEqual(list((root / "predictions").rglob("*.tmp")), [])
            self.assertTrue(
                seed_outputs_complete(root / "predictions", seed=42, sample_count=2, compress_full_confidence=True)
            )
            published[1].pde_path.write_bytes(b"")
            self.assertFalse(
                seed_outputs_complete(root / "predictions", seed=42, sample_count=2, compress_full_confidence=True)
            )

    def test_expected_paths_require_the_exact_sample_set(self):
        paths = expected_seed_samples("predictions", seed=9, sample_count=2, compress_full_confidence=True)
        self.assertEqual(paths[0].model_path.name, "seed-9_sample-0_model.cif")
        self.assertEqual(paths[1].pae_path.name, "pae_seed-9_sample-1.npz")
        with self.assertRaisesRegex(PreparedOutputError, "positive"):
            expected_seed_samples("predictions", seed=9, sample_count=0, compress_full_confidence=True)

    def test_legacy_paths_do_not_complete_new_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary) / "job"
            old = expected_seed_samples(
                job / "predictions", seed=7, sample_count=1
            , compress_full_confidence=True)[0]
            paths = (
                old.model_path,
                old.summary_path,
                old.pae_path,
                old.pde_path,
                old.plddt_path,
            )
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"legacy")

            self.assertFalse(seed_outputs_complete(job, seed=7, sample_count=1, compress_full_confidence=True))
            self.assertTrue(all(path.read_bytes() == b"legacy" for path in paths))

    def test_inconsistent_native_candidates_are_rejected(self):
        candidates = SimpleNamespace(
            cif_paths=[],
            ranking_data=[],
            pae=torch.empty((1, 0, 0)),
            pde=torch.empty((0, 0, 0)),
            plddt=torch.empty((0, 0)),
        )
        with self.assertRaisesRegex(PreparedOutputError, "inconsistent"):
            publish_structure_candidates(candidates, predictions_dir="unused", seed=1, compress_full_confidence=True)

    def test_independent_processes_publish_different_seeds_to_one_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = multiprocessing.get_context("spawn")
            processes = [
                context.Process(
                    target=_publish_seed_in_process,
                    args=(str(root), seed),
                )
                for seed in range(10, 16)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)

            for seed in range(10, 16):
                self.assertTrue(
                    seed_outputs_complete(
                        root, seed=seed, sample_count=1
                    , compress_full_confidence=True)
                )
            self.assertEqual(list(root.rglob("*.tmp")), [])
            self.assertFalse((root / "msa_depth.pdf").exists())


if __name__ == "__main__":
    unittest.main()
