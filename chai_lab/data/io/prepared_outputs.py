# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

"""Publish native Chai candidates in the EnsembleFold seed/sample layout."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from chai_lab.ranking.rank import get_scores

if TYPE_CHECKING:
    from chai_lab.chai1 import StructureCandidates


class PreparedOutputError(ValueError):
    """Raised when native candidates cannot satisfy the wrapper output contract."""


@dataclass(frozen=True)
class PublishedSample:
    """Paths published for one seed/sample pair."""

    seed: int
    sample: int
    model_path: Path
    summary_path: Path
    pae_path: Path
    pde_path: Path
    plddt_path: Path


def expected_seed_samples(
    predictions_dir: str | Path,
    *,
    seed: int,
    sample_count: int,
    compress_full_confidence: bool = False,
) -> tuple[PublishedSample, ...]:
    """Return the exact wrapper paths expected for one seed."""
    if sample_count <= 0:
        raise PreparedOutputError("sample_count must be positive")
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    extension = "npz" if compress_full_confidence else "json"
    return tuple(
        PublishedSample(
            seed=seed,
            sample=sample,
            model_path=(
                predictions_dir / "models" / f"seed-{seed}_sample-{sample}_model.cif"
            ),
            summary_path=(
                predictions_dir
                / "summary_confidences"
                / f"seed-{seed}_sample-{sample}_summary_confidences.json"
            ),
            pae_path=(
                predictions_dir / "full_data" / f"pae_seed-{seed}_sample-{sample}.{extension}"
            ),
            pde_path=(
                predictions_dir / "full_data" / f"pde_seed-{seed}_sample-{sample}.{extension}"
            ),
            plddt_path=(
                predictions_dir / "full_data" / f"plddt_seed-{seed}_sample-{sample}.{extension}"
            ),
        )
        for sample in range(sample_count)
    )


def seed_outputs_complete(
    predictions_dir: str | Path,
    *,
    seed: int,
    sample_count: int,
    compress_full_confidence: bool = False,
) -> bool:
    """Lightly check that every expected file for a seed exists and is non-empty."""
    return all(
        path.is_file() and path.stat().st_size > 0
        for sample in expected_seed_samples(
            predictions_dir, seed=seed, sample_count=sample_count,
            compress_full_confidence=compress_full_confidence
        )
        for path in (
            sample.model_path,
            sample.summary_path,
            sample.pae_path,
            sample.pde_path,
            sample.plddt_path,
        )
    )


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _atomic_copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _atomic_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_npz(path: Path, key: str, value: Any) -> Path:
    if path.suffix == ".json":
        return _atomic_json(path, {key: np.asarray(value).tolist()})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **{key: np.asarray(value)})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _json_score(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.size == 1:
        return array.reshape(-1)[0].item()
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return array.tolist()


def publish_structure_candidates(
    candidates: StructureCandidates,
    *,
    predictions_dir: str | Path,
    seed: int,
    compress_full_confidence: bool = False,
) -> tuple[PublishedSample, ...]:
    """Atomically publish one seed's native candidates without ranking/reordering."""
    predictions_dir = Path(predictions_dir).expanduser().resolve()
    sample_count = len(candidates.cif_paths)
    if not (
        sample_count
        == len(candidates.ranking_data)
        == candidates.pae.shape[0]
        == candidates.pde.shape[0]
        == candidates.plddt.shape[0]
    ):
        raise PreparedOutputError("Native candidate arrays have inconsistent sizes")

    expected = expected_seed_samples(
        predictions_dir, seed=seed, sample_count=sample_count,
            compress_full_confidence=compress_full_confidence
    )
    published: list[PublishedSample] = []
    for sample, paths in enumerate(expected):
        model_path = _atomic_copy(candidates.cif_paths[sample], paths.model_path)

        native_scores = get_scores(candidates.ranking_data[sample])
        summary = {
            "seed": seed,
            "sample": sample,
            **{key: _json_score(value) for key, value in native_scores.items()},
        }
        summary_path = _atomic_json(paths.summary_path, summary)
        pae_path = _atomic_npz(paths.pae_path, "pae", candidates.pae[sample])
        pde_path = _atomic_npz(paths.pde_path, "pde", candidates.pde[sample])
        plddt_path = _atomic_npz(paths.plddt_path, "plddt", candidates.plddt[sample])
        for path in (pae_path, pde_path, plddt_path):
            path.with_suffix(".json" if compress_full_confidence else ".npz").unlink(missing_ok=True)
        published.append(
            PublishedSample(
                seed=seed,
                sample=sample,
                model_path=model_path,
                summary_path=summary_path,
                pae_path=pae_path,
                pde_path=pde_path,
                plddt_path=plddt_path,
            )
        )

    # The native MSA coverage PDF is diagnostic only. Do not publish it into the
    # shared prediction tree: every seed produces the same target-level plot, so
    # multi-process seed jobs would repeatedly replace one common file.
    # if candidates.msa_coverage_plot_path is not None:
    #     _atomic_copy(
    #         candidates.msa_coverage_plot_path, predictions_dir / "msa_depth.pdf"
    #     )
    return tuple(published)
