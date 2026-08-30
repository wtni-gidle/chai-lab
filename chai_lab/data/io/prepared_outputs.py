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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **{key: np.asarray(value)})
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

    models_dir = predictions_dir / "models"
    summaries_dir = predictions_dir / "summary_confidences"
    full_data_dir = predictions_dir / "full_data"
    published: list[PublishedSample] = []
    for sample in range(sample_count):
        stem = f"seed-{seed}_sample-{sample}"
        model_path = _atomic_copy(
            candidates.cif_paths[sample], models_dir / f"{stem}_model.cif"
        )

        native_scores = get_scores(candidates.ranking_data[sample])
        summary = {
            "seed": seed,
            "sample": sample,
            **{key: _json_score(value) for key, value in native_scores.items()},
        }
        summary_path = _atomic_json(
            summaries_dir / f"{stem}_summary_confidences.json", summary
        )
        pae_path = _atomic_npz(
            full_data_dir / f"pae_{stem}.npz", "pae", candidates.pae[sample]
        )
        pde_path = _atomic_npz(
            full_data_dir / f"pde_{stem}.npz", "pde", candidates.pde[sample]
        )
        plddt_path = _atomic_npz(
            full_data_dir / f"plddt_{stem}.npz", "plddt", candidates.plddt[sample]
        )
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

    if candidates.msa_coverage_plot_path is not None:
        _atomic_copy(
            candidates.msa_coverage_plot_path, predictions_dir / "msa_depth.pdf"
        )
    return tuple(published)
