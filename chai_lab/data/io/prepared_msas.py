# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

"""Persist and reconstruct MSAs for the EnsembleFold prepared-input workflow."""

import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import pandas as pd

from chai_lab.data.io.compression import read_text_auto, write_zstd_text
from chai_lab.data.io.prepared_input import (
    PreparedEntity,
    PreparedInput,
    PreparedTemplates,
    load_prepared_input,
    write_prepared_input,
)
from chai_lab.data.parsing.msas.prepared_a3m import (
    PreparedMSAError,
    colabfold_a3ms_to_dataframe,
    protein_msa_query_sequence,
)
from chai_lab.data.parsing.msas.sequence_hash import expected_basename


class A3MResult(Protocol):
    paired: str
    unpaired: str


MSASearcher = Callable[
    [list[str], Path, str, bool], tuple[Sequence[A3MResult], Path | None]
]


def _relative_path(resource: Path, manifest_path: Path) -> Path:
    return Path(os.path.relpath(resource, manifest_path.parent))


def _canonical_text(text: str) -> str:
    return text if not text or text.endswith("\n") else f"{text}\n"


def _rebase_templates(
    templates: PreparedTemplates | None,
    source_manifest: Path,
    output_manifest: Path,
) -> PreparedTemplates | None:
    if templates is None:
        return None
    resolved = templates.resolved(source_manifest)
    return replace(
        resolved,
        hits_path=_relative_path(resolved.hits_path, output_manifest),
        cif_directory=_relative_path(resolved.cif_directory, output_manifest),
    )


def prepare_msa_bundle(
    input_path: str | Path,
    output_manifest_path: str | Path,
    *,
    use_msa_server: bool,
    msa_server_url: str = "https://api.colabfold.com",
    searcher: MSASearcher | None = None,
) -> PreparedInput:
    """Write canonical A3M.zst files and the resulting ``*_data.json``.

    Declared MSA paths win. If server search is enabled, it fills only missing
    paired/unpaired inputs while still searching the complete protein assembly.
    """
    source_manifest = Path(input_path).expanduser().resolve()
    output_manifest = Path(output_manifest_path).expanduser().resolve()
    prepared = load_prepared_input(source_manifest)
    resolved = prepared.resolved(source_manifest)

    protein_entities = [
        entity for entity in resolved.sequences if entity.kind == "protein"
    ]
    query_sequences = [
        protein_msa_query_sequence(entity.sequence) for entity in protein_entities
    ]
    needs_search = use_msa_server and any(
        entity.paired_msa is None or entity.unpaired_msa is None
        for entity in protein_entities
    )

    searched_by_entity: list[A3MResult | None] = [None] * len(protein_entities)
    if needs_search:
        if searcher is None:
            from chai_lab.data.dataset.msas.colabfold import generate_colabfold_a3ms

            searcher = generate_colabfold_a3ms
        expanded_queries = [
            query
            for query, entity in zip(query_sequences, protein_entities, strict=True)
            for _ in entity.ids
        ]
        with tempfile.TemporaryDirectory(prefix="chai_msa_search_") as temporary:
            search_results, _ = searcher(
                expanded_queries,
                Path(temporary),
                msa_server_url,
                False,
            )
        if len(search_results) != len(expanded_queries):
            raise PreparedMSAError(
                "ColabFold search returned a different number of per-chain MSAs"
            )
        cursor = 0
        for index, entity in enumerate(protein_entities):
            searched_by_entity[index] = search_results[cursor]
            cursor += len(entity.ids)

    msa_dir = output_manifest.parent / "msas"
    rewritten_proteins: dict[int, PreparedEntity] = {}
    for index, (entity, query, searched) in enumerate(
        zip(protein_entities, query_sequences, searched_by_entity, strict=True)
    ):
        paired_text = (
            read_text_auto(entity.paired_msa)
            if entity.paired_msa is not None
            else (None if searched is None else searched.paired)
        )
        unpaired_text = (
            read_text_auto(entity.unpaired_msa)
            if entity.unpaired_msa is not None
            else (None if searched is None else searched.unpaired)
        )
        paired_text = (
            None if paired_text is None or not paired_text.strip() else paired_text
        )
        unpaired_text = (
            None
            if unpaired_text is None or not unpaired_text.strip()
            else unpaired_text
        )

        if paired_text is not None or unpaired_text is not None:
            colabfold_a3ms_to_dataframe(
                query_sequence=query,
                paired_a3m=paired_text,
                unpaired_a3m=unpaired_text,
                unpaired_fallback_source=entity.unpaired_msa_fallback_source,
            )

        first_id = entity.ids[0]
        paired_path = None
        if paired_text is not None:
            paired_absolute = write_zstd_text(
                msa_dir / f"{prepared.name}_{first_id}_paired.a3m.zst",
                _canonical_text(paired_text),
            )
            paired_path = _relative_path(paired_absolute, output_manifest)

        unpaired_path = None
        if unpaired_text is not None:
            unpaired_absolute = write_zstd_text(
                msa_dir / f"{prepared.name}_{first_id}_unpaired.a3m.zst",
                _canonical_text(unpaired_text),
            )
            unpaired_path = _relative_path(unpaired_absolute, output_manifest)

        rewritten_proteins[index] = replace(
            entity,
            paired_msa=paired_path,
            unpaired_msa=unpaired_path,
        )

    protein_index = 0
    rewritten_entities: list[PreparedEntity] = []
    for original_entity in prepared.sequences:
        if original_entity.kind == "protein":
            rewritten_entities.append(rewritten_proteins[protein_index])
            protein_index += 1
        else:
            rewritten_entities.append(original_entity)

    resolved_constraint = (
        None
        if resolved.constraint_path is None
        else _relative_path(resolved.constraint_path, output_manifest)
    )
    bundled = replace(
        prepared,
        sequences=tuple(rewritten_entities),
        templates=_rebase_templates(
            prepared.templates, source_manifest, output_manifest
        ),
        constraint_path=resolved_constraint,
    )
    write_prepared_input(bundled, output_manifest)
    return bundled


def build_private_msa_directory(
    prepared: PreparedInput,
    msa_directory: str | Path,
) -> Path:
    """Reconstruct hash-named ``.aligned.pqt`` files in a private directory."""
    msa_directory = Path(msa_directory).expanduser().resolve()
    msa_directory.mkdir(parents=True, exist_ok=False)
    for entity in prepared.sequences:
        if entity.kind != "protein":
            continue
        if entity.paired_msa is None and entity.unpaired_msa is None:
            continue
        for path in (entity.paired_msa, entity.unpaired_msa):
            if path is not None and not path.is_absolute():
                raise PreparedMSAError(
                    "MSA paths must be resolved before private reconstruction; "
                    "call validate_resources(manifest_path) first"
                )
        query = protein_msa_query_sequence(entity.sequence)
        dataframe = colabfold_a3ms_to_dataframe(
            query_sequence=query,
            paired_a3m=(
                None if entity.paired_msa is None else read_text_auto(entity.paired_msa)
            ),
            unpaired_a3m=(
                None
                if entity.unpaired_msa is None
                else read_text_auto(entity.unpaired_msa)
            ),
            unpaired_fallback_source=entity.unpaired_msa_fallback_source,
        )
        parquet_path = msa_directory / expected_basename(query)
        if parquet_path.exists():
            if not pd.read_parquet(parquet_path).equals(dataframe):
                raise PreparedMSAError(
                    "Multiple protein entities normalize to the same Chai MSA query "
                    "but declare different paired/unpaired alignments"
                )
            continue
        dataframe.to_parquet(parquet_path)
    return msa_directory
