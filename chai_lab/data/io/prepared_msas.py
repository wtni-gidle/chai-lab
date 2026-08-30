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
    PreparedTemplate,
    load_prepared_input,
    write_prepared_input,
)
from chai_lab.data.io.prepared_templates import (
    NativeTemplateParser,
    materialize_template_structures,
    parse_native_template_hits,
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


def _msa_is_supplied(content: str | None, path: Path | None) -> bool:
    return content is not None or path is not None


def _read_msa(content: str | None, path: Path | None) -> str | None:
    if path is not None:
        return read_text_auto(path)
    return content


def prepare_data_bundle(
    input_path: str | Path,
    output_manifest_path: str | Path,
    *,
    use_msa_server: bool,
    use_templates_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    searcher: MSASearcher | None = None,
    template_parser: NativeTemplateParser | None = None,
) -> PreparedInput:
    """Materialize MSA and parsed templates into a prepared bundle.

    Declared inline MSA content or MSA paths win. If server search is enabled, it
    fills only missing paired/unpaired inputs while still searching the complete
    protein assembly. Template M8 is an ephemeral native intermediate: accepted
    structures and Chai-derived residue mappings are persisted instead.
    """
    source_manifest = Path(input_path).expanduser().resolve()
    output_manifest = Path(output_manifest_path).expanduser().resolve()
    prepared = load_prepared_input(source_manifest)
    resolved = prepared.validate_resources(source_manifest)

    protein_entities = [
        entity for entity in resolved.sequences if entity.kind == "protein"
    ]
    query_sequences = [
        protein_msa_query_sequence(entity.sequence) for entity in protein_entities
    ]
    missing_msas = any(
        (
            not _msa_is_supplied(entity.paired_msa, entity.paired_msa_path)
            or not _msa_is_supplied(entity.unpaired_msa, entity.unpaired_msa_path)
        )
        for entity in protein_entities
    )
    needs_template_search = use_templates_server and any(
        entity.templates is None for entity in protein_entities
    )
    needs_search = (use_msa_server and missing_msas) or needs_template_search

    searched_by_entity: list[A3MResult | None] = [None] * len(protein_entities)
    expanded_queries = [
        query
        for query, entity in zip(query_sequences, protein_entities, strict=True)
        for _ in entity.ids
    ]
    server_query_ids: dict[str, int] = {}
    for query in expanded_queries:
        server_query_ids.setdefault(query, 101 + len(server_query_ids))
    searched_templates_by_entity: list[tuple[PreparedTemplate, ...] | None] = [
        None
    ] * len(protein_entities)
    if needs_search:
        if searcher is None:
            from chai_lab.data.dataset.msas.colabfold import generate_colabfold_a3ms

            searcher = generate_colabfold_a3ms
        with tempfile.TemporaryDirectory(prefix="chai_msa_search_") as temporary:
            search_results, template_hits_path = searcher(
                expanded_queries,
                Path(temporary),
                msa_server_url,
                needs_template_search,
            )
            if needs_template_search:
                if template_hits_path is None or not template_hits_path.is_file():
                    raise PreparedMSAError(
                        "Template server did not return a readable M8 file"
                    )
                if template_parser is None:
                    template_parser = parse_native_template_hits
            if len(search_results) != len(expanded_queries):
                raise PreparedMSAError(
                    "ColabFold search returned a different number of per-chain MSAs"
                )
            cursor = 0
            for index, (entity, query) in enumerate(
                zip(protein_entities, query_sequences, strict=True)
            ):
                if use_msa_server:
                    searched_by_entity[index] = search_results[cursor]
                if entity.templates is None and needs_template_search:
                    assert template_hits_path is not None
                    assert template_parser is not None
                    searched_templates_by_entity[index] = tuple(
                        template_parser(
                            str(server_query_ids[query]),
                            query,
                            template_hits_path,
                            Path(temporary) / "template_cifs",
                        )
                    )
                cursor += len(entity.ids)

    msa_dir = output_manifest.parent / "msas"
    rewritten_proteins: dict[int, PreparedEntity] = {}
    for index, (entity, query, searched) in enumerate(
        zip(protein_entities, query_sequences, searched_by_entity, strict=True)
    ):
        paired_text = _read_msa(entity.paired_msa, entity.paired_msa_path)
        if paired_text is None and searched is not None:
            paired_text = searched.paired
        unpaired_text = _read_msa(entity.unpaired_msa, entity.unpaired_msa_path)
        if unpaired_text is None and searched is not None:
            unpaired_text = searched.unpaired
        paired_text = (
            None
            if paired_text is None
            else (paired_text if paired_text.strip() else "")
        )
        unpaired_text = (
            None
            if unpaired_text is None
            else (unpaired_text if unpaired_text.strip() else "")
        )

        if paired_text or unpaired_text:
            colabfold_a3ms_to_dataframe(
                query_sequence=query,
                paired_a3m=paired_text,
                unpaired_a3m=unpaired_text,
                unpaired_fallback_source=entity.unpaired_msa_fallback_source,
            )

        first_id = entity.ids[0]
        paired_content = paired_text
        paired_path = None
        if paired_text:
            paired_absolute = write_zstd_text(
                msa_dir / f"{prepared.name}__{first_id}_pairedmsa.a3m.zst",
                _canonical_text(paired_text),
            )
            paired_path = _relative_path(paired_absolute, output_manifest)
            paired_content = None

        unpaired_content = unpaired_text
        unpaired_path = None
        if unpaired_text:
            unpaired_absolute = write_zstd_text(
                msa_dir / f"{prepared.name}__{first_id}_unpairedmsa.a3m.zst",
                _canonical_text(unpaired_text),
            )
            unpaired_path = _relative_path(unpaired_absolute, output_manifest)
            unpaired_content = None

        templates = entity.templates
        if templates is None:
            templates = searched_templates_by_entity[index] or ()
        templates = materialize_template_structures(
            entity=entity,
            templates=templates,
            target_name=prepared.name,
            output_manifest=output_manifest,
        )

        rewritten_proteins[index] = replace(
            entity,
            paired_msa=paired_content,
            paired_msa_path=paired_path,
            unpaired_msa=unpaired_content,
            unpaired_msa_path=unpaired_path,
            templates=templates,
        )

    protein_index = 0
    rewritten_entities: list[PreparedEntity] = []
    for original_entity in prepared.sequences:
        if original_entity.kind == "protein":
            rewritten_entities.append(rewritten_proteins[protein_index])
            protein_index += 1
        else:
            rewritten_entities.append(original_entity)

    bundled_constraint = (
        None
        if resolved.constraint_path is None
        else _relative_path(resolved.constraint_path, output_manifest)
    )
    bundled = replace(
        prepared,
        sequences=tuple(rewritten_entities),
        constraint_path=bundled_constraint,
    )
    write_prepared_input(bundled, output_manifest)
    return bundled


def prepare_msa_bundle(
    input_path: str | Path,
    output_manifest_path: str | Path,
    *,
    use_msa_server: bool,
    msa_server_url: str = "https://api.colabfold.com",
    searcher: MSASearcher | None = None,
) -> PreparedInput:
    """Compatibility API for the stage-2 MSA-only prepared workflow."""
    return prepare_data_bundle(
        input_path,
        output_manifest_path,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        searcher=searcher,
    )


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
        if not _msa_is_supplied(
            entity.paired_msa, entity.paired_msa_path
        ) and not _msa_is_supplied(entity.unpaired_msa, entity.unpaired_msa_path):
            continue
        for path in (entity.paired_msa_path, entity.unpaired_msa_path):
            if path is not None and not path.is_absolute():
                raise PreparedMSAError(
                    "MSA paths must be resolved before private reconstruction; "
                    "call validate_resources(manifest_path) first"
                )
        paired_text = _read_msa(entity.paired_msa, entity.paired_msa_path)
        unpaired_text = _read_msa(entity.unpaired_msa, entity.unpaired_msa_path)
        if not (paired_text and paired_text.strip()) and not (
            unpaired_text and unpaired_text.strip()
        ):
            continue
        query = protein_msa_query_sequence(entity.sequence)
        dataframe = colabfold_a3ms_to_dataframe(
            query_sequence=query,
            paired_a3m=paired_text,
            unpaired_a3m=unpaired_text,
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
