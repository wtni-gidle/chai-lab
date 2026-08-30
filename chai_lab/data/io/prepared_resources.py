# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

"""Template and restraint handoff for EnsembleFold prepared inputs."""

import csv
import gzip
import io
import os
import tempfile
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from chai_lab.data.io.compression import read_text_auto
from chai_lab.data.io.prepared_input import (
    PreparedInput,
    PreparedTemplates,
)
from chai_lab.data.parsing.msas.sequence_hash import hash_sequence

TemplateDownloader = Callable[[str, Path], Path]


class PreparedResourceError(ValueError):
    """Raised when templates or restraints cannot satisfy the prepared contract."""


@dataclass(frozen=True)
class PreparedInferenceResources:
    """Resolved native options consumed when the model path is connected."""

    template_hits_path: Path | None
    template_cif_directory: Path | None
    use_sequence_hash_for_template_lookup: bool
    constraint_path: Path | None


def _write_bytes_atomic(path: Path, data: bytes) -> Path:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_bytes(data)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def _read_m8_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line_number, row in enumerate(csv.reader(io.StringIO(text), delimiter="\t"), 1):
        if not row or all(not field.strip() for field in row):
            continue
        if len(row) not in (12, 13):
            raise PreparedResourceError(
                f"Template M8 line {line_number} has {len(row)} columns; expected 12 or 13"
            )
        rows.append(row)
    return rows


def _m8_text(rows: Sequence[Sequence[str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter="\t", lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue()


def _validate_template_query_ids(
    rows: Sequence[Sequence[str]],
    prepared: PreparedInput,
    query_sequences: Sequence[str],
    query_id_mode: str,
) -> None:
    if query_id_mode == "entity_name":
        expected = {
            entity_id
            for entity in prepared.sequences
            if entity.kind == "protein"
            for entity_id in entity.ids
        }
    else:
        expected = {hash_sequence(sequence) for sequence in query_sequences}
    unexpected = sorted({row[0] for row in rows} - expected)
    if unexpected:
        raise PreparedResourceError(
            f"Template M8 query IDs do not match query_id_mode={query_id_mode!r}: "
            f"{', '.join(unexpected)}"
        )


def _remap_server_query_ids(
    rows: list[list[str]], expanded_query_sequences: Sequence[str]
) -> None:
    query_map = {
        str(query_id): hash_sequence(sequence)
        for query_id, sequence in enumerate(expanded_query_sequences, start=101)
    }
    for row in rows:
        try:
            row[0] = query_map[row[0]]
        except KeyError as error:
            raise PreparedResourceError(
                f"ColabFold template M8 has unknown query ID {row[0]!r}"
            ) from error


def _template_pdb_ids(rows: Sequence[Sequence[str]]) -> list[str]:
    pdb_ids: set[str] = set()
    for row in rows:
        subject = row[1]
        subject_parts = subject.split("_")
        if len(subject_parts) != 2:
            raise PreparedResourceError(
                f"Template M8 subject ID must be '<PDB>_<chain>': {subject!r}"
            )
        pdb_id, chain_id = subject_parts
        if not pdb_id or not chain_id:
            raise PreparedResourceError(f"Invalid template M8 subject ID: {subject!r}")
        pdb_ids.add(pdb_id.upper())
    return sorted(pdb_ids)


def _find_local_cif(directory: Path, pdb_id: str) -> Path | None:
    supported_suffixes = (".cif.gz", ".cif", ".mmcif.gz", ".mmcif", ".cif.zst")
    candidates = {
        path.name.lower(): path for path in directory.iterdir() if path.is_file()
    }
    for suffix in supported_suffixes:
        candidate = candidates.get(f"{pdb_id}{suffix}".lower())
        if candidate is not None:
            return candidate
    return None


def _materialize_cif(
    pdb_id: str,
    output_directory: Path,
    source_directory: Path | None,
    downloader: TemplateDownloader | None,
) -> Path:
    output_path = output_directory / f"{pdb_id}.cif.gz"
    source_path = (
        None if source_directory is None else _find_local_cif(source_directory, pdb_id)
    )
    if source_path is None:
        if downloader is None:
            from chai_lab.data.io.rcsb import download_cif_file

            downloader = download_cif_file
        output_directory.mkdir(parents=True, exist_ok=True)
        downloaded = downloader(pdb_id, output_directory)
        if downloaded.resolve() != output_path.resolve():
            text = read_text_auto(downloaded)
            _write_bytes_atomic(
                output_path, gzip.compress(text.encode("utf-8"), mtime=0)
            )
    else:
        text = read_text_auto(source_path)
        _write_bytes_atomic(output_path, gzip.compress(text.encode("utf-8"), mtime=0))
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise PreparedResourceError(f"Template CIF was not materialized: {output_path}")
    return output_path


def materialize_template_bundle(
    *,
    prepared: PreparedInput,
    resolved: PreparedInput,
    output_manifest: Path,
    query_sequences: Sequence[str],
    expanded_query_sequences: Sequence[str],
    server_m8_text: str | None,
    downloader: TemplateDownloader | None = None,
) -> PreparedTemplates | None:
    """Publish canonical M8/CIF resources and return manifest-relative fields."""
    if prepared.templates is not None and server_m8_text is not None:
        raise PreparedResourceError(
            "Cannot combine declared templates with template-server results"
        )
    if prepared.templates is None and server_m8_text is None:
        return None

    source_cif_directory: Path | None
    if server_m8_text is not None:
        rows = _read_m8_rows(server_m8_text)
        _remap_server_query_ids(rows, expanded_query_sequences)
        query_id_mode = "sequence_hash"
        source_cif_directory = None
    else:
        assert resolved.templates is not None
        rows = _read_m8_rows(read_text_auto(resolved.templates.hits_path))
        query_id_mode = resolved.templates.query_id_mode
        source_cif_directory = resolved.templates.cif_directory

    if not rows:
        return None
    _validate_template_query_ids(rows, prepared, query_sequences, query_id_mode)

    template_directory = output_manifest.parent / "templates"
    cif_directory = template_directory / "cifs"
    m8_path = _write_bytes_atomic(
        template_directory / "all_chain_templates.m8",
        _m8_text(rows).encode("utf-8"),
    )
    for pdb_id in _template_pdb_ids(rows):
        _materialize_cif(pdb_id, cif_directory, source_cif_directory, downloader)

    return PreparedTemplates(
        hits_path=Path(os.path.relpath(m8_path, output_manifest.parent)),
        cif_directory=Path(os.path.relpath(cif_directory, output_manifest.parent)),
        query_id_mode=query_id_mode,
    )


def _automatic_subchain_id(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    value = ""
    while index >= 0:
        value = alphabet[index % len(alphabet)] + value
        index = index // len(alphabet) - 1
    return value


def materialize_restraint_bundle(
    *,
    prepared: PreparedInput,
    resolved: PreparedInput,
    output_manifest: Path,
) -> Path | None:
    """Validate a native restraint CSV and publish it inside the target bundle."""
    if resolved.constraint_path is None:
        return None
    text = read_text_auto(resolved.constraint_path)
    with tempfile.TemporaryDirectory(prefix="chai_restraint_validate_") as temporary:
        validation_path = Path(temporary) / "restraints.csv"
        validation_path.write_text(text, encoding="utf-8")
        from chai_lab.data.parsing.restraints import parse_pairwise_table

        try:
            interactions = parse_pairwise_table(validation_path)
        except Exception as error:
            raise PreparedResourceError(
                f"Invalid native Chai restraint CSV: {error}"
            ) from error

    if prepared.entity_ids_as_cif_chains:
        allowed_chains = {
            entity_id for entity in prepared.sequences for entity_id in entity.ids
        }
    else:
        chain_count = sum(len(entity.ids) for entity in prepared.sequences)
        allowed_chains = {_automatic_subchain_id(index) for index in range(chain_count)}
    unknown_chains = sorted(
        {
            chain
            for interaction in interactions
            for chain in (interaction.chainA, interaction.chainB)
        }
        - allowed_chains
    )
    if unknown_chains:
        raise PreparedResourceError(
            "Restraint chain IDs do not match entity_ids_as_cif_chains: "
            + ", ".join(unknown_chains)
        )

    output_path = _write_bytes_atomic(
        output_manifest.parent / "constraints" / f"{prepared.name}.restraints.csv",
        text.encode("utf-8"),
    )
    return Path(os.path.relpath(output_path, output_manifest.parent))


def get_prepared_inference_resources(
    prepared: PreparedInput,
) -> PreparedInferenceResources:
    """Translate a resolved prepared input to native template/restraint options."""
    templates = prepared.templates
    if templates is not None and (
        not templates.hits_path.is_absolute()
        or not templates.cif_directory.is_absolute()
    ):
        raise PreparedResourceError(
            "Template paths must be resolved; call validate_resources() first"
        )
    if (
        prepared.constraint_path is not None
        and not prepared.constraint_path.is_absolute()
    ):
        raise PreparedResourceError(
            "constraint_path must be resolved; call validate_resources() first"
        )
    return PreparedInferenceResources(
        template_hits_path=None if templates is None else templates.hits_path,
        template_cif_directory=(None if templates is None else templates.cif_directory),
        use_sequence_hash_for_template_lookup=(
            False if templates is None else templates.query_id_mode == "sequence_hash"
        ),
        constraint_path=prepared.constraint_path,
    )
