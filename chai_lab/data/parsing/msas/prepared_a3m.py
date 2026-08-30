# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

"""Conversion of persisted paired/unpaired A3Ms to Chai's native MSA table."""

import string
from io import StringIO

import pandas as pd

from chai_lab.data.parsing.fasta import Fasta, read_fasta_content
from chai_lab.data.parsing.input_validation import constituents_of_modified_fasta
from chai_lab.data.parsing.msas.data_source import MSADataSource
from chai_lab.data.parsing.structure.entity_type import EntityType


class PreparedMSAError(ValueError):
    """Raised when a persisted A3M cannot be used for the declared protein."""


def protein_msa_query_sequence(sequence: str) -> str:
    """Return the exact protein sequence Chai uses for MSA lookup and hashing."""
    # Keep these imports lazy so reading the JSON schema remains lightweight.
    from chai_lab.data.parsing.fasta import get_residue_name
    from chai_lab.data.parsing.structure.sequence import protein_one_letter_sequence

    constituents = constituents_of_modified_fasta(sequence)
    if constituents is None:
        raise PreparedMSAError(f"Invalid Chai protein sequence: {sequence!r}")
    residue_names = [
        get_residue_name(residue, EntityType.PROTEIN) if len(residue) == 1 else residue
        for residue in constituents
    ]
    return protein_one_letter_sequence(residue_names)


def _read_a3m_text(text: str, location: str) -> list[Fasta]:
    if not text.strip():
        return []
    try:
        records = read_fasta_content(StringIO(text.replace("\x00", "")))
    except Exception as error:
        raise PreparedMSAError(f"Cannot parse {location} as A3M: {error}") from error
    if not records:
        raise PreparedMSAError(f"{location} contains no A3M records")
    return records


def _aligned_sequence(sequence: str) -> str:
    """Remove A3M insertions/skips while retaining aligned residues and gaps."""
    return "".join(
        character
        for character in sequence
        if character in string.ascii_uppercase or character == "-"
    )


def _validate_records(records: list[Fasta], expected_query: str, location: str) -> None:
    query = _aligned_sequence(records[0].sequence).replace("-", "")
    if query.upper() != expected_query.upper():
        raise PreparedMSAError(
            f"{location} query does not match the JSON protein sequence: "
            f"expected {expected_query!r}, found {query!r}"
        )
    expected_width = len(expected_query)
    for index, record in enumerate(records):
        width = len(_aligned_sequence(record.sequence))
        if width != expected_width:
            raise PreparedMSAError(
                f"{location} record {index} has aligned width {width}; "
                f"expected {expected_width}"
            )


def _is_padding_msa_row(sequence: str) -> bool:
    return bool(sequence) and set(sequence) == {"-"}


def _source_for_header(header: str, fallback_source: str = "auto") -> str:
    if header.startswith("UniRef"):
        return MSADataSource.UNIREF90.value
    if fallback_source != "auto":
        return fallback_source
    return MSADataSource.BFD_UNICLUST.value


def colabfold_a3ms_to_dataframe(
    *,
    query_sequence: str,
    paired_a3m: str | None,
    unpaired_a3m: str | None,
    unpaired_fallback_source: str = "auto",
) -> pd.DataFrame:
    """Reproduce Chai's native ColabFold paired/unpaired merge semantics."""
    paired_records = _read_a3m_text(paired_a3m or "", "paired_msa")
    unpaired_records = _read_a3m_text(unpaired_a3m or "", "unpaired_msa")
    if not paired_records and not unpaired_records:
        raise PreparedMSAError(
            "At least one paired or unpaired A3M must contain records"
        )
    if paired_records:
        _validate_records(paired_records, query_sequence, "paired_msa")
    if unpaired_records:
        _validate_records(unpaired_records, query_sequence, "unpaired_msa")

    paired_fasta: list[tuple[str, str, str]] = [
        (str(pairkey), record.header, record.sequence)
        for pairkey, record in enumerate(paired_records)
        if not _is_padding_msa_row(record.sequence)
    ]
    pairing_keys, paired_headers, paired_sequences = (
        zip(*paired_fasta, strict=True) if paired_fasta else ((), (), ())
    )
    unique_paired_sequences = set(paired_sequences)

    single_fasta = [
        record
        for index, record in enumerate(unpaired_records)
        if (
            (not paired_headers or index > 0)
            and not _is_padding_msa_row(record.sequence)
            and record.sequence not in unique_paired_sequences
        )
    ]
    single_headers = [record.header for record in single_fasta]
    single_sequences = [record.sequence for record in single_fasta]

    all_headers = list(paired_headers) + single_headers
    sources = [MSADataSource.QUERY.value]
    for index, header in enumerate(all_headers[1:], start=1):
        is_unpaired = index >= len(paired_headers)
        sources.append(
            _source_for_header(
                header,
                unpaired_fallback_source if is_unpaired else "auto",
            )
        )

    all_sequences = list(paired_sequences) + single_sequences
    all_pairing_keys = list(pairing_keys) + [""] * len(single_sequences)
    if not (len(all_sequences) == len(all_pairing_keys) == len(sources)):
        raise PreparedMSAError("Paired/unpaired A3M merge produced inconsistent rows")

    return pd.DataFrame(
        data={
            "sequence": all_sequences,
            "source_database": sources,
            "pairing_key": all_pairing_keys,
            "comment": "",
        }
    )
