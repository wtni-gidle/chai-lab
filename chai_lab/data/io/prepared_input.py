# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

"""Schema and path handling for EnsembleFold prepared Chai-1 inputs."""

import json
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chai_lab.data.parsing.msas.data_source import MSADataSource

PREPARED_INPUT_VERSION = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_SEQUENCE_HASH = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_FIELDS = {
    "version",
    "name",
    "fasta_path",
    "use_esm_embeddings",
    "fasta_names_as_cif_chains",
    "msas",
    "templates",
    "constraint_path",
}
_MSA_FIELDS = {
    "chains",
    "sequence_hash",
    "paired_msa",
    "unpaired_msa",
    "unpaired_source_database",
}
_TEMPLATE_FIELDS = {"hits_path", "cif_directory", "query_id_mode"}
_QUERY_ID_MODES = {"entity_name", "sequence_hash"}
_MSA_SOURCES = {"auto", *(source.value for source in MSADataSource)}


class PreparedInputError(ValueError):
    """Raised when a prepared input does not satisfy the wrapper contract."""


def validate_target_name(name: str) -> str:
    """Validate a target name that will be used in directory and file names."""
    if not isinstance(name, str) or not _SAFE_NAME.fullmatch(name):
        raise PreparedInputError(
            "name must contain only ASCII letters, digits, '_', '-', and '.', "
            "and must start with a letter, digit, or '_'"
        )
    return name


def _expect_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreparedInputError(f"{location} must be a JSON object")
    return value


def _expect_exact_fields(
    value: Mapping[str, Any], expected: set[str], location: str
) -> None:
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing:
        raise PreparedInputError(
            f"{location} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise PreparedInputError(
            f"{location} has unknown fields: {', '.join(sorted(unknown))}"
        )


def _expect_bool(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise PreparedInputError(f"{location} must be a boolean")
    return value


def _expect_nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PreparedInputError(f"{location} must be a non-empty string")
    return value


def _expect_path(value: Any, location: str) -> Path:
    return Path(_expect_nonempty_string(value, location))


def _optional_path(value: Any, location: str) -> Path | None:
    if value is None:
        return None
    return _expect_path(value, location)


def _resolve_path(path: Path, manifest_path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (manifest_path.expanduser().resolve().parent / path).resolve()


@dataclass(frozen=True)
class PreparedMsa:
    """Persistent paired and unpaired MSAs for one unique protein sequence."""

    chains: tuple[str, ...]
    sequence_hash: str
    paired_msa: Path
    unpaired_msa: Path
    unpaired_source_database: str

    @classmethod
    def from_dict(cls, value: Any, index: int) -> "PreparedMsa":
        location = f"msas[{index}]"
        data = _expect_mapping(value, location)
        _expect_exact_fields(data, _MSA_FIELDS, location)

        raw_chains = data["chains"]
        if not isinstance(raw_chains, list) or not raw_chains:
            raise PreparedInputError(f"{location}.chains must be a non-empty list")
        chains = tuple(
            _expect_nonempty_string(chain, f"{location}.chains[{chain_index}]")
            for chain_index, chain in enumerate(raw_chains)
        )
        if len(chains) != len(set(chains)):
            raise PreparedInputError(f"{location}.chains contains duplicates")

        source = _expect_nonempty_string(
            data["unpaired_source_database"],
            f"{location}.unpaired_source_database",
        )
        if source not in _MSA_SOURCES:
            raise PreparedInputError(
                f"{location}.unpaired_source_database must be 'auto' or a "
                "Chai MSA data source"
            )

        sequence_hash = _expect_nonempty_string(
            data["sequence_hash"], f"{location}.sequence_hash"
        )
        if not _SEQUENCE_HASH.fullmatch(sequence_hash):
            raise PreparedInputError(
                f"{location}.sequence_hash must be a lowercase SHA-256 hex digest"
            )

        return cls(
            chains=chains,
            sequence_hash=sequence_hash,
            paired_msa=_expect_path(data["paired_msa"], f"{location}.paired_msa"),
            unpaired_msa=_expect_path(data["unpaired_msa"], f"{location}.unpaired_msa"),
            unpaired_source_database=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chains": list(self.chains),
            "sequence_hash": self.sequence_hash,
            "paired_msa": os.fspath(self.paired_msa),
            "unpaired_msa": os.fspath(self.unpaired_msa),
            "unpaired_source_database": self.unpaired_source_database,
        }

    def resolved(self, manifest_path: Path) -> "PreparedMsa":
        return PreparedMsa(
            chains=self.chains,
            sequence_hash=self.sequence_hash,
            paired_msa=_resolve_path(self.paired_msa, manifest_path),
            unpaired_msa=_resolve_path(self.unpaired_msa, manifest_path),
            unpaired_source_database=self.unpaired_source_database,
        )


@dataclass(frozen=True)
class PreparedTemplates:
    """Template hits and the job-local structure directory used to resolve them."""

    hits_path: Path
    cif_directory: Path
    query_id_mode: str

    @classmethod
    def from_dict(cls, value: Any) -> "PreparedTemplates":
        location = "templates"
        data = _expect_mapping(value, location)
        _expect_exact_fields(data, _TEMPLATE_FIELDS, location)
        query_id_mode = _expect_nonempty_string(
            data["query_id_mode"], "templates.query_id_mode"
        )
        if query_id_mode not in _QUERY_ID_MODES:
            raise PreparedInputError(
                "templates.query_id_mode must be 'entity_name' or 'sequence_hash'"
            )
        return cls(
            hits_path=_expect_path(data["hits_path"], "templates.hits_path"),
            cif_directory=_expect_path(
                data["cif_directory"], "templates.cif_directory"
            ),
            query_id_mode=query_id_mode,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "hits_path": os.fspath(self.hits_path),
            "cif_directory": os.fspath(self.cif_directory),
            "query_id_mode": self.query_id_mode,
        }

    def resolved(self, manifest_path: Path) -> "PreparedTemplates":
        return PreparedTemplates(
            hits_path=_resolve_path(self.hits_path, manifest_path),
            cif_directory=_resolve_path(self.cif_directory, manifest_path),
            query_id_mode=self.query_id_mode,
        )


@dataclass(frozen=True)
class PreparedInput:
    """Versioned, editable manifest for one Chai-1 prediction target."""

    version: int
    name: str
    fasta_path: Path
    use_esm_embeddings: bool
    fasta_names_as_cif_chains: bool
    msas: tuple[PreparedMsa, ...]
    templates: PreparedTemplates | None
    constraint_path: Path | None

    @classmethod
    def from_dict(cls, value: Any) -> "PreparedInput":
        data = _expect_mapping(value, "prepared input")
        _expect_exact_fields(data, _TOP_LEVEL_FIELDS, "prepared input")

        version = data["version"]
        if type(version) is not int or version != PREPARED_INPUT_VERSION:
            raise PreparedInputError(
                f"version must be the integer {PREPARED_INPUT_VERSION}"
            )

        raw_msas = data["msas"]
        if not isinstance(raw_msas, list):
            raise PreparedInputError("msas must be a list")
        msas = tuple(
            PreparedMsa.from_dict(msa, index) for index, msa in enumerate(raw_msas)
        )

        seen_chains: set[str] = set()
        seen_hashes: set[str] = set()
        for index, msa in enumerate(msas):
            duplicate_chains = seen_chains.intersection(msa.chains)
            if duplicate_chains:
                raise PreparedInputError(
                    f"msas[{index}] reuses chains: {', '.join(sorted(duplicate_chains))}"
                )
            if msa.sequence_hash in seen_hashes:
                raise PreparedInputError(
                    f"msas[{index}] reuses sequence_hash {msa.sequence_hash!r}"
                )
            seen_chains.update(msa.chains)
            seen_hashes.add(msa.sequence_hash)

        raw_templates = data["templates"]
        templates = (
            None
            if raw_templates is None
            else PreparedTemplates.from_dict(raw_templates)
        )

        return cls(
            version=version,
            name=validate_target_name(data["name"]),
            fasta_path=_expect_path(data["fasta_path"], "fasta_path"),
            use_esm_embeddings=_expect_bool(
                data["use_esm_embeddings"], "use_esm_embeddings"
            ),
            fasta_names_as_cif_chains=_expect_bool(
                data["fasta_names_as_cif_chains"],
                "fasta_names_as_cif_chains",
            ),
            msas=msas,
            templates=templates,
            constraint_path=_optional_path(data["constraint_path"], "constraint_path"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "fasta_path": os.fspath(self.fasta_path),
            "use_esm_embeddings": self.use_esm_embeddings,
            "fasta_names_as_cif_chains": self.fasta_names_as_cif_chains,
            "msas": [msa.to_dict() for msa in self.msas],
            "templates": None if self.templates is None else self.templates.to_dict(),
            "constraint_path": (
                None
                if self.constraint_path is None
                else os.fspath(self.constraint_path)
            ),
        }

    def resolved(self, manifest_path: str | Path) -> "PreparedInput":
        """Return a copy with every declared resource path made absolute."""
        manifest_path = Path(manifest_path)
        return PreparedInput(
            version=self.version,
            name=self.name,
            fasta_path=_resolve_path(self.fasta_path, manifest_path),
            use_esm_embeddings=self.use_esm_embeddings,
            fasta_names_as_cif_chains=self.fasta_names_as_cif_chains,
            msas=tuple(msa.resolved(manifest_path) for msa in self.msas),
            templates=(
                None
                if self.templates is None
                else self.templates.resolved(manifest_path)
            ),
            constraint_path=(
                None
                if self.constraint_path is None
                else _resolve_path(self.constraint_path, manifest_path)
            ),
        )

    def validate_resources(self, manifest_path: str | Path) -> "PreparedInput":
        """Resolve paths, require declared resources to exist, and return the copy."""
        resolved = self.resolved(manifest_path)
        _require_file(resolved.fasta_path, "fasta_path")
        for index, msa in enumerate(resolved.msas):
            _require_file(msa.paired_msa, f"msas[{index}].paired_msa")
            _require_file(msa.unpaired_msa, f"msas[{index}].unpaired_msa")
        if resolved.templates is not None:
            _require_file(resolved.templates.hits_path, "templates.hits_path")
            _require_directory(
                resolved.templates.cif_directory, "templates.cif_directory"
            )
        if resolved.constraint_path is not None:
            _require_file(resolved.constraint_path, "constraint_path")
        return resolved


def _require_file(path: Path, location: str) -> None:
    if not path.is_file():
        raise PreparedInputError(f"{location} does not exist or is not a file: {path}")


def _require_directory(path: Path, location: str) -> None:
    if not path.is_dir():
        raise PreparedInputError(
            f"{location} does not exist or is not a directory: {path}"
        )


def load_prepared_input(path: str | Path) -> PreparedInput:
    """Read and validate a prepared JSON without resolving its declared paths."""
    path = Path(path).expanduser()
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as error:
        raise PreparedInputError(f"Invalid JSON in {path}: {error}") from error
    except OSError as error:
        raise PreparedInputError(
            f"Cannot read prepared input {path}: {error}"
        ) from error
    return PreparedInput.from_dict(value)


def prepared_input_path(output_dir: str | Path, name: str) -> Path:
    """Return ``<output>/<name>/<name>_data.json`` for a validated name."""
    name = validate_target_name(name)
    return Path(output_dir).expanduser().resolve() / name / f"{name}_data.json"


def write_prepared_input(prepared: PreparedInput, path: str | Path) -> Path:
    """Atomically write a validated prepared input and return its absolute path."""
    prepared = PreparedInput.from_dict(prepared.to_dict())
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(prepared.to_dict(), handle, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path
