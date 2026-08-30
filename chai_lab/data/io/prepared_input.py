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
from itertools import pairwise
from pathlib import Path
from typing import Any

from chai_lab.data.io.entity_input import Input
from chai_lab.data.parsing.structure.entity_type import EntityType

PREPARED_INPUT_VERSION = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_TOP_LEVEL_FIELDS = {
    "version",
    "name",
    "sequences",
    "use_esm_embeddings",
    "entity_ids_as_cif_chains",
    "constraint_path",
}
_ENTITY_TYPES = {"protein", "rna", "dna", "ligand", "glycan"}
_ENTITY_REQUIRED_FIELDS = {
    "protein": {"id", "sequence"},
    "rna": {"id", "sequence"},
    "dna": {"id", "sequence"},
    "ligand": {"id", "smiles"},
    "glycan": {"id", "sequence"},
}
_ENTITY_OPTIONAL_FIELDS = {
    "protein": {
        "pairedMsa",
        "pairedMsaPath",
        "unpairedMsa",
        "unpairedMsaPath",
        "unpairedMsaFallbackSource",
        "templates",
    },
    "rna": set(),
    "dna": set(),
    "ligand": set(),
    "glycan": set(),
}
_ENTITY_TYPE_VALUES = {
    "protein": EntityType.PROTEIN.value,
    "rna": EntityType.RNA.value,
    "dna": EntityType.DNA.value,
    "ligand": EntityType.LIGAND.value,
    "glycan": EntityType.MANUAL_GLYCAN.value,
}
_UNPAIRED_FALLBACK_SOURCES = {
    "auto",
    "bfd_uniclust",
    "mgnify",
    "uniprot",
    "uniref90",
}
_TEMPLATE_REQUIRED_FIELDS = {"queryIndices", "templateIndices"}
_TEMPLATE_OPTIONAL_FIELDS = {"mmcif", "mmcifPath"}


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


def _validate_entity_id(entity_id: Any, location: str) -> str:
    if not isinstance(entity_id, str) or not _SAFE_NAME.fullmatch(entity_id):
        raise PreparedInputError(
            f"{location} must contain only ASCII letters, digits, '_', '-', and '.', "
            "and must start with a letter, digit, or '_'"
        )
    return entity_id


def _expect_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreparedInputError(f"{location} must be a JSON object")
    return value


def _expect_fields(
    value: Mapping[str, Any],
    required: set[str],
    optional: set[str],
    location: str,
) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise PreparedInputError(
            f"{location} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise PreparedInputError(
            f"{location} has unknown fields: {', '.join(sorted(unknown))}"
        )


def _expect_exact_fields(
    value: Mapping[str, Any], expected: set[str], location: str
) -> None:
    _expect_fields(value, expected, set(), location)


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


def _optional_msa_content(value: Any, location: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or "\x00" in value:
        raise PreparedInputError(f"{location} must be a string or null")
    return value


def _expect_indices(value: Any, location: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise PreparedInputError(f"{location} must be a non-empty list")
    indices = tuple(value)
    if any(type(index) is not int or index < 0 for index in indices):
        raise PreparedInputError(f"{location} must contain non-negative integers")
    if any(right <= left for left, right in pairwise(indices)):
        raise PreparedInputError(f"{location} must be strictly increasing")
    return indices


def _resolve_path(path: Path, manifest_path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (manifest_path.expanduser().resolve().parent / path).resolve()


@dataclass(frozen=True)
class PreparedTemplate:
    """One AF3-style template structure and its Chai-derived residue mapping."""

    mmcif: str | None
    mmcif_path: Path | None
    query_indices: tuple[int, ...]
    template_indices: tuple[int, ...]

    @classmethod
    def from_dict(cls, value: Any, location: str) -> "PreparedTemplate":
        data = _expect_mapping(value, location)
        _expect_fields(
            data,
            _TEMPLATE_REQUIRED_FIELDS,
            _TEMPLATE_OPTIONAL_FIELDS,
            location,
        )
        mmcif = data.get("mmcif")
        mmcif_path = data.get("mmcifPath")
        if mmcif is not None:
            mmcif = _expect_nonempty_string(mmcif, f"{location}.mmcif")
        if mmcif_path is not None:
            mmcif_path = _expect_path(mmcif_path, f"{location}.mmcifPath")
        if (mmcif is None) == (mmcif_path is None):
            raise PreparedInputError(
                f"{location} must set exactly one of mmcif/mmcifPath"
            )

        query_indices = _expect_indices(
            data["queryIndices"], f"{location}.queryIndices"
        )
        template_indices = _expect_indices(
            data["templateIndices"], f"{location}.templateIndices"
        )
        if len(query_indices) != len(template_indices):
            raise PreparedInputError(
                f"{location}.queryIndices and templateIndices must have equal length"
            )
        return cls(
            mmcif=mmcif,
            mmcif_path=mmcif_path,
            query_indices=query_indices,
            template_indices=template_indices,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "queryIndices": list(self.query_indices),
            "templateIndices": list(self.template_indices),
        }
        if self.mmcif_path is None:
            data["mmcif"] = self.mmcif
        else:
            data["mmcifPath"] = os.fspath(self.mmcif_path)
        return data

    def resolved(self, manifest_path: Path) -> "PreparedTemplate":
        return PreparedTemplate(
            mmcif=self.mmcif,
            mmcif_path=(
                None
                if self.mmcif_path is None
                else _resolve_path(self.mmcif_path, manifest_path)
            ),
            query_indices=self.query_indices,
            template_indices=self.template_indices,
        )


@dataclass(frozen=True)
class PreparedEntity:
    """One unique molecular entity, possibly instantiated as several chains."""

    kind: str
    ids: tuple[str, ...]
    sequence: str
    paired_msa: str | None = None
    paired_msa_path: Path | None = None
    unpaired_msa: str | None = None
    unpaired_msa_path: Path | None = None
    unpaired_msa_fallback_source: str = "auto"
    templates: tuple[PreparedTemplate, ...] | None = None

    @classmethod
    def from_dict(cls, value: Any, index: int) -> "PreparedEntity":
        location = f"sequences[{index}]"
        wrapper = _expect_mapping(value, location)
        if len(wrapper) != 1:
            raise PreparedInputError(
                f"{location} must contain exactly one molecular entity type"
            )
        kind = next(iter(wrapper))
        if kind not in _ENTITY_TYPES:
            raise PreparedInputError(f"{location} has unsupported entity type {kind!r}")

        entity_location = f"{location}.{kind}"
        data = _expect_mapping(wrapper[kind], entity_location)
        _expect_fields(
            data,
            _ENTITY_REQUIRED_FIELDS[kind],
            _ENTITY_OPTIONAL_FIELDS[kind],
            entity_location,
        )

        raw_ids = data["id"]
        if not isinstance(raw_ids, list) or not raw_ids:
            raise PreparedInputError(f"{entity_location}.id must be a non-empty list")
        ids = tuple(
            _validate_entity_id(entity_id, f"{entity_location}.id[{id_index}]")
            for id_index, entity_id in enumerate(raw_ids)
        )
        if len(ids) != len(set(ids)):
            raise PreparedInputError(f"{entity_location}.id contains duplicates")

        sequence_field = "smiles" if kind == "ligand" else "sequence"
        sequence = _expect_nonempty_string(
            data[sequence_field], f"{entity_location}.{sequence_field}"
        )

        paired_msa = None
        paired_msa_path = None
        unpaired_msa = None
        unpaired_msa_path = None
        fallback_source = "auto"
        templates = None
        if kind == "protein":
            paired_msa = _optional_msa_content(
                data.get("pairedMsa"), f"{entity_location}.pairedMsa"
            )
            paired_msa_path = _optional_path(
                data.get("pairedMsaPath"), f"{entity_location}.pairedMsaPath"
            )
            if paired_msa is not None and paired_msa_path is not None:
                raise PreparedInputError(
                    f"{entity_location} can set only one of pairedMsa/pairedMsaPath"
                )
            unpaired_msa = _optional_msa_content(
                data.get("unpairedMsa"), f"{entity_location}.unpairedMsa"
            )
            unpaired_msa_path = _optional_path(
                data.get("unpairedMsaPath"), f"{entity_location}.unpairedMsaPath"
            )
            if unpaired_msa is not None and unpaired_msa_path is not None:
                raise PreparedInputError(
                    f"{entity_location} can set only one of unpairedMsa/unpairedMsaPath"
                )
            fallback_source = _expect_nonempty_string(
                data.get("unpairedMsaFallbackSource", "auto"),
                f"{entity_location}.unpairedMsaFallbackSource",
            )
            if fallback_source not in _UNPAIRED_FALLBACK_SOURCES:
                raise PreparedInputError(
                    f"{entity_location}.unpairedMsaFallbackSource must be "
                    "'auto' or a supported Chai MSA source"
                )
            raw_templates = data.get("templates")
            if raw_templates is not None:
                if not isinstance(raw_templates, list):
                    raise PreparedInputError(
                        f"{entity_location}.templates must be a list or null"
                    )
                templates = tuple(
                    PreparedTemplate.from_dict(
                        template,
                        f"{entity_location}.templates[{template_index}]",
                    )
                    for template_index, template in enumerate(raw_templates)
                )

        return cls(
            kind=kind,
            ids=ids,
            sequence=sequence,
            paired_msa=paired_msa,
            paired_msa_path=paired_msa_path,
            unpaired_msa=unpaired_msa,
            unpaired_msa_path=unpaired_msa_path,
            unpaired_msa_fallback_source=fallback_source,
            templates=templates,
        )

    def to_dict(self) -> dict[str, Any]:
        sequence_field = "smiles" if self.kind == "ligand" else "sequence"
        entity: dict[str, Any] = {
            "id": list(self.ids),
            sequence_field: self.sequence,
        }
        if self.kind == "protein":
            if self.paired_msa_path is None:
                entity["pairedMsa"] = self.paired_msa
            else:
                entity["pairedMsaPath"] = os.fspath(self.paired_msa_path)
            if self.unpaired_msa_path is None:
                entity["unpairedMsa"] = self.unpaired_msa
            else:
                entity["unpairedMsaPath"] = os.fspath(self.unpaired_msa_path)
            entity["unpairedMsaFallbackSource"] = self.unpaired_msa_fallback_source
            entity["templates"] = (
                None
                if self.templates is None
                else [template.to_dict() for template in self.templates]
            )
        return {self.kind: entity}

    def resolved(self, manifest_path: Path) -> "PreparedEntity":
        return PreparedEntity(
            kind=self.kind,
            ids=self.ids,
            sequence=self.sequence,
            paired_msa=self.paired_msa,
            paired_msa_path=(
                None
                if self.paired_msa_path is None
                else _resolve_path(self.paired_msa_path, manifest_path)
            ),
            unpaired_msa=self.unpaired_msa,
            unpaired_msa_path=(
                None
                if self.unpaired_msa_path is None
                else _resolve_path(self.unpaired_msa_path, manifest_path)
            ),
            unpaired_msa_fallback_source=self.unpaired_msa_fallback_source,
            templates=(
                None
                if self.templates is None
                else tuple(
                    template.resolved(manifest_path) for template in self.templates
                )
            ),
        )

    def to_chai_inputs(self) -> list[Input]:
        """Expand entity IDs into the same Input objects produced by FASTA parsing."""
        entity_type = _ENTITY_TYPE_VALUES[self.kind]
        return [
            Input(
                sequence=self.sequence,
                entity_type=entity_type,
                entity_name=entity_id,
            )
            for entity_id in self.ids
        ]


@dataclass(frozen=True)
class PreparedInput:
    """Versioned, self-contained manifest for one Chai-1 prediction target."""

    version: int
    name: str
    sequences: tuple[PreparedEntity, ...]
    use_esm_embeddings: bool
    entity_ids_as_cif_chains: bool
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

        raw_sequences = data["sequences"]
        if not isinstance(raw_sequences, list) or not raw_sequences:
            raise PreparedInputError("sequences must be a non-empty list")
        sequences = tuple(
            PreparedEntity.from_dict(entity, index)
            for index, entity in enumerate(raw_sequences)
        )

        seen_ids: set[str] = set()
        seen_proteins: dict[str, int] = {}
        for index, entity in enumerate(sequences):
            duplicate_ids = seen_ids.intersection(entity.ids)
            if duplicate_ids:
                raise PreparedInputError(
                    f"sequences[{index}] reuses entity IDs: "
                    f"{', '.join(sorted(duplicate_ids))}"
                )
            seen_ids.update(entity.ids)
            if entity.kind == "protein":
                normalized_sequence = entity.sequence.upper()
                if normalized_sequence in seen_proteins:
                    previous_index = seen_proteins[normalized_sequence]
                    raise PreparedInputError(
                        f"sequences[{index}] duplicates the protein sequence from "
                        f"sequences[{previous_index}]; merge their id lists"
                    )
                seen_proteins[normalized_sequence] = index

        return cls(
            version=version,
            name=validate_target_name(data["name"]),
            sequences=sequences,
            use_esm_embeddings=_expect_bool(
                data["use_esm_embeddings"], "use_esm_embeddings"
            ),
            entity_ids_as_cif_chains=_expect_bool(
                data["entity_ids_as_cif_chains"],
                "entity_ids_as_cif_chains",
            ),
            constraint_path=_optional_path(data["constraint_path"], "constraint_path"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "sequences": [entity.to_dict() for entity in self.sequences],
            "use_esm_embeddings": self.use_esm_embeddings,
            "entity_ids_as_cif_chains": self.entity_ids_as_cif_chains,
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
            sequences=tuple(
                entity.resolved(manifest_path) for entity in self.sequences
            ),
            use_esm_embeddings=self.use_esm_embeddings,
            entity_ids_as_cif_chains=self.entity_ids_as_cif_chains,
            constraint_path=(
                None
                if self.constraint_path is None
                else _resolve_path(self.constraint_path, manifest_path)
            ),
        )

    def validate_resources(self, manifest_path: str | Path) -> "PreparedInput":
        """Resolve paths, require every declared resource, and return the copy."""
        resolved = self.resolved(manifest_path)
        for index, entity in enumerate(resolved.sequences):
            if entity.paired_msa_path is not None:
                _require_file(
                    entity.paired_msa_path,
                    f"sequences[{index}].protein.pairedMsaPath",
                )
            if entity.unpaired_msa_path is not None:
                _require_file(
                    entity.unpaired_msa_path,
                    f"sequences[{index}].protein.unpairedMsaPath",
                )
            if entity.templates is not None:
                for template_index, template in enumerate(entity.templates):
                    if template.mmcif_path is not None:
                        _require_file(
                            template.mmcif_path,
                            f"sequences[{index}].protein.templates["
                            f"{template_index}].mmcifPath",
                        )
        if resolved.constraint_path is not None:
            _require_file(resolved.constraint_path, "constraint_path")
        return resolved

    def to_chai_inputs(self) -> list[Input]:
        """Convert JSON entities to the native entity inputs used by Chai-1."""
        return [
            chai_input
            for entity in self.sequences
            for chai_input in entity.to_chai_inputs()
        ]


def _require_file(path: Path, location: str) -> None:
    if not path.is_file():
        raise PreparedInputError(f"{location} does not exist or is not a file: {path}")


def load_prepared_input(path: str | Path) -> PreparedInput:
    """Read and validate a self-contained prepared JSON."""
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
