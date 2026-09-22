# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

"""AF3-style persistence for templates parsed by Chai's native pipeline."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path

from chai_lab.data.io.compression import read_text_auto, write_zstd_text
from chai_lab.data.io.prepared_input import (
    PreparedEntity,
    PreparedInput,
    PreparedTemplate,
)


class PreparedTemplateError(ValueError):
    """Raised when prepared template structures or mappings are invalid."""


logger = logging.getLogger(__name__)


DEFAULT_MAX_TEMPLATE_DATE = None

NativeTemplateParser = Callable[
    [str, str, Path, Path, date | None], Sequence[PreparedTemplate]
]


def _validate_resource_names(resource_names: Sequence[str]) -> None:
    """Reject output names that alias on a case-insensitive filesystem."""
    seen: dict[str, str] = {}
    for name in resource_names:
        folded = name.casefold()
        previous = seen.get(folded)
        if previous is not None:
            raise ValueError(
                "Prepared resource names collide on a case-insensitive filesystem: "
                f"{previous!r} and {name!r}"
            )
        seen[folded] = name


def _validate_resource_names_against_directory(
    resource_names: Sequence[str], directory: Path
) -> None:
    """Reject case-only aliases of resources already present in a bundle."""
    if not directory.is_dir():
        return
    for name in resource_names:
        for path in directory.iterdir():
            if path.name.casefold() == name.casefold() and path.name != name:
                raise ValueError(
                    "Prepared resource names collide on a case-insensitive "
                    f"filesystem: {path.name!r} and {name!r}"
                )


def _template_resource_names(
    *,
    entity: PreparedEntity,
    templates: Sequence[PreparedTemplate],
    target_name: str,
) -> tuple[str, ...]:
    first_id = entity.ids[0]
    return tuple(
        f"{target_name}__{first_id}_template_{index}.cif.zst"
        for index in range(len(templates))
    )


def parse_max_template_date(value: str | date | None) -> date | None:
    """Normalize an ISO template cutoff while keeping it outside prepared JSON."""
    if value is None or isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise PreparedTemplateError(
            "max_template_date must use YYYY-MM-DD format"
        ) from error


def _single_chain_mmcif(cif_path: Path, chain_id: str) -> str:
    """Extract one auth chain while retaining Chai's source structure coordinates."""
    import gemmi

    source = gemmi.read_structure(str(cif_path))
    if not source or not source[0]:
        raise PreparedTemplateError(f"Template structure is empty: {cif_path}")
    selected = next((chain for chain in source[0] if chain.name == chain_id), None)
    if selected is None:
        raise PreparedTemplateError(
            f"Template chain {chain_id!r} is absent from {cif_path}"
        )
    source_polymer = selected.get_polymer()
    source_entity = source.get_entity_of(source_polymer)
    source_full_sequence = list(source_entity.full_sequence)
    if not source_full_sequence:
        raise PreparedTemplateError(
            f"Template chain {chain_id!r} has no entity polymer sequence"
        )

    structure = gemmi.Structure()
    structure.name = f"{source.name}_{chain_id}"
    structure.cell = source.cell
    structure.spacegroup_hm = source.spacegroup_hm
    model = gemmi.Model("1")
    chain = gemmi.Chain(chain_id)
    for residue in source_polymer:
        chain.add_residue(residue.clone())
    if not chain:
        raise PreparedTemplateError(
            f"Template chain {chain_id!r} has no polymer residues in {cif_path}"
        )
    model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    if len(structure.entities) != 1:
        raise PreparedTemplateError(
            f"Expected one entity after extracting template chain {chain_id!r}"
        )
    # Preserve the full polymer sequence in the CIF, including unresolved residues.
    # Public indices refer to this full sequence. Only the runtime adapter converts
    # them to positions in Chai's unresolved-filtered context.
    structure.entities[0].full_sequence = source_full_sequence
    structure.assign_subchains()
    structure.assign_label_seq_id()
    return structure.make_mmcif_document().as_string()


def _matches_native_template_features(native_loaded, prepared: PreparedTemplate) -> bool:
    """Require a saved template to reconstruct the exact native Chai features."""
    import torch

    try:
        rebuilt = _as_loaded_template(
            prepared,
            query_id=native_loaded.query_identifier,
            query_token_count=int(native_loaded.query_crop_indices.shape[0]),
        )
        if rebuilt is None:
            return False
        exact_fields = (
            "template_restype",
            "template_pseudo_beta_mask",
            "template_backbone_frame_mask",
        )
        float_fields = (
            "template_pseudo_beta_distances",
            "template_unit_vector",
        )
        for field in exact_fields:
            if not torch.equal(getattr(native_loaded, field), getattr(rebuilt, field)):
                logger.warning(
                    "Template %s round-trip mismatch in %s",
                    native_loaded.hit_identifier,
                    field,
                )
                return False
        for field in float_fields:
            if not torch.allclose(
                getattr(native_loaded, field),
                getattr(rebuilt, field),
                rtol=1e-5,
                atol=1e-5,
                equal_nan=True,
            ):
                logger.warning(
                    "Template %s round-trip mismatch in %s (rtol=1e-5, atol=1e-5)",
                    native_loaded.hit_identifier,
                    field,
                )
                return False
    except (IndexError, PreparedTemplateError, RuntimeError, ValueError):
        logger.warning(
            "Skipping template %s because its saved structure cannot reconstruct "
            "the native Chai template features",
            native_loaded.hit_identifier,
            exc_info=True,
        )
        return False
    return True


def prepared_template_from_loaded(loaded_template, mmcif: str) -> PreparedTemplate:
    """Translate a native filtered-context mapping to full polymer positions."""
    hit = loaded_template.template_hit
    valid = hit.hit_valid_mask
    query_indices = tuple(
        int(index)
        for index in loaded_template.template_query_match_indices[valid].tolist()
    )
    # index_select() preserves token_residue_index from the unfiltered structure:
    # these are zero-based label_seq positions, not author residue numbers.
    full_positions = loaded_template.template_hit_structure_context.token_residue_index
    template_indices = tuple(
        int(index)
        for index in full_positions[loaded_template.template_hit_indices[valid]].tolist()
    )
    if not query_indices:
        raise PreparedTemplateError("Native template has no mapped residues")
    return PreparedTemplate(
        mmcif=mmcif,
        mmcif_path=None,
        query_indices=query_indices,
        template_indices=template_indices,
    )


def parse_native_template_hits(
    query_id: str,
    query_sequence: str,
    m8_path: Path,
    cif_cache_directory: Path,
    max_template_date: date | None = DEFAULT_MAX_TEMPLATE_DATE,
) -> tuple[PreparedTemplate, ...]:
    """Run Chai's native M8/Kalign/structure filtering, then retain final mappings."""
    import torch

    from chai_lab.data.dataset.structure.all_atom_residue_tokenizer import (
        AllAtomResidueTokenizer,
    )
    from chai_lab.data.dataset.templates.load import get_template_data
    from chai_lab.data.parsing.templates.m8 import parse_m8_to_template_hits
    from chai_lab.data.sources.rdkit import RefConformerGenerator

    cif_cache_directory.mkdir(parents=True, exist_ok=True)
    hits = parse_m8_to_template_hits(
        query_pdb_id=query_id,
        query_sequence=query_sequence,
        m8_path=m8_path,
        template_cif_folder=cif_cache_directory,
        max_template_date=max_template_date,
    )
    loaded = get_template_data(
        template_hits=hits,
        query_crop_indices=torch.arange(len(query_sequence)),
        tokenizer=AllAtomResidueTokenizer(RefConformerGenerator()),
        strict_subsequence_check=True,
        drop_unresolved_from_hits=True,
    )

    prepared_templates: list[PreparedTemplate] = []
    retained_hits: list[str] = []
    discarded_hits: list[str] = []
    for template in loaded:
        hit = template.template_hit
        try:
            if hit.cif_path is None:
                raise PreparedTemplateError(
                    "Native template parsing did not retain a local CIF path"
                )
            prepared = prepared_template_from_loaded(
                template,
                _single_chain_mmcif(hit.cif_path, hit.chain_id),
            )
        except (ValueError, RuntimeError, IndexError, OSError) as error:
            discarded_hits.append(template.hit_identifier)
            logger.warning(
                "[%s] Discarding template %s during single-chain export: %s: %s",
                query_id,
                template.hit_identifier,
                type(error).__name__,
                error,
            )
            continue
        if not _matches_native_template_features(template, prepared):
            discarded_hits.append(template.hit_identifier)
            logger.warning(
                "[%s] Discarding template %s because prepared-template round-trip "
                "validation failed (see mismatch/error above)",
                query_id,
                template.hit_identifier,
            )
            continue
        prepared_templates.append(prepared)
        retained_hits.append(template.hit_identifier)
    logger.log(
        logging.WARNING if discarded_hits else logging.INFO,
        "[%s] Retained templates (%d/%d native-selected): %s; discarded: %s",
        query_id,
        len(retained_hits),
        len(loaded),
        ", ".join(
            f"template_{index}={hit_id}" for index, hit_id in enumerate(retained_hits)
        ) or "none",
        ", ".join(discarded_hits) or "none",
    )
    return tuple(prepared_templates)


def materialize_template_structures(
    *,
    entity: PreparedEntity,
    templates: Sequence[PreparedTemplate],
    target_name: str,
    output_manifest: Path,
) -> tuple[PreparedTemplate, ...]:
    """Externalize template mmCIFs beside MSAs and return manifest-relative paths."""
    msa_directory = output_manifest.parent / "msas"
    resource_names = _template_resource_names(
        entity=entity,
        templates=templates,
        target_name=target_name,
    )
    _validate_resource_names(resource_names)
    _validate_resource_names_against_directory(resource_names, msa_directory)
    template_contents: list[str] = []
    for template in templates:
        mmcif = (
            template.mmcif
            if template.mmcif_path is None
            else read_text_auto(template.mmcif_path)
        )
        if not mmcif:
            raise PreparedTemplateError("Template mmCIF content must not be empty")
        template_contents.append(mmcif)

    materialized: list[PreparedTemplate] = []
    for template, mmcif, resource_name in zip(
        templates,
        template_contents,
        resource_names,
        strict=True,
    ):
        output_path = write_zstd_text(
            msa_directory / resource_name,
            mmcif,
        )
        materialized.append(
            replace(
                template,
                mmcif=None,
                mmcif_path=output_path.relative_to(output_manifest.parent),
            )
        )
    logger.info(
        "[%s] Prepared templates for chains %s: %s",
        target_name,
        ", ".join(entity.ids),
        ", ".join(str(template.mmcif_path) for template in materialized) or "none",
    )
    return tuple(materialized)


def _load_template_structure_context(mmcif: str):
    """Return the native filtered context (also used by parity tests)."""
    return _load_template_structure_context_with_indices(mmcif)[0]


def _load_template_structure_context_with_indices(mmcif: str):
    """Apply the same protein extraction, tokenization, and unresolved filtering as Chai."""
    import gemmi
    import torch

    from chai_lab.data.dataset.structure.all_atom_residue_tokenizer import (
        AllAtomResidueTokenizer,
    )
    from chai_lab.data.features.token_utils import get_centre_positions_and_mask
    from chai_lab.data.parsing.structure.all_atom_entity_data import (
        structure_to_entities_data,
    )
    from chai_lab.data.parsing.structure.entity_type import EntityType
    from chai_lab.data.sources.rdkit import RefConformerGenerator

    structure = gemmi.read_structure_string(mmcif)
    entities = [
        entity
        for entity in structure_to_entities_data(structure, make_assembly=False)
        if entity.entity_type == EntityType.PROTEIN
    ]
    if len(entities) != 1:
        raise PreparedTemplateError(
            f"Prepared template must contain exactly one protein chain, got {len(entities)}"
        )
    if entities[0].has_modifications:
        raise PreparedTemplateError(
            "Modified-residue templates are not supported by Chai"
        )
    context = AllAtomResidueTokenizer(RefConformerGenerator()).tokenize_entity(
        entities[0]
    )
    if context is None:
        raise PreparedTemplateError("Chai failed to tokenize a prepared template")
    _, mask = get_centre_positions_and_mask(
        atom_gt_coords=context.atom_gt_coords,
        atom_exists_mask=context.atom_exists_mask,
        token_centre_atom_index=context.token_centre_atom_index,
        token_exists_mask=context.token_exists_mask,
    )
    (resolved_indices,) = torch.where(mask)
    filtered = context.index_select(resolved_indices)
    # Keep the native filter, and remember which full-sequence positions survived.
    # Do not align sequences again: repeats and gaps must not change correspondence.
    full_positions = tuple(int(index) for index in filtered.token_residue_index.tolist())
    return filtered, full_positions, len(entities[0].full_sequence)


def _as_loaded_template(
    template: PreparedTemplate, query_id: str, query_token_count: int
):
    """Convert full polymer indices to Chai indices; return None if all are unresolved."""
    import torch

    from chai_lab.data import residue_constants as rc
    from chai_lab.data.dataset.templates.load import LoadedTemplate
    from chai_lab.data.parsing.templates.template_hit import TemplateHit

    if template.mmcif_path is not None and not template.mmcif_path.is_absolute():
        raise PreparedTemplateError(
            "Template paths must be resolved before inference; "
            "call validate_resources(manifest_path) first"
        )
    mmcif = (
        template.mmcif
        if template.mmcif_path is None
        else read_text_auto(template.mmcif_path)
    )
    if mmcif is None:
        raise PreparedTemplateError("Prepared template has no mmCIF content")
    structure_context, full_positions, full_length = (
        _load_template_structure_context_with_indices(mmcif)
    )
    if template.query_indices[-1] >= query_token_count:
        raise PreparedTemplateError(
            "Template queryIndices exceed the Chai query token count"
        )
    template_restype = structure_context.token_residue_type.squeeze(0)
    if template.template_indices[-1] >= full_length:
        raise PreparedTemplateError(
            "Template templateIndices exceed the full saved polymer sequence"
        )

    full_to_filtered = {
        full_index: filtered_index
        for filtered_index, full_index in enumerate(full_positions)
    }
    pairs = tuple(zip(template.query_indices, template.template_indices, strict=True))
    retained = tuple((q, t) for q, t in pairs if t in full_to_filtered)
    discarded = tuple((q, t) for q, t in pairs if t not in full_to_filtered)
    if discarded:
        logger.warning(
            "[%s] Discarding unresolved template mappings (query, full-template): %s; "
            "retained mappings: %s",
            query_id, discarded, retained or "none",
        )
    if not retained:
        logger.warning("[%s] Discarding template: no resolved mapped residues remain", query_id)
        return None
    query_indices = tuple(q for q, _ in retained)
    template_indices = tuple(full_to_filtered[t] for _, t in retained)

    gap = rc.residue_types_with_nucleotides_order["-"]
    hit_tokens = torch.full((query_indices[-1] + 1,), gap, dtype=torch.int32)
    deletion_matrix = torch.zeros_like(hit_tokens, dtype=torch.uint8)
    previous_template_index: int | None = None
    for query_index, template_index in zip(
        query_indices, template_indices, strict=True
    ):
        hit_tokens[query_index] = template_restype[template_index]
        if previous_template_index is not None:
            deletion_count = template_index - previous_template_index - 1
            if deletion_count > 255:
                raise PreparedTemplateError(
                    "Prepared template mapping contains an insertion larger than 255 residues"
                )
            deletion_matrix[query_index] = deletion_count
        previous_template_index = template_index

    query_seq_realigned = "".join(
        rc.residue_types_with_nucleotides[index] for index in hit_tokens.tolist()
    )
    hit = TemplateHit(
        query_pdb_id=query_id,
        query_sequence="X" * query_token_count,
        index=0,
        pdb_id="prepared",
        chain_id="prepared",
        hit_start=template_indices[0],
        hit_end=template_indices[-1] + 1,
        hit_tokens=hit_tokens,
        deletion_matrix=deletion_matrix,
        query_seq_realigned=query_seq_realigned,
    )
    return LoadedTemplate(
        query_crop_indices=torch.arange(query_token_count),
        template_hit=hit,
        template_hit_structure_context=structure_context,
    )


def get_prepared_template_context(chains: list, prepared: PreparedInput):
    """Build Chai's TemplateContext directly from prepared mappings and structures."""
    import torch

    from chai_lab.data.dataset.templates.context import TemplateContext
    from chai_lab.data.parsing.structure.entity_type import EntityType

    by_id = {
        entity_id: entity
        for entity in prepared.sequences
        if entity.kind == "protein"
        for entity_id in entity.ids
    }
    contexts: list[TemplateContext] = []
    for chain in chains:
        structure_context = chain.structure_context
        _, token_residue_index_zeroed = torch.unique(
            structure_context.token_residue_index,
            return_inverse=True,
        )
        if chain.entity_data.entity_type != EntityType.PROTEIN:
            loaded = []
        else:
            entity = by_id[chain.entity_data.entity_name]
            if entity.templates is None:
                raise PreparedTemplateError(
                    f"Protein {entity.ids[0]!r} has not completed template preparation"
                )
            loaded = []
            retained_templates = []
            discarded_templates = []
            for index, template in enumerate(entity.templates):
                candidate = _as_loaded_template(
                    template,
                    query_id=entity.ids[0],
                    query_token_count=structure_context.num_tokens,
                )
                if candidate is None:
                    discarded_templates.append(f"template_{index}")
                else:
                    loaded.append(candidate)
                    retained_templates.append(f"template_{index}")
            if discarded_templates:
                logger.warning(
                    "[%s] Retained input templates: %s; discarded input templates: %s",
                    entity.ids[0], ", ".join(retained_templates) or "none",
                    ", ".join(discarded_templates),
                )
        if loaded:
            context = TemplateContext.from_loaded_templates(
                n_tokens=structure_context.num_tokens,
                templates=loaded,
                pad_to_n_templates=0,
                apply_crop=False,
            )
        else:
            context = TemplateContext.empty(
                n_templates=1, n_tokens=structure_context.num_tokens
            )
        contexts.append(context.index_select(token_residue_index_zeroed))

    for chain, context in zip(chains, contexts, strict=True):
        if chain.num_tokens != context.num_tokens:
            raise PreparedTemplateError("Prepared template token count mismatch")
    return TemplateContext.merge(contexts)
