# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

"""AF3-style persistence for templates parsed by Chai's native pipeline."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from chai_lab.data.io.compression import read_text_auto, write_zstd_text
from chai_lab.data.io.prepared_input import (
    PreparedEntity,
    PreparedInput,
    PreparedTemplate,
)


class PreparedTemplateError(ValueError):
    """Raised when prepared template structures or mappings are invalid."""


NativeTemplateParser = Callable[[str, str, Path, Path], Sequence[PreparedTemplate]]


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

    structure = gemmi.Structure()
    structure.name = f"{source.name}_{chain_id}"
    structure.cell = source.cell
    structure.spacegroup_hm = source.spacegroup_hm
    model = gemmi.Model("1")
    chain = gemmi.Chain(chain_id)
    for residue in selected.get_polymer():
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
    structure.entities[0].full_sequence = [
        residue.name for residue in structure[0][0].get_polymer()
    ]
    structure.assign_subchains()
    structure.assign_label_seq_id()
    return structure.make_mmcif_document().as_string()


def prepared_template_from_loaded(loaded_template, mmcif: str) -> PreparedTemplate:
    """Serialize the final native LoadedTemplate mapping in AF3-style form."""
    hit = loaded_template.template_hit
    valid = hit.hit_valid_mask
    query_indices = tuple(
        int(index)
        for index in loaded_template.template_query_match_indices[valid].tolist()
    )
    template_indices = tuple(
        int(index) for index in loaded_template.template_hit_indices[valid].tolist()
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
    )
    loaded = get_template_data(
        template_hits=hits,
        query_crop_indices=torch.arange(len(query_sequence)),
        tokenizer=AllAtomResidueTokenizer(RefConformerGenerator()),
        strict_subsequence_check=True,
        drop_unresolved_from_hits=True,
    )

    prepared_templates: list[PreparedTemplate] = []
    for template in loaded:
        hit = template.template_hit
        if hit.cif_path is None:
            raise PreparedTemplateError(
                "Native template parsing did not retain a local CIF path"
            )
        prepared_templates.append(
            prepared_template_from_loaded(
                template,
                _single_chain_mmcif(hit.cif_path, hit.chain_id),
            )
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
    first_id = entity.ids[0]
    msa_directory = output_manifest.parent / "msas"
    materialized: list[PreparedTemplate] = []
    for template_index, template in enumerate(templates):
        mmcif = (
            template.mmcif
            if template.mmcif_path is None
            else read_text_auto(template.mmcif_path)
        )
        if not mmcif:
            raise PreparedTemplateError("Template mmCIF content must not be empty")
        output_path = write_zstd_text(
            msa_directory
            / f"{target_name}__{first_id}_template_{template_index}.cif.zst",
            mmcif,
        )
        materialized.append(
            replace(
                template,
                mmcif=None,
                mmcif_path=output_path.relative_to(output_manifest.parent),
            )
        )
    return tuple(materialized)


def _load_template_structure_context(mmcif: str):
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
    return context.index_select(resolved_indices)


def _as_loaded_template(
    template: PreparedTemplate, query_id: str, query_token_count: int
):
    """Reconstruct Chai's LoadedTemplate boundary from an explicit residue mapping."""
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
    structure_context = _load_template_structure_context(mmcif)
    if template.query_indices[-1] >= query_token_count:
        raise PreparedTemplateError(
            "Template queryIndices exceed the Chai query token count"
        )
    template_restype = structure_context.token_residue_type.squeeze(0)
    if template.template_indices[-1] >= template_restype.shape[0]:
        raise PreparedTemplateError(
            "Template templateIndices exceed the saved structure sequence"
        )

    gap = rc.residue_types_with_nucleotides_order["-"]
    hit_tokens = torch.full((template.query_indices[-1] + 1,), gap, dtype=torch.int32)
    deletion_matrix = torch.zeros_like(hit_tokens, dtype=torch.uint8)
    previous_template_index: int | None = None
    for query_index, template_index in zip(
        template.query_indices, template.template_indices, strict=True
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
        hit_start=template.template_indices[0],
        hit_end=template.template_indices[-1] + 1,
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
            loaded = [
                _as_loaded_template(
                    template,
                    query_id=entity.ids[0],
                    query_token_count=structure_context.num_tokens,
                )
                for template in entity.templates
            ]
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
