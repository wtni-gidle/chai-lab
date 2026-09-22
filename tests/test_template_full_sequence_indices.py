"""Public template positions must survive Chai's native residue filtering."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from chai_lab.data.dataset.templates.context import TemplateContext
from chai_lab.data.dataset.templates.load import LoadedTemplate
from chai_lab.data.io.prepared_input import PreparedEntity, PreparedInput, PreparedTemplate
from chai_lab.data.io.prepared_templates import (
    _as_loaded_template,
    _load_template_structure_context,
    _single_chain_mmcif,
    get_prepared_template_context,
    prepared_template_from_loaded,
)
from chai_lab.data.parsing.structure.entity_type import EntityType
from chai_lab.data.parsing.templates.template_hit import TemplateHit
from tests import test_prepared_templates as fixtures


def template_cif(full_sequence, observed_positions, *, missing_ca=None):
    import gemmi

    structure = gemmi.read_structure_string(
        fixtures.PreparedTemplateFeatureRoundTripTest._protein_mmcif()
    )
    structure.entities[0].full_sequence = list(full_sequence)
    for residue, position in zip(structure[0][0], observed_positions, strict=True):
        residue.label_seq = position + 1
        if position == missing_ca:
            residue.remove_atom("CA", "\x00")
    return structure.make_mmcif_document().as_string()


def native_template(mmcif):
    # Construct a native hit over the observed ACDE sequence, independently of
    # the public serializer. The real tokenizer and native filter still run.
    context = _load_template_structure_context(mmcif)
    hit = TemplateHit(
        query_pdb_id="A", query_sequence="ACDE", index=0,
        pdb_id="fixture", chain_id="A", hit_start=0, hit_end=4,
        hit_tokens=context.token_residue_type.squeeze(0).to(torch.int32),
        deletion_matrix=torch.zeros(4, dtype=torch.uint8),
        query_seq_realigned="ACDE",
    )
    return LoadedTemplate(torch.arange(4), hit, context)


@pytest.mark.parametrize("full_sequence,positions", [
    (("ALA", "CYS", "ASP", "GLU"), (0, 1, 2, 3)),
    (("ALA", "CYS", "ASP", "GLY", "GLU"), (0, 1, 2, 4)),
    (("GLY", "ALA", "CYS", "ASP", "GLU", "GLY"), (1, 2, 3, 4)),
    (("ALA", "GLY", "GLY", "CYS", "ASP", "GLU"), (0, 3, 4, 5)),
    (("ALA", "CYS", "CYS", "ASP", "GLU"), (0, 2, 3, 4)),
])
def test_native_export_uses_full_positions_and_preserves_features(full_sequence, positions, tmp_path):
    mmcif = template_cif(full_sequence, positions)
    native = native_template(mmcif)
    source = tmp_path / "source.cif"
    source.write_text(mmcif)
    public = prepared_template_from_loaded(native, _single_chain_mmcif(source, "A"))
    assert public.query_indices == (0, 1, 2, 3)
    assert public.template_indices == positions
    rebuilt = _as_loaded_template(public, "A", 4)
    before = TemplateContext.from_loaded_templates(n_tokens=4, templates=[native])
    after = TemplateContext.from_loaded_templates(n_tokens=4, templates=[rebuilt])
    for field in ("template_restype", "template_pseudo_beta_mask",
                  "template_backbone_frame_mask", "template_distances", "template_unit_vector"):
        torch.testing.assert_close(getattr(before, field), getattr(after, field), rtol=0, atol=0)


def test_user_full_indices_map_to_correct_residues_without_realigning():
    mmcif = template_cif(("ALA", "CYS", "ASP", "GLY", "GLU"), (0, 1, 2, 4))
    public = PreparedTemplate(mmcif, None, (0, 1, 2, 4), (0, 1, 2, 4))
    loaded = _as_loaded_template(public, "A", 5)
    valid = loaded.template_hit.hit_valid_mask
    assert loaded.template_hit_indices[valid].tolist() == [0, 1, 2, 3]
    assert loaded.template_query_match_indices[valid].tolist() == [0, 1, 2, 4]
    assert loaded.template_hit.query_seq_realigned == "ACD-E"


@pytest.mark.parametrize("missing_ca", [None, 3])
def test_missing_mapping_is_removed_and_reported(caplog, missing_ca):
    # Either G is absent entirely, or C exists but has no native centre atom.
    if missing_ca is None:
        mmcif = template_cif(("ALA", "CYS", "ASP", "GLY", "GLU"), (0, 1, 2, 4))
        full_indices, wanted_query, wanted_internal = (0, 1, 2, 3, 4), [0, 1, 2, 4], [0, 1, 2, 3]
    else:
        mmcif = template_cif(("ALA", "GLY", "GLY", "CYS", "ASP", "GLU"), (0, 3, 4, 5), missing_ca=3)
        full_indices, wanted_query, wanted_internal = (0, 3, 4, 5), [0, 2, 3], [0, 1, 2]
    public = PreparedTemplate(mmcif, None, tuple(range(len(full_indices))), full_indices)
    loaded = _as_loaded_template(public, "A", len(full_indices))
    valid = loaded.template_hit.hit_valid_mask
    assert loaded.template_query_match_indices[valid].tolist() == wanted_query
    assert loaded.template_hit_indices[valid].tolist() == wanted_internal
    assert "unresolved" in caplog.text.lower()
    assert "retained" in caplog.text.lower()
    # A native hit with gaps at the discarded query positions must produce the
    # same consumer features, including when atoms exist but the CA is missing.
    from chai_lab.data import residue_constants as rc

    aligned = "ACD-E" if missing_ca is None else "A-DE"
    native_hit = TemplateHit(
        query_pdb_id="A", query_sequence="ACDGE" if missing_ca is None else "ACDE",
        index=0, pdb_id="fixture", chain_id="A", hit_start=0,
        hit_end=len(wanted_internal),
        hit_tokens=torch.tensor([rc.residue_types_with_nucleotides_order[r] for r in aligned], dtype=torch.int32),
        deletion_matrix=torch.zeros(len(aligned), dtype=torch.uint8),
        query_seq_realigned=aligned,
    )
    native = LoadedTemplate(torch.arange(len(full_indices)), native_hit,
                            _load_template_structure_context(mmcif))
    before = TemplateContext.from_loaded_templates(n_tokens=len(full_indices), templates=[native])
    after = TemplateContext.from_loaded_templates(n_tokens=len(full_indices), templates=[loaded])
    for field in ("template_restype", "template_pseudo_beta_mask",
                  "template_backbone_frame_mask", "template_distances", "template_unit_vector"):
        torch.testing.assert_close(getattr(before, field), getattr(after, field), rtol=0, atol=0)


def test_all_unresolved_template_is_dropped_but_other_template_survives(caplog):
    mmcif = template_cif(("ALA", "CYS", "ASP", "GLY", "GLU"), (0, 1, 2, 4))
    missing = PreparedTemplate(mmcif, None, (0,), (3,))
    usable = replace(missing, template_indices=(4,))
    protein = PreparedEntity(kind="protein", ids=("A",), sequence="ACDE", templates=(missing, usable))
    prepared = PreparedInput(version=1, name="test", sequences=(protein,))
    chain = SimpleNamespace(
        structure_context=_load_template_structure_context(mmcif),
        entity_data=SimpleNamespace(entity_type=EntityType.PROTEIN, entity_name="A"),
        num_tokens=4,
    )
    actual = get_prepared_template_context([chain], prepared)
    expected = get_prepared_template_context([chain], replace(prepared, sequences=(replace(protein, templates=(usable,)),)))
    assert torch.equal(actual.template_restype, expected.template_restype)
    assert torch.equal(actual.template_pseudo_beta_mask, expected.template_pseudo_beta_mask)
    assert "discard" in caplog.text.lower()
    assert "template_1" in caplog.text
    empty = get_prepared_template_context(
        [chain], replace(prepared, sequences=(replace(protein, templates=(missing,)),))
    )
    assert not empty.template_pseudo_beta_mask.any()
    assert not empty.template_backbone_frame_mask.any()


def test_out_of_range_full_index_is_an_error_not_silent_discard():
    mmcif = template_cif(("ALA", "CYS", "ASP", "GLY", "GLU"), (0, 1, 2, 4))
    with pytest.raises(ValueError, match="exceed"):
        _as_loaded_template(PreparedTemplate(mmcif, None, (0,), (5,)), "A", 4)
