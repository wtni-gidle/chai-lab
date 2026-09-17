# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from chai_lab.data.io.compression import read_text_auto, write_zstd_text
from chai_lab.data.io.prepared_input import PreparedTemplate
from chai_lab.data.io.prepared_msas import prepare_data_bundle
from chai_lab.data.io.prepared_templates import (
    _as_loaded_template,
    _load_template_structure_context,
    _matches_native_template_features,
    _single_chain_mmcif,
    parse_max_template_date,
    prepared_template_from_loaded,
)
from chai_lab.data.parsing.templates.m8 import (
    get_mmcif_release_date,
    parse_m8_to_template_hits,
)

PAIRED = ">101\nAAAA\n>paired\nAA-A\n"
UNPAIRED = ">101\nAAAA\n>single\nA-AA\n"
MMCIF_A = "data_template_A\n#\n"
MMCIF_B = "data_template_B\n#\n"


def _manifest(
    *,
    templates: list[dict] | None = None,
) -> dict:
    return {
        "version": 1,
        "name": "seq",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "AAAA",
                    "pairedMsa": PAIRED,
                    "unpairedMsa": UNPAIRED,
                    "templates": templates,
                }
            }
        ],
    }


def _write_request(root: Path, manifest: dict) -> Path:
    path = root / "request.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class PreparedTemplateWorkflowTest(unittest.TestCase):
    def test_server_m8_is_ephemeral_and_final_mapping_is_af3_style(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _write_request(root, _manifest())
            output = root / "result/seq/seq_data.json"
            observed: dict[str, object] = {}

            def fake_searcher(queries, work_dir, url, search_templates):
                observed["queries"] = queries
                observed["search_templates"] = search_templates
                m8_path = work_dir / "all_chain_templates.m8"
                m8_path.write_text("101\t1abc_A\t...\n", encoding="utf-8")
                return [
                    SimpleNamespace(paired=PAIRED, unpaired=UNPAIRED),
                    SimpleNamespace(paired=PAIRED, unpaired=UNPAIRED),
                ], m8_path

            def fake_template_parser(
                query_id, query, m8_path, cif_cache, max_template_date
            ):
                observed["query_id"] = query_id
                observed["query"] = query
                observed["m8_exists_during_parse"] = m8_path.is_file()
                observed["cif_cache"] = cif_cache
                observed["max_template_date"] = max_template_date
                return (
                    PreparedTemplate(
                        mmcif=MMCIF_A,
                        mmcif_path=None,
                        query_indices=(0, 2, 3),
                        template_indices=(1, 2, 4),
                    ),
                )

            prepare_data_bundle(
                request,
                output,
                use_msa_server=False,
                use_templates_server=True,
                max_template_date="2021-09-30",
                searcher=fake_searcher,
                template_parser=fake_template_parser,
            )

            self.assertEqual(observed["queries"], ["AAAA", "AAAA"])
            self.assertTrue(observed["search_templates"])
            self.assertEqual(observed["query_id"], "101")
            self.assertEqual(observed["query"], "AAAA")
            self.assertTrue(observed["m8_exists_during_parse"])
            self.assertEqual(observed["max_template_date"], date(2021, 9, 30))

            data = json.loads(output.read_text())
            protein = data["sequences"][0]["protein"]
            self.assertNotIn("templates", data)
            self.assertNotIn("max_template_date", data)
            self.assertEqual(
                protein["templates"],
                [
                    {
                        "mmcifPath": "msas/seq__A_template_0.cif.zst",
                        "queryIndices": [0, 2, 3],
                        "templateIndices": [1, 2, 4],
                    }
                ],
            )
            self.assertEqual(
                read_text_auto(output.parent / protein["templates"][0]["mmcifPath"]),
                MMCIF_A,
            )
            self.assertEqual(list(output.parent.rglob("*.m8")), [])
            self.assertFalse((output.parent / "templates").exists())

    def test_inline_and_path_templates_are_externalized_without_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = write_zstd_text(root / "source.cif", MMCIF_B)
            request = _write_request(
                root,
                _manifest(
                    templates=[
                        {
                            "mmcif": MMCIF_A,
                            "queryIndices": [0, 1],
                            "templateIndices": [1, 2],
                        },
                        {
                            "mmcifPath": source_path.name,
                            "queryIndices": [2, 3],
                            "templateIndices": [3, 4],
                        },
                    ]
                ),
            )
            output = root / "result/seq/seq_data.json"

            def unexpected_search(*args, **kwargs):
                self.fail("Fully declared templates must not trigger a server search")

            prepare_data_bundle(
                request,
                output,
                use_msa_server=False,
                use_templates_server=True,
                searcher=unexpected_search,
            )

            templates = json.loads(output.read_text())["sequences"][0]["protein"][
                "templates"
            ]
            self.assertEqual(len(templates), 2)
            for index, expected in enumerate((MMCIF_A, MMCIF_B)):
                self.assertEqual(
                    templates[index]["mmcifPath"],
                    f"msas/seq__A_template_{index}.cif.zst",
                )
                self.assertEqual(
                    read_text_auto(output.parent / templates[index]["mmcifPath"]),
                    expected,
                )

    def test_missing_templates_become_explicit_empty_list_without_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _write_request(root, _manifest())
            output = root / "result/seq/seq_data.json"
            prepare_data_bundle(request, output, use_msa_server=False)
            protein = json.loads(output.read_text())["sequences"][0]["protein"]
            self.assertEqual(protein["templates"], [])

    def test_template_query_ids_follow_unique_entities_not_expanded_homomer_chains(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = _manifest()
            manifest["sequences"].append(
                {
                    "protein": {
                        "id": ["C"],
                        "sequence": "CCCC",
                        "pairedMsa": ">102\nCCCC\n",
                        "unpairedMsa": ">102\nCCCC\n",
                        "templates": None,
                    }
                }
            )
            request = _write_request(root, manifest)
            output = root / "result/seq/seq_data.json"
            observed_query_ids: list[str] = []

            def fake_searcher(queries, work_dir, url, search_templates):
                m8_path = work_dir / "all_chain_templates.m8"
                m8_path.write_text("", encoding="utf-8")
                return [
                    SimpleNamespace(paired="", unpaired="") for _ in queries
                ], m8_path

            def fake_template_parser(
                query_id, query, m8_path, cif_cache, max_template_date
            ):
                observed_query_ids.append(query_id)
                return ()

            prepare_data_bundle(
                request,
                output,
                use_msa_server=False,
                use_templates_server=True,
                searcher=fake_searcher,
                template_parser=fake_template_parser,
            )

            self.assertEqual(observed_query_ids, ["101", "102"])


class TemplateDateCutoffTest(unittest.TestCase):
    def test_release_date_is_the_earliest_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            cif_path = Path(temporary) / "template.cif"
            cif_path.write_text(
                """data_template
loop_
_pdbx_audit_revision_history.ordinal
_pdbx_audit_revision_history.revision_date
1 2021-09-30
2 2023-01-01
#
""",
                encoding="utf-8",
            )
            self.assertEqual(get_mmcif_release_date(cif_path), date(2021, 9, 30))

    def test_invalid_cutoff_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
            parse_max_template_date("2021/09/30")

    def test_newer_hit_is_skipped_before_kalign_and_next_hit_is_considered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m8 = root / "hits.m8"
            rows = [
                "101\t1new_A\t90\t4\t0\t0\t1\t4\t1\t4\t1e-10\t100\tnew",
                "101\t1old_A\t90\t4\t0\t0\t1\t4\t1\t4\t1e-9\t90\told",
            ]
            m8.write_text("\n".join(rows) + "\n", encoding="utf-8")

            class FakePolymer:
                @staticmethod
                def make_one_letter_sequence():
                    return "AAAA"

            class FakeChain:
                @staticmethod
                def get_polymer():
                    return FakePolymer()

            class FakeModel:
                @staticmethod
                def __getitem__(chain_id):
                    return FakeChain()

            class FakeStructure:
                @staticmethod
                def __getitem__(model_index):
                    return FakeModel()

            release_dates = {
                "1new.cif.gz": date(2021, 10, 1),
                "1old.cif.gz": date(2021, 9, 30),
            }

            def fake_download(pdb_id, directory):
                return root / f"{pdb_id.lower()}.cif.gz"

            def fake_release_date(path):
                return release_dates[path.name]

            with (
                patch(
                    "chai_lab.data.parsing.templates.m8.download_cif_file",
                    side_effect=fake_download,
                ),
                patch(
                    "chai_lab.data.parsing.templates.m8.get_mmcif_release_date",
                    side_effect=fake_release_date,
                ),
                patch(
                    "chai_lab.data.parsing.templates.m8.gemmi.read_structure",
                    return_value=FakeStructure(),
                ),
                patch(
                    "chai_lab.data.parsing.templates.m8.kalign_query_to_reference",
                    return_value=None,
                ) as kalign,
            ):
                self.assertEqual(
                    list(
                        parse_m8_to_template_hits(
                            "101",
                            "AAAA",
                            m8,
                            template_cif_folder=root / "cache",
                            max_template_date=date(2021, 9, 30),
                        )
                    ),
                    [],
                )

            self.assertEqual(kalign.call_count, 1)

    def test_unknown_release_date_is_skipped_before_kalign(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            m8 = root / "hits.m8"
            m8.write_text(
                "101\t1unk_A\t90\t4\t0\t0\t1\t4\t1\t4\t1e-10\t100\tunknown\n",
                encoding="utf-8",
            )
            with (
                patch(
                    "chai_lab.data.parsing.templates.m8.download_cif_file",
                    return_value=root / "1unk.cif.gz",
                ),
                patch(
                    "chai_lab.data.parsing.templates.m8.get_mmcif_release_date",
                    return_value=None,
                ),
                patch(
                    "chai_lab.data.parsing.templates.m8.kalign_query_to_reference"
                ) as kalign,
            ):
                self.assertEqual(
                    list(
                        parse_m8_to_template_hits(
                            "101",
                            "AAAA",
                            m8,
                            max_template_date=date(2099, 1, 1),
                        )
                    ),
                    [],
                )
            kalign.assert_not_called()


class PreparedTemplateFeatureRoundTripTest(unittest.TestCase):
    @staticmethod
    def _protein_mmcif() -> str:
        import gemmi

        structure = gemmi.Structure()
        structure.name = "template"
        model = gemmi.Model("1")
        chain = gemmi.Chain("A")
        for residue_index, residue_name in enumerate(
            ("ALA", "CYS", "ASP", "GLU"), start=1
        ):
            residue = gemmi.Residue()
            residue.name = residue_name
            residue.seqid = gemmi.SeqId(residue_index, " ")
            for atom_offset, (atom_name, element) in enumerate(
                (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"))
            ):
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.element = gemmi.Element(element)
                atom.pos = gemmi.Position(
                    residue_index * 4.0 + atom_offset * 0.2,
                    atom_offset * 0.1,
                    0.0,
                )
                residue.add_atom(atom)
            chain.add_residue(residue)
        model.add_chain(chain)
        structure.add_model(model)
        structure.setup_entities()
        structure.entities[0].full_sequence = ["ALA", "CYS", "ASP", "GLU"]
        structure.assign_subchains()
        structure.assign_label_seq_id()
        return structure.make_mmcif_document().as_string()

    def test_single_chain_mmcif_can_be_read_by_chai_after_externalization(self):
        import gemmi

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.cif"
            structure = gemmi.read_structure_string(self._protein_mmcif())
            water = gemmi.Residue()
            water.name = "HOH"
            water.seqid = gemmi.SeqId(10, " ")
            water.het_flag = "H"
            oxygen = gemmi.Atom()
            oxygen.name = "O"
            oxygen.element = gemmi.Element("O")
            water.add_atom(oxygen)
            structure[0][0].add_residue(water)
            source.write_text(
                structure.make_mmcif_document().as_string(), encoding="utf-8"
            )
            extracted = _single_chain_mmcif(source, "A")
            context = _load_template_structure_context(extracted)
            self.assertEqual(context.num_tokens, 4)

    def test_single_chain_mmcif_preserves_unresolved_entity_residues(self):
        import gemmi

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source_with_gap.cif"
            structure = gemmi.read_structure_string(self._protein_mmcif())
            structure.entities[0].full_sequence = [
                "ALA",
                "CYS",
                "ASP",
                "GLY",
                "GLU",
            ]
            structure.assign_label_seq_id()
            source.write_text(
                structure.make_mmcif_document().as_string(), encoding="utf-8"
            )

            extracted = _single_chain_mmcif(source, "A")
            round_tripped = gemmi.read_structure_string(extracted)
            self.assertEqual(
                list(round_tripped.entities[0].full_sequence),
                ["ALA", "CYS", "ASP", "GLY", "GLU"],
            )

    def test_saved_structure_and_mapping_rebuild_native_template_features(self):
        import torch

        from chai_lab.data import residue_constants as rc
        from chai_lab.data.dataset.templates.context import TemplateContext
        from chai_lab.data.dataset.templates.load import LoadedTemplate
        from chai_lab.data.parsing.templates.template_hit import TemplateHit

        mmcif = self._protein_mmcif()
        structure_context = _load_template_structure_context(mmcif)
        restype = structure_context.token_residue_type.squeeze(0)
        gap = rc.residue_types_with_nucleotides_order["-"]
        native_hit = TemplateHit(
            query_pdb_id="A",
            query_sequence="ACDE",
            index=0,
            pdb_id="template",
            chain_id="A",
            hit_start=0,
            hit_end=4,
            hit_tokens=torch.tensor(
                [restype[0], restype[2], gap, restype[3]], dtype=torch.int32
            ),
            deletion_matrix=torch.tensor([0, 1, 0, 0], dtype=torch.uint8),
            query_seq_realigned="AD-E",
        )
        native_loaded = LoadedTemplate(
            query_crop_indices=torch.arange(4),
            template_hit=native_hit,
            template_hit_structure_context=structure_context,
        )

        prepared = prepared_template_from_loaded(native_loaded, mmcif)
        self.assertEqual(prepared.query_indices, (0, 1, 3))
        self.assertEqual(prepared.template_indices, (0, 2, 3))
        rebuilt_loaded = _as_loaded_template(prepared, "A", 4)
        self.assertTrue(_matches_native_template_features(native_loaded, prepared))

        invalid = PreparedTemplate(
            mmcif=mmcif,
            mmcif_path=None,
            query_indices=(0, 1, 3),
            template_indices=(0, 2, 4),
        )
        self.assertFalse(_matches_native_template_features(native_loaded, invalid))

        native_context = TemplateContext.from_loaded_templates(
            n_tokens=4, templates=[native_loaded]
        )
        rebuilt_context = TemplateContext.from_loaded_templates(
            n_tokens=4, templates=[rebuilt_loaded]
        )
        for field in (
            "template_restype",
            "template_pseudo_beta_mask",
            "template_backbone_frame_mask",
            "template_distances",
            "template_unit_vector",
        ):
            torch.testing.assert_close(
                getattr(native_context, field), getattr(rebuilt_context, field)
            )


if __name__ == "__main__":
    unittest.main()
