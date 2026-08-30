# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from chai_lab.data.io.compression import read_text_auto, write_zstd_text
from chai_lab.data.io.prepared_input import load_prepared_input
from chai_lab.data.io.prepared_msas import prepare_data_bundle
from chai_lab.data.io.prepared_resources import (
    PreparedResourceError,
    get_prepared_inference_resources,
)
from chai_lab.data.parsing.msas.sequence_hash import hash_sequence

M8_ENTITY = "A\t1abc_X\t100\t4\t0\t0\t1\t4\t1\t4\t1e-5\t20\tuser\n"
M8_SERVER = "101\t1abc_X\t100\t4\t0\t0\t1\t4\t1\t4\t1e-5\t20\tserver\n"
PAIRED = ">101\nAAAA\n>paired\nAA-A\n"
UNPAIRED = ">101\nAAAA\n>single\nA-AA\n"
RESTRAINT_HEADER = (
    "restraint_id,chainA,res_idxA,chainB,res_idxB,max_distance_angstrom,"
    "min_distance_angstrom,connection_type,confidence,comment\n"
)


def _manifest(
    *,
    templates: dict | None = None,
    constraint_path: str | None = None,
    entity_ids_as_cif_chains: bool = True,
) -> dict:
    return {
        "version": 1,
        "name": "seq",
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": "AAAA",
                    "pairedMsa": PAIRED,
                    "unpairedMsa": UNPAIRED,
                }
            },
            {"protein": {"id": ["B"], "sequence": "CCCC"}},
        ],
        "use_esm_embeddings": True,
        "entity_ids_as_cif_chains": entity_ids_as_cif_chains,
        "templates": templates,
        "constraint_path": constraint_path,
    }


def _write_request(root: Path, manifest: dict) -> Path:
    path = root / "request.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class PreparedTemplateWorkflowTest(unittest.TestCase):
    def test_declared_template_bundle_is_self_contained(self):
        for encoding in ("plain", "gzip", "zstd"):
            with self.subTest(encoding=encoding), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                template_dir = root / "source-cifs"
                template_dir.mkdir()
                cif_text = "data_1ABC\n#\n"
                if encoding == "plain":
                    (template_dir / "1abc.cif").write_text(cif_text, encoding="utf-8")
                elif encoding == "gzip":
                    (template_dir / "1ABC.mmcif.gz").write_bytes(
                        gzip.compress(cif_text.encode())
                    )
                else:
                    write_zstd_text(template_dir / "1ABC.cif.zst", cif_text)
                (root / "hits.m8").write_text(M8_ENTITY, encoding="utf-8")
                request = _write_request(
                    root,
                    _manifest(
                        templates={
                            "hits_path": "hits.m8",
                            "cif_directory": "source-cifs",
                            "query_id_mode": "entity_name",
                        }
                    ),
                )
                output = root / "result/seq/seq_data.json"

                prepare_data_bundle(request, output, use_msa_server=False)

                data = json.loads(output.read_text())
                self.assertEqual(
                    data["templates"],
                    {
                        "hits_path": "templates/all_chain_templates.m8",
                        "cif_directory": "templates/cifs",
                        "query_id_mode": "entity_name",
                    },
                )
                self.assertEqual(
                    (output.parent / "templates/all_chain_templates.m8").read_text(),
                    M8_ENTITY,
                )
                canonical_cif = output.parent / "templates/cifs/1ABC.cif.gz"
                self.assertEqual(read_text_auto(canonical_cif), cif_text)

    def test_server_template_ids_are_remapped_to_runtime_sequence_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = _write_request(root, _manifest())
            output = root / "result/seq/seq_data.json"
            observed: dict[str, object] = {}

            def fake_searcher(queries, work_dir, url, search_templates):
                observed["queries"] = queries
                observed["search_templates"] = search_templates
                hits = work_dir / "hits.m8"
                hits.write_text(M8_SERVER, encoding="utf-8")
                return [
                    SimpleNamespace(paired=PAIRED, unpaired=UNPAIRED),
                    SimpleNamespace(paired=">102\nCCCC\n", unpaired=">102\nCCCC\n"),
                ], hits

            def fake_downloader(pdb_id: str, directory: Path) -> Path:
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{pdb_id}.cif.gz"
                path.write_bytes(gzip.compress(b"data_1ABC\n#\n"))
                return path

            prepare_data_bundle(
                request,
                output,
                use_msa_server=False,
                use_templates_server=True,
                searcher=fake_searcher,
                template_downloader=fake_downloader,
            )

            self.assertEqual(observed["queries"], ["AAAA", "CCCC"])
            self.assertTrue(observed["search_templates"])
            data = json.loads(output.read_text())
            self.assertEqual(data["templates"]["query_id_mode"], "sequence_hash")
            m8 = (output.parent / data["templates"]["hits_path"]).read_text()
            self.assertTrue(m8.startswith(hash_sequence("AAAA") + "\t"))
            protein_b = data["sequences"][1]["protein"]
            self.assertIsNone(protein_b["pairedMsa"])
            self.assertIsNone(protein_b["unpairedMsa"])

    def test_declared_and_server_templates_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hits.m8").write_text(M8_ENTITY, encoding="utf-8")
            (root / "cifs").mkdir()
            request = _write_request(
                root,
                _manifest(
                    templates={
                        "hits_path": "hits.m8",
                        "cif_directory": "cifs",
                        "query_id_mode": "entity_name",
                    }
                ),
            )
            with self.assertRaisesRegex(ValueError, "Cannot combine"):
                prepare_data_bundle(
                    request,
                    root / "result/seq/seq_data.json",
                    use_msa_server=False,
                    use_templates_server=True,
                    searcher=lambda *args: self.fail("Search must not start"),
                )

    def test_query_id_mode_is_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hits.m8").write_text(
                M8_ENTITY.replace("A\t", "unknown\t", 1), encoding="utf-8"
            )
            (root / "cifs").mkdir()
            request = _write_request(
                root,
                _manifest(
                    templates={
                        "hits_path": "hits.m8",
                        "cif_directory": "cifs",
                        "query_id_mode": "entity_name",
                    }
                ),
            )
            with self.assertRaisesRegex(PreparedResourceError, "query IDs"):
                prepare_data_bundle(
                    request,
                    root / "result/seq/seq_data.json",
                    use_msa_server=False,
                )

    def test_missing_local_cif_uses_downloader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hits.m8").write_text(M8_ENTITY, encoding="utf-8")
            (root / "cifs").mkdir()
            requested: list[str] = []

            def fake_downloader(pdb_id: str, directory: Path) -> Path:
                requested.append(pdb_id)
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{pdb_id}.cif.gz"
                path.write_bytes(gzip.compress(b"data_downloaded\n#\n"))
                return path

            request = _write_request(
                root,
                _manifest(
                    templates={
                        "hits_path": "hits.m8",
                        "cif_directory": "cifs",
                        "query_id_mode": "entity_name",
                    }
                ),
            )
            output = root / "result/seq/seq_data.json"
            prepare_data_bundle(
                request,
                output,
                use_msa_server=False,
                template_downloader=fake_downloader,
            )
            self.assertEqual(requested, ["1ABC"])
            self.assertTrue((output.parent / "templates/cifs/1ABC.cif.gz").is_file())


class PreparedRestraintWorkflowTest(unittest.TestCase):
    def test_restraint_is_validated_and_materialized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            restraints = root / "restraints.csv"
            restraints.write_text(
                RESTRAINT_HEADER + "r1,A,A1,B,C1,8.0,0.0,contact,0.8,test\n",
                encoding="utf-8",
            )
            request = _write_request(root, _manifest(constraint_path=restraints.name))
            output = root / "result/seq/seq_data.json"
            prepare_data_bundle(request, output, use_msa_server=False)

            data = json.loads(output.read_text())
            self.assertEqual(data["constraint_path"], "constraints/seq.restraints.csv")
            self.assertEqual(
                (output.parent / data["constraint_path"]).read_text(),
                restraints.read_text(),
            )

    def test_restraint_chain_convention_is_validated(self):
        for use_entity_ids, bad_chain in ((True, "C"), (False, "target")):
            with (
                self.subTest(use_entity_ids=use_entity_ids),
                tempfile.TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                restraints = root / "restraints.csv"
                restraints.write_text(
                    RESTRAINT_HEADER
                    + f"r1,{bad_chain},A1,B,C1,8.0,0.0,contact,1.0,test\n",
                    encoding="utf-8",
                )
                request = _write_request(
                    root,
                    _manifest(
                        constraint_path=restraints.name,
                        entity_ids_as_cif_chains=use_entity_ids,
                    ),
                )
                with self.assertRaisesRegex(
                    PreparedResourceError, "Restraint chain IDs"
                ):
                    prepare_data_bundle(
                        request,
                        root / "result/seq/seq_data.json",
                        use_msa_server=False,
                    )


class PreparedInferenceResourcesTest(unittest.TestCase):
    def test_resolved_bundle_maps_to_native_inference_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "hits.m8").write_text(M8_ENTITY, encoding="utf-8")
            (root / "cifs").mkdir()
            (root / "restraints.csv").write_text(
                RESTRAINT_HEADER + "r1,A,A1,B,C1,8.0,0.0,contact,1.0,test\n",
                encoding="utf-8",
            )
            request = _write_request(
                root,
                _manifest(
                    templates={
                        "hits_path": "hits.m8",
                        "cif_directory": "cifs",
                        "query_id_mode": "sequence_hash",
                    },
                    constraint_path="restraints.csv",
                ),
            )
            unresolved = load_prepared_input(request)
            with self.assertRaisesRegex(PreparedResourceError, "must be resolved"):
                get_prepared_inference_resources(unresolved)

            resolved = unresolved.validate_resources(request)
            resources = get_prepared_inference_resources(resolved)
            self.assertEqual(resources.template_hits_path, root.resolve() / "hits.m8")
            self.assertEqual(resources.template_cif_directory, root.resolve() / "cifs")
            self.assertTrue(resources.use_sequence_hash_for_template_lookup)
            self.assertEqual(
                resources.constraint_path, root.resolve() / "restraints.csv"
            )


if __name__ == "__main__":
    unittest.main()
