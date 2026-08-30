# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import gzip
import json
import lzma
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from chai_lab.data.io.compression import read_text_auto, write_zstd_text
from chai_lab.data.io.prepared_input import load_prepared_input
from chai_lab.data.io.prepared_msas import (
    build_private_msa_directory,
    prepare_msa_bundle,
)
from chai_lab.data.parsing.msas.prepared_a3m import (
    PreparedMSAError,
    colabfold_a3ms_to_dataframe,
    protein_msa_query_sequence,
)
from chai_lab.data.parsing.msas.sequence_hash import expected_basename

QUERY = ">101\nAAAA\n"
PAIRED = ">101\nAAAA\n>UniRef_pair\nAA-A\n>padding\n----\n>pair_other\nA-AA\n"
UNPAIRED = (
    ">101\nAAAA\n>UniRef_duplicate\nAA-A\n>UniRef_single\nA--A\n>deepmsa_hit\nAAA-\n"
)
PAIRED_C = ">102\nCCCC\n>UniRef_pair\nCC-C\n>padding\n----\n>pair_other\nC-CC\n"
UNPAIRED_C = ">102\nCCCC\n>UniRef_single\nC--C\n>other\nCCC-\n"


def _manifest(proteins: list[dict]) -> dict:
    return {
        "version": 1,
        "name": "seq",
        "sequences": [{"protein": protein} for protein in proteins],
        "use_esm_embeddings": True,
        "entity_ids_as_cif_chains": False,
        "templates": None,
        "constraint_path": None,
    }


class CompressionTest(unittest.TestCase):
    def test_reader_uses_magic_bytes_not_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = ">q\nAAAA\n"
            encodings = {
                "plain.gz": payload.encode(),
                "gzip.a3m": gzip.compress(payload.encode()),
                "xz.zst": lzma.compress(payload.encode()),
            }
            for filename, content in encodings.items():
                path = root / filename
                path.write_bytes(content)
                self.assertEqual(read_text_auto(path), payload)

            zstd_path = write_zstd_text(root / "zstd.txt", payload)
            self.assertEqual(zstd_path.read_bytes()[:4], b"\x28\xb5\x2f\xfd")
            self.assertEqual(read_text_auto(zstd_path), payload)


class PreparedA3MTest(unittest.TestCase):
    def test_modified_residue_query_uses_native_chai_normalization(self):
        self.assertEqual(protein_msa_query_sequence("AS(SEP)TG"), "ASSTG")

    def test_merge_matches_native_colabfold_rules(self):
        dataframe = colabfold_a3ms_to_dataframe(
            query_sequence="AAAA",
            paired_a3m=PAIRED,
            unpaired_a3m=UNPAIRED,
            unpaired_fallback_source="mgnify",
        )
        self.assertEqual(
            dataframe.to_dict(orient="list"),
            {
                "sequence": ["AAAA", "AA-A", "A-AA", "A--A", "AAA-"],
                "source_database": [
                    "query",
                    "uniref90",
                    "bfd_uniclust",
                    "uniref90",
                    "mgnify",
                ],
                "pairing_key": ["0", "1", "3", "", ""],
                "comment": ["", "", "", "", ""],
            },
        )

    def test_query_and_alignment_width_are_validated(self):
        with self.assertRaisesRegex(PreparedMSAError, "does not match"):
            colabfold_a3ms_to_dataframe(
                query_sequence="CCCC",
                paired_a3m=None,
                unpaired_a3m=QUERY,
            )
        with self.assertRaisesRegex(PreparedMSAError, "aligned width"):
            colabfold_a3ms_to_dataframe(
                query_sequence="AAAA",
                paired_a3m=None,
                unpaired_a3m=">q\nAAAA\n>bad\nAAA\n",
            )


class PreparedMSAWorkflowTest(unittest.TestCase):
    def test_split_reconstruction_matches_native_colabfold_output(self):
        from chai_lab.data.dataset.msas.colabfold import generate_colabfold_msas

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_dir = root / "native"
            native_dir.mkdir()

            def fake_mmseqs(sequences, prefix, **kwargs):
                if kwargs["use_pairing"]:
                    return [PAIRED, PAIRED_C], None
                return [UNPAIRED, UNPAIRED_C], None

            with patch(
                "chai_lab.data.dataset.msas.colabfold._run_mmseqs2",
                side_effect=fake_mmseqs,
            ):
                generate_colabfold_msas(
                    ["AAAA", "CCCC"],
                    native_dir,
                    "https://unused.test",
                )

            input_dir = root / "input"
            input_dir.mkdir()
            msa_texts = {
                "A_pair.a3m": PAIRED,
                "A_single.a3m": UNPAIRED,
                "C_pair.a3m": PAIRED_C,
                "C_single.a3m": UNPAIRED_C,
            }
            for filename, text in msa_texts.items():
                (input_dir / filename).write_text(text, encoding="utf-8")
            input_path = input_dir / "request.json"
            input_path.write_text(
                json.dumps(
                    _manifest(
                        [
                            {
                                "id": ["A"],
                                "sequence": "AAAA",
                                "paired_msa": "A_pair.a3m",
                                "unpaired_msa": "A_single.a3m",
                            },
                            {
                                "id": ["C"],
                                "sequence": "CCCC",
                                "paired_msa": "C_pair.a3m",
                                "unpaired_msa": "C_single.a3m",
                            },
                        ]
                    )
                ),
                encoding="utf-8",
            )
            output_manifest = root / "result" / "seq" / "seq_data.json"
            prepare_msa_bundle(input_path, output_manifest, use_msa_server=False)
            resolved = load_prepared_input(output_manifest).validate_resources(
                output_manifest
            )
            reconstructed_dir = build_private_msa_directory(resolved, root / "private")
            for sequence in ("AAAA", "CCCC"):
                filename = expected_basename(sequence)
                pd.testing.assert_frame_equal(
                    pd.read_parquet(native_dir / filename),
                    pd.read_parquet(reconstructed_dir / filename),
                )

    def test_bundle_and_private_parquet_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            input_dir.mkdir()
            paired_path = input_dir / "paired.not_zst"
            paired_path.write_bytes(gzip.compress(PAIRED.encode()))
            unpaired_path = input_dir / "unpaired.gz"
            unpaired_path.write_text(UNPAIRED, encoding="utf-8")
            input_path = input_dir / "request.json"
            input_path.write_text(
                json.dumps(
                    _manifest(
                        [
                            {
                                "id": ["A", "B"],
                                "sequence": "AAAA",
                                "paired_msa": paired_path.name,
                                "unpaired_msa": unpaired_path.name,
                                "unpaired_msa_fallback_source": "auto",
                            }
                        ]
                    )
                ),
                encoding="utf-8",
            )

            output_manifest = root / "result" / "seq" / "seq_data.json"
            prepare_msa_bundle(
                input_path,
                output_manifest,
                use_msa_server=False,
            )
            bundled_json = json.loads(output_manifest.read_text(encoding="utf-8"))
            protein = bundled_json["sequences"][0]["protein"]
            self.assertEqual(protein["paired_msa"], "msas/seq_A_paired.a3m.zst")
            self.assertEqual(protein["unpaired_msa"], "msas/seq_A_unpaired.a3m.zst")
            self.assertNotIn("sequence_hash", protein)
            for key in ("paired_msa", "unpaired_msa"):
                artifact = output_manifest.parent / protein[key]
                self.assertEqual(artifact.read_bytes()[:4], b"\x28\xb5\x2f\xfd")

            resolved = load_prepared_input(output_manifest).validate_resources(
                output_manifest
            )
            private_dir = build_private_msa_directory(
                resolved, root / "private-inference-msas"
            )
            parquet_path = private_dir / expected_basename("AAAA")
            self.assertTrue(parquet_path.is_file())
            dataframe = pd.read_parquet(parquet_path)
            self.assertEqual(dataframe.iloc[0]["source_database"], "query")
            self.assertFalse((output_manifest.parent / "processed").exists())

    def test_search_expands_chain_ids_and_uses_first_homomer_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "request.json"
            input_path.write_text(
                json.dumps(
                    _manifest(
                        [
                            {
                                "id": ["A", "B"],
                                "sequence": "AAAA",
                                "paired_msa": None,
                                "unpaired_msa": None,
                            },
                            {
                                "id": ["C"],
                                "sequence": "CCCC",
                                "paired_msa": None,
                                "unpaired_msa": None,
                            },
                        ]
                    )
                ),
                encoding="utf-8",
            )
            observed_queries: list[str] = []

            def fake_searcher(queries, work_dir, url, search_templates):
                observed_queries.extend(queries)
                self.assertTrue(work_dir.is_dir())
                self.assertFalse(search_templates)
                return (
                    [
                        SimpleNamespace(
                            paired=">first_A\nAAAA\n", unpaired=">first_A\nAAAA\n"
                        ),
                        SimpleNamespace(
                            paired=">second_B\nAAAA\n", unpaired=">second_B\nAAAA\n"
                        ),
                        SimpleNamespace(
                            paired=">first_C\nCCCC\n", unpaired=">first_C\nCCCC\n"
                        ),
                    ],
                    None,
                )

            output_manifest = root / "result" / "seq" / "seq_data.json"
            prepare_msa_bundle(
                input_path,
                output_manifest,
                use_msa_server=True,
                searcher=fake_searcher,
            )
            self.assertEqual(observed_queries, ["AAAA", "AAAA", "CCCC"])
            self.assertIn(
                ">first_A",
                read_text_auto(output_manifest.parent / "msas/seq_A_paired.a3m.zst"),
            )
            self.assertNotIn(
                ">second_B",
                read_text_auto(output_manifest.parent / "msas/seq_A_paired.a3m.zst"),
            )

    def test_monomer_server_result_does_not_write_empty_paired_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "request.json"
            input_path.write_text(
                json.dumps(
                    _manifest(
                        [
                            {
                                "id": ["A"],
                                "sequence": "AAAA",
                                "paired_msa": None,
                                "unpaired_msa": None,
                            }
                        ]
                    )
                ),
                encoding="utf-8",
            )

            def fake_searcher(queries, work_dir, url, search_templates):
                return [SimpleNamespace(paired="", unpaired=QUERY)], None

            output_manifest = root / "result" / "seq" / "seq_data.json"
            prepare_msa_bundle(
                input_path,
                output_manifest,
                use_msa_server=True,
                searcher=fake_searcher,
            )
            protein = json.loads(output_manifest.read_text())["sequences"][0]["protein"]
            self.assertIsNone(protein["paired_msa"])
            self.assertEqual(protein["unpaired_msa"], "msas/seq_A_unpaired.a3m.zst")

    def test_normalized_query_collision_cannot_silently_overwrite_parquet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_msa = root / "first.a3m"
            first_msa.write_text(">q\nASSTG\n>hit\nAS-TG\n", encoding="utf-8")
            second_msa = root / "second.a3m"
            second_msa.write_text(">q\nASSTG\n>hit\nA-STG\n", encoding="utf-8")
            manifest_path = root / "request.json"
            manifest_path.write_text(
                json.dumps(
                    _manifest(
                        [
                            {
                                "id": ["A"],
                                "sequence": "AS(SEP)TG",
                                "paired_msa": None,
                                "unpaired_msa": first_msa.name,
                            },
                            {
                                "id": ["B"],
                                "sequence": "ASSTG",
                                "paired_msa": None,
                                "unpaired_msa": second_msa.name,
                            },
                        ]
                    )
                ),
                encoding="utf-8",
            )
            prepared = load_prepared_input(manifest_path).validate_resources(
                manifest_path
            )
            with self.assertRaisesRegex(PreparedMSAError, "same Chai MSA query"):
                build_private_msa_directory(prepared, root / "private")


if __name__ == "__main__":
    unittest.main()
