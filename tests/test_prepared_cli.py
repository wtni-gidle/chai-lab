# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from chai_lab.main import build_app
from chai_lab.workflow import WorkflowResult


class PreparedCLITest(unittest.TestCase):
    def test_fold_help_exposes_wrapper_stages_and_plural_seeds(self):
        result = CliRunner().invoke(build_app(), ["fold", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--run-data-pipeline", result.output)
        self.assertIn("--run-inference", result.output)
        self.assertIn("--seeds", result.output)
        self.assertIn("--diffusion-samples", result.output)
        self.assertIn("--skip", result.output)
        self.assertNotIn("num-trunk-samples", result.output)

    def test_cli_boolean_values_and_seed_string_reach_workflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = root / "seq.json"
            request.write_text("{}", encoding="utf-8")
            expected = WorkflowResult(
                prepared_path=(root / "result/seq/seq_data.json").resolve(),
                seeds=(),
                prediction_paths=(),
            )
            with patch(
                "chai_lab.main.run_prepared_workflow", return_value=expected
            ) as workflow:
                result = CliRunner().invoke(
                    build_app(),
                    [
                        "fold",
                        str(request),
                        str(root / "result"),
                        "-D",
                        "true",
                        "-P",
                        "false",
                        "-M",
                        "true",
                        "-T",
                        "false",
                        "-r",
                        "4,5",
                        "-S",
                        "true",
                    ],
                )

            self.assertEqual(result.exit_code, 0, result.output)
            kwargs = workflow.call_args.kwargs
            self.assertTrue(kwargs["run_data_pipeline"])
            self.assertFalse(kwargs["run_inference"])
            self.assertTrue(kwargs["use_msa_server"])
            self.assertFalse(kwargs["use_templates_server"])
            self.assertEqual(kwargs["seeds"], "4,5")
            self.assertTrue(kwargs["skip"])

    def test_real_data_only_cli_writes_prepared_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = root / "seq.json"
            request.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "name": "seq",
                        "sequences": [
                            {
                                "protein": {
                                    "id": ["A"],
                                    "sequence": "AAAA",
                                    "templates": None,
                                }
                            }
                        ],
                        "use_esm_embeddings": False,
                        "entity_ids_as_cif_chains": False,
                        "constraint_path": None,
                    }
                ),
                encoding="utf-8",
            )

            result = CliRunner().invoke(
                build_app(),
                [
                    "fold",
                    str(request),
                    str(root / "result"),
                    "-D",
                    "true",
                    "-P",
                    "false",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            prepared_path = root / "result/seq/seq_data.json"
            self.assertTrue(prepared_path.is_file())
            prepared = json.loads(prepared_path.read_text())
            self.assertEqual(prepared["sequences"][0]["protein"]["templates"], [])


if __name__ == "__main__":
    unittest.main()
