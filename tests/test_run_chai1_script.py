# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class RunChai1ScriptTest(unittest.TestCase):
    def test_script_forwards_af3_style_options_and_gpu(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            capture_args = root / "args.txt"
            capture_gpu = root / "gpu.txt"
            fake_chai = fake_bin / "chai-lab"
            fake_chai.write_text(
                "#!/bin/bash\n"
                'printf \'%s\\n\' "$@" > "$CHAI_CAPTURE_ARGS"\n'
                'printf \'%s\\n\' "$CUDA_VISIBLE_DEVICES" > "$CHAI_CAPTURE_GPU"\n',
                encoding="utf-8",
            )
            fake_chai.chmod(0o755)
            request = root / "seq.json"
            request.write_text("{}\n", encoding="utf-8")
            environment = {
                **os.environ,
                "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                "CHAI_CAPTURE_ARGS": str(capture_args),
                "CHAI_CAPTURE_GPU": str(capture_gpu),
            }

            completed = subprocess.run(
                [
                    str(repository / "run_chai1.sh"),
                    "-i",
                    str(request),
                    "-o",
                    str(root / "result"),
                    "-d",
                    "2",
                    "-D",
                    "false",
                    "-P",
                    "true",
                    "-r",
                    "7,8",
                    "-n",
                    "2",
                    "-c",
                    "4",
                    "-p",
                    "50",
                    "-M",
                    "false",
                    "-T",
                    "true",
                    "-S",
                    "true",
                ],
                cwd=repository,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(capture_gpu.read_text().strip(), "2")
            self.assertEqual(
                capture_args.read_text().splitlines(),
                [
                    "fold",
                    str(request),
                    str(root / "result"),
                    "--run-data-pipeline",
                    "false",
                    "--run-inference",
                    "true",
                    "--diffusion-samples",
                    "2",
                    "--recycling-steps",
                    "4",
                    "--sampling-steps",
                    "50",
                    "--use-msa-server",
                    "false",
                    "--use-templates-server",
                    "true",
                    "--skip",
                    "true",
                    "--seeds",
                    "7,8",
                ],
            )


if __name__ == "__main__":
    unittest.main()
