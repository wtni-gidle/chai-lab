# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class RunChai1ScriptTest(unittest.TestCase):
    def test_boolean_spellings_are_normalized_before_gpu_selection_and_forwarding(self):
        repository = Path(__file__).resolve().parents[1]
        options = {
            "-D": "--run-data-pipeline", "-P": "--run-inference",
            "-J": "--write-input-json", "-z": "--compress-fold-input",
            "-f": "--compress-full-confidence", "-M": "--use-msa-server",
            "-T": "--use-templates-server", "-E": "--use-esm-embeddings",
            "-C": "--fasta-names-as-cif-chains", "-S": "--skip",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "chai-lab"
            executable.write_text('#!/bin/bash\nprintf "GPU=%s\\n" "$CUDA_VISIBLE_DEVICES"\nprintf "ARG=%s\\n" "$@"\n')
            executable.chmod(0o755)
            source = root / "input.json"
            source.write_text("{}")
            environment = {**os.environ, "PATH": f"{root}:{os.environ['PATH']}",
                           "CUDA_VISIBLE_DEVICES": "9"}
            command = ["bash", str(repository / "run_chai1.sh"), "-i", str(source),
                       "-o", str(root / "out"), "-d", "7"]
            for value, expected in [("True", "true"), ("YES", "true"), ("1", "true"),
                                    (" On ", "true"), ("False", "false"), ("0", "false"),
                                    ("NO", "false"), (" off ", "false")]:
                with self.subTest(value=value):
                    args = [item for flag in options for item in (flag, value)]
                    # Keep data enabled when testing false inference and other flags.
                    if expected == "false":
                        args += ["-D", "true"]
                    result = subprocess.run(command + args, env=environment,
                        text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    lines = result.stdout.splitlines()
                    self.assertIn("GPU=7" if expected == "true" else "GPU=9", lines)
                    forwarded = [line[4:] for line in lines if line.startswith("ARG=")]
                    for flag, long_option in options.items():
                        want = "true" if flag == "-D" else expected
                        self.assertEqual(forwarded[forwarded.index(long_option) + 1], want)
            for flag in options:
                for invalid in ("typo", "", " "):
                    with self.subTest(flag=flag, invalid=invalid):
                        result = subprocess.run(command + [flag, invalid], env=environment,
                            text=True, capture_output=True)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(flag, result.stderr)
                        self.assertNotIn("ARG=", result.stdout)
            result = subprocess.run(command + ["-D", "FALSE", "-P", "OFF"],
                env=environment, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("ARG=", result.stdout)

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
            constraint = root / "constraints.csv"
            constraint.write_text("test\n", encoding="utf-8")
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
                    "-E",
                    "false",
                    "-x",
                    str(constraint),
                    "-C",
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
                    "--compress-fold-input",
                    "false",
                    "--compress-full-confidence",
                    "false",
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
                    "--use-esm-embeddings",
                    "false",
                    "--fasta-names-as-cif-chains",
                    "true",
                    "--skip",
                    "true",
                    "--write-input-json",
                    "true",
                    "--seeds",
                    "7,8",
                    "--constraint-path",
                    str(constraint),
                ],
            )

            # An explicit cutoff must still reach the CLI, without introducing
            # any cutoff when -m is absent in the invocation above.
            completed = subprocess.run(
                [
                    str(repository / "run_chai1.sh"), "-i", str(request),
                    "-o", str(root / "result"), "-m", "2021-09-30",
                ],
                cwd=repository,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            args = capture_args.read_text().splitlines()
            self.assertEqual(args[args.index("--max-template-date") + 1], "2021-09-30")


if __name__ == "__main__":
    unittest.main()
