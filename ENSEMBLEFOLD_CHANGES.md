# EnsembleFold wrapper changes

This branch is based on upstream commit
`66c38d1fe5c6756a89ff8596b1dea87d305ec06f` (`v0.6.1-20-g66c38d1`). It is the
development branch for an EnsembleFold-oriented Chai-1 wrapper. Model architecture,
checkpoints, feature definitions, and sampling mathematics are outside the scope of
the wrapper.

The detailed Chinese design and operating notes are maintained in
`/Users/wtni/Projects/EnsembleFold/CHAI1_WRAPPER_NOTES.md`.

## Implementation status

Stage 0 established the fork, branch, and upstream baseline. Stage 1 adds the
side-effect-free prepared-input foundation; there are still no native CLI or inference
behavior changes.

`chai_lab.data.io.prepared_input` defines a strict, versioned `_data.json` schema,
relative-path resolution, declared-resource checks, atomic JSON writing, and the
`<output>/<name>/<name>_data.json` path rule. The top-level `name` is the formal target
identity, so renaming the JSON file does not rename the target. Unknown fields and
unsafe target names are rejected instead of being silently corrected.

`chai_lab.workflow.build_workflow_plan` validates data/inference stage combinations and
calculates target, job, prepared-input, and prediction locations. In stage 1 it only
returns a plan: it does not search data, run Chai-1, or write prediction artifacts. The
existing `chai-lab fold` command still calls the native `run_inference` function.

Planned work is intentionally gated and will be implemented one stage at a time:

1. prepared JSON schema and workflow foundation (implemented in stage 1);
2. paired/unpaired A3M.zst persistence and private Parquet reconstruction;
3. template and restraint handoff;
4. multi-seed CLI and AF3-style result writer;
5. lightweight skip, concurrency hardening, and `run_chai1.sh`;
6. native/combined/split and GPU validation.

The wrapper contract will use plural `--seeds`, with one trunk execution per seed.
The wrapper will not expose `num_trunk_samples`; multiple independent trunk runs are
represented by multiple seeds. Chai-1's native Python API remains unchanged during
the early stages.

## Stage 0 baseline

- Repository state before changes: clean `main` at the exact upstream commit above.
- `python3 -m compileall -q chai_lab tests`: passed under Python 3.14.6, with two
  upstream invalid-escape `SyntaxWarning` messages.
- Full pytest collection was not run because the local base environment has neither
  PyTorch nor pytest installed. GPU/native equivalence testing belongs to the final
  validation stage rather than this metadata-only stage.
- `git diff --check`: passed.
