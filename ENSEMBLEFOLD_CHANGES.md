# EnsembleFold wrapper changes

This branch is based on upstream commit
`66c38d1fe5c6756a89ff8596b1dea87d305ec06f` (`v0.6.1-20-g66c38d1`). It is the
development branch for an EnsembleFold-oriented Chai-1 wrapper. Model architecture,
checkpoints, feature definitions, and sampling mathematics are outside the scope of
the wrapper.

The detailed Chinese design and operating notes are maintained in
`/Users/wtni/Projects/EnsembleFold/CHAI1_WRAPPER_NOTES.md`.

## Current wrapper contract at a glance

The implemented workflow is:

```text
self-contained JSON
  -> <output>/<name>/<name>_data.json + editable A3M/template resources
  -> process-private reconstruction of native Chai inputs
  -> one native trunk run per requested seed
  -> atomic seed/sample prediction files
```

Compared with upstream Chai-1, this branch replaces the FASTA-oriented `fold` CLI
with a prepared-JSON workflow, adds a durable data/inference boundary, persists
paired and unpaired MSA as `.a3m.zst`, and persists templates only after Chai's
native M8/RCSB/Kalign parsing as mmCIF plus residue mappings. Inference reconstructs
hash-named `.aligned.pqt` files in a private temporary directory and still uses the
native MSA, template, ESM, restraint, feature, trunk, diffusion, and confidence code.

The wrapper exposes one plural `--seeds` option, with one trunk execution and five
diffusion samples per seed by default. The upstream `num_trunk_samples` execution
axis is removed. Results are appendable by non-overlapping seed under
`predictions/{models,summary_confidences,full_data}`, and lightweight `--skip`
checks the five expected nonempty files for every seed/sample. Same-seed concurrent
execution is deliberately not locked and remains unsupported.

## Implementation status

Stage 0 established the fork, branch, and upstream baseline. Stage 1 added the
side-effect-free prepared-input foundation. Stage 2 added the MSA handoff APIs.
Stage 3 added template and restraint handoff. Stage 4 connects the prepared inputs to
the native feature/model boundary, plural seeds, and the wrapper prediction writer.
Stage 5 adds lightweight per-seed skip, validates non-overlapping multi-process writes,
provides `run_chai1.sh`, and removes the old `num_trunk_samples` execution axis. Stage 6
validates the complete contract on an NVIDIA A100 with real model checkpoints.

`chai_lab.data.io.prepared_input` defines a strict, versioned, self-contained
`_data.json` schema, relative-path resolution, declared-resource checks, atomic JSON
writing, and the `<output>/<name>/<name>_data.json` path rule. Molecular entities and
sequences live directly in JSON; there is no `fasta_path`. Protein MSA content or paths
are stored with their protein entity. Sequence hashes are derived internally rather than
persisted.
The top-level `name` is the formal target identity, so renaming the JSON file does not
rename the target. Unknown fields and unsafe target/entity names are rejected instead
of being silently corrected.

Prepared entities expand directly to the lightweight native `Input` objects also used
by the FASTA reader. A protein entry with `id: [A, B]` therefore becomes two native
chain inputs while keeping one sequence and one pair of MSA inputs. The FASTA-based
`chai_lab.chai1.run_inference()` path remains available, but this branch intentionally
removes its `num_trunk_samples` argument: one call now performs exactly one trunk run.

`chai_lab.workflow.build_workflow_plan` remains the side-effect-free path planner.
`run_prepared_workflow()` now executes data-only, inference-only, or combined mode.
Inference reconstructs private Parquet, builds one native `AllAtomFeatureContext`, and
reuses that context while invoking the unchanged `run_folding_on_context()` once for
each requested seed. Temporary MSA/native output directories prefer `SLURM_TMPDIR` and
otherwise use the system temporary directory; no user work-directory option is added.

Stage 2 provides `prepare_msa_bundle()` and `build_private_msa_directory()`. The first
normalizes supplied or ColabFold-generated paired/unpaired A3Ms into canonical
`<target>__<first-entity-id>_{pairedmsa,unpairedmsa}.a3m.zst` artifacts and writes the
updated `_data.json`. Protein JSON follows AF3 Pro's alternatives:
`pairedMsa`/`pairedMsaPath` and `unpairedMsa`/`unpairedMsaPath`; each MSA may be supplied
as inline A3M text or a path, but never both. Explicit empty inline content is retained
and does not trigger a server search. The second API reads the current A3Ms and
reconstructs Chai's hash-named
`.aligned.pqt` files in a caller-owned private temporary directory. No `processed`
directory is persisted. Plain, gzip, xz, and zstd text are detected from magic bytes,
not filename suffixes. Query sequence, aligned width, source database fallback,
pairing-key assignment, padding removal, and paired/unpaired de-duplication are checked
before inference.

The original ColabFold routine now shares the same raw-search and A3M conversion helpers
as the split workflow. Its public function signature and output remain unchanged. A
synthetic two-chain test compares native and persisted-then-reconstructed Parquet files
row for row. The internal sequence hash is still derived at runtime and is not added to
JSON.

Stage 3 generalizes the data API to `prepare_data_bundle()`. Template search still uses
Chai's native ColabFold M8, RCSB coordinate loading, Kalign realignment,
`get_template_data()` filtering, unresolved-residue handling, and four-template limit.
The persistence boundary is after that native parsing. M8 and the download cache remain
temporary; the prepared JSON instead stores an AF3-style template list on each protein:
`mmcifPath`, `queryIndices`, and `templateIndices`. Each accepted single-chain structure
is written as
`msas/<target>__<first-entity-id>_template_<index>.cif.zst`. Missing/null templates mean
"search if enabled" on input, while an empty list means explicitly no templates. Output
is always a list, possibly empty.

`get_prepared_template_context()` reads those structures and mappings, applies Chai's
native protein extraction, tokenization, unresolved-residue filtering, and
`TemplateContext` construction, without re-reading M8 or re-running Kalign. A feature
round-trip test serializes a native `LoadedTemplate`, reconstructs it from only mmCIF
plus residue mappings, and compares all five template feature tensors exactly. The
restraint path is no longer wrapped: preparation neither parses nor copies the CSV. It
only preserves a path that inference will pass to Chai's native restraint parser. The
older `prepare_msa_bundle()` remains as a compatibility wrapper with template search
disabled.

Stage 4 changes `chai-lab fold` to the prepared-JSON workflow. `-D/--run-data-pipeline`
and `-P/--run-inference` accept explicit boolean values; `-r/--seeds` accepts one seed
or a comma-separated ordered list. Omitted seeds are materialized as one printed uint32
value. Each seed runs exactly one trunk and all of its diffusion samples; the wrapper
uses no `num_trunk_samples` axis.

Native candidates are first written in a process-private temporary directory and then
atomically published without confidence re-ranking to:

```text
predictions/models/seed-<seed>_sample-<sample>_model.cif
predictions/summary_confidences/seed-<seed>_sample-<sample>_summary_confidences.json
predictions/full_data/{pae,pde,plddt}_seed-<seed>_sample-<sample>.npz
```

Summary JSON retains Chai's aggregate, pTM, ipTM, per-chain/pair, and clash scores and
adds the seed/sample identity. PAE, PDE, and per-token pLDDT remain separate NPZ files.
There are no rank names, rank CSV, best-model copy, trunk directory, or persistent
native `pred.model_idx_*`/`scores.model_idx_*` files.

Stage 5 adds `-S/--skip`. For each requested seed, it checks the exact sample range
`0..diffusion_samples-1`; a seed is skipped only when its model, summary, PAE, PDE, and
pLDDT files all exist and are nonempty. This is deliberately lightweight: there is no
request hash, manifest, deep file parsing, or cross-process lock. Extra files do not
invalidate a complete seed. Completed and pending seeds may be mixed in one call, and
an all-complete request returns before feature construction or model loading.

The AF3-Pro-style `run_chai1.sh` exposes the data/inference switches, one-or-many seeds,
sampling settings, MSA/template-server switches, GPU selection, and skip. Six spawned
OS processes publishing different seeds into the same target passed the concurrency
regression. Same-seed concurrent jobs remain unsupported: they may duplicate work and
atomically replace the same final names.

Planned work is intentionally gated and will be implemented one stage at a time:

1. prepared JSON schema and workflow foundation (implemented in stage 1);
2. paired/unpaired A3M.zst persistence and private Parquet reconstruction (implemented
   in stage 2);
3. template and restraint handoff (implemented in stage 3);
4. multi-seed CLI and AF3-style result writer (implemented in stage 4);
5. lightweight skip, concurrency validation, `run_chai1.sh`, and removal of
   `num_trunk_samples` (implemented in stage 5);
6. native/combined/split and GPU validation (completed in stage 6).

The wrapper contract uses plural `--seeds`, with one trunk execution per seed.
`num_trunk_samples` is removed from this branch's Python API rather than retained as a
second diversity axis; multiple independent trunk runs are represented by seeds.

## Stage 0 baseline

- Repository state before changes: clean `main` at the exact upstream commit above.
- `python3 -m compileall -q chai_lab tests`: passed under Python 3.14.6, with two
  upstream invalid-escape `SyntaxWarning` messages.
- Full pytest collection was not run because the local base environment has neither
  PyTorch nor pytest installed. GPU/native equivalence testing belongs to the final
  validation stage rather than this metadata-only stage.
- `git diff --check`: passed.

## Stage 2 validation

- 24 tests plus 2 parameterized subtests passed in an isolated Python 3.12 environment.
- Coverage includes self-contained JSON, plain/gzip/xz/zstd magic-byte reads, canonical
  zstd writes, A3M query/width validation, native paired/unpaired merge semantics,
  first-ID naming for homomers, chain expansion during server search, monomer empty
  paired behavior, AF3-style inline/path alternatives, explicit-empty behavior, private
  Parquet construction, normalized-query collision detection, and native/split Parquet
  equality.
- `python3 -m compileall -q chai_lab tests`, focused Ruff checks, Ruff formatting, and
  `git diff --check` passed.

## Stage 3 validation

- 35 tests plus 2 parameterized subtests passed in an isolated Python 3.12 environment
  using Chai's supported pandas 2.x line, including the native online ColabFold tests.
- Coverage includes ephemeral M8 handling, first-ID template naming, AF3-style inline
  and path templates, explicit empty templates, zstd structure output, native constraint
  passthrough, and exact equality of template residue-type, pseudo-beta mask, backbone
  mask, distance, and unit-vector tensors after structure/mapping round-trip.
- Focused Ruff checks and formatting, `python3 -m compileall -q chai_lab tests`, and
  `git diff --check` passed. Full end-to-end model/GPU validation remains stage 6.

## Stage 4 validation

- 48 tests plus 8 parameterized subtests passed for the complete offline wrapper and
  native-restraint regression set in the isolated Python 3.12 environment.
- Coverage includes real CLI data-only execution, CLI option forwarding, seed parsing
  and validation, one feature-context construction reused across multiple seeds,
  native restraint parsing at inference, exact output names, summary serialization,
  separate NPZ keys and rejection of inconsistent candidate arrays. The wrapper does
  not publish Chai's diagnostic `msa_depth.pdf`: it is independent of seed and would
  otherwise be repeatedly overwritten by concurrent seed jobs.
- `chai-lab fold --help`, focused Ruff checks/formatting,
  `python3 -m compileall -q chai_lab tests`, and `git diff --check` passed.

## Stage 5 validation

- 54 tests plus 8 parameterized subtests passed for the complete offline wrapper and
  native-restraint regression set in the isolated Python 3.12 environment.
- Coverage includes complete/partial per-seed skip, early return before model loading,
  removal of `num_trunk_samples` from the Python signature, exact shell-wrapper option
  forwarding, and six independent spawned processes publishing non-overlapping seeds
  into one prediction tree.
- Focused Ruff checks, `bash -n run_chai1.sh`, `python3 -m compileall -q chai_lab tests`,
  and `git diff --check` passed.

## Stage 6 validation

- Deployment: Rocky Linux 10.2, Python 3.12.14, PyTorch 2.7.1+cu118, Kalign 3.6.0,
  and an NVIDIA A100-SXM4-40GB. The shared environment, repository, and 6.5 GiB official
  model assets occupy 13 GiB under `/media/share/db-af3/apps/chai1`; activation is via
  `/media/share/db-af3/apps/chai1/activation.sh`.
- Native FASTA, combined prepared workflow, and split data/inference runs completed on
  GPU with the same seed and sampling settings. After model-size padding, 51 of 52
  feature-context dictionary fields are exactly equal; the sole difference is `pdb_id`
  metadata (`"test"` in the upstream FASTA builder versus the prepared target name),
  which is not consumed by a feature generator. Combined and split `_data.json` files
  are byte-identical.
- Separate A100 processes are not bitwise deterministic even at a fixed seed. In the
  small 10-step diagnostic, aligned atom RMSD was 0.21 Å for native versus combined and
  0.55 Å for combined versus split; aggregate-score differences were 0.00131 and
  0.00038. This is treated as expected numerical variation, not a data-pipeline
  difference.
- A homomer test completed inference with ESM2, replaced unpaired A3M.zst, paired MSA,
  a persisted 1CRN template, and a native contact restraint. Replacing the unpaired MSA
  changed reconstructed Parquet depth from three to four rows and retained the declared
  `uniref90` source.
- A real ColabFold data-only run found 461 MSA rows and four templates. Native
  M8/RCSB/Kalign parsing produced four mapping+CIF.zst artifacts, no M8 was persisted,
  and inference from the resulting `_data.json` completed successfully. Kalign is an
  external runtime requirement and is not installed by the Python package itself.
- Default production settings (three recycles, 200 diffusion steps, five samples)
  produced five complete model/summary/PAE/PDE/pLDDT sets. One-process multi-seed,
  complete and partial skip, and two concurrent GPU processes writing non-overlapping
  seeds all passed; concurrent peak memory for the small test was 10,812 MiB and no
  temporary result files remained.
- The full offline regression suite also passed in the deployed Linux environment:
  54 tests plus 8 subtests, with only upstream deprecation warnings.
