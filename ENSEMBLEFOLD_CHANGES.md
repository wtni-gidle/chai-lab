# EnsembleFold wrapper changes

This branch is based on upstream commit
`66c38d1fe5c6756a89ff8596b1dea87d305ec06f` (`v0.6.1-20-g66c38d1`). It is the
development branch for an EnsembleFold-oriented Chai-1 wrapper. Model architecture,
checkpoints, feature definitions, and sampling mathematics are outside the scope of
the wrapper.

The detailed Chinese design and operating notes are maintained in
`/Users/wtni/Projects/EnsembleFold/CHAI1_WRAPPER_NOTES.md`.

## Implementation status

Stage 0 established the fork, branch, and upstream baseline. Stage 1 added the
side-effect-free prepared-input foundation. Stage 2 added the MSA handoff APIs.
Stage 3 adds template and restraint handoff; the native CLI and prediction writer are
still unchanged.

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
chain inputs while keeping one sequence and one pair of MSA inputs. The public native
FASTA path remains available and behavior-compatible.

`chai_lab.workflow.build_workflow_plan` validates data/inference stage combinations and
calculates target, job, prepared-input, and prediction locations. In stage 1 it only
returns a plan: it does not search data, run Chai-1, or write prediction artifacts. The
existing `chai-lab fold` command still calls the native `run_inference` function.

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

Stage 3 generalizes the data API to `prepare_data_bundle()`. In addition to MSA files,
it publishes template hits as `templates/all_chain_templates.m8`, normalizes every CIF
referenced by the M8 table into `templates/cifs/<PDB>.cif.gz`, and copies a native
restraint table to `constraints/<target>.restraints.csv`. All paths written to the
prepared JSON are relative to that JSON. A declared template bundle uses M8 entity
names or sequence hashes according to `templates.query_id_mode`; a ColabFold template
search remaps server IDs `101, 102, ...` to the sequence hashes used by Chai at
inference. Declared and server templates cannot be combined accidentally.

The restraint CSV is parsed by Chai's native parser during preparation and its chain
names are checked against either JSON entity IDs or automatic A/B/C subchain names,
according to `entity_ids_as_cif_chains`. `get_prepared_inference_resources()` converts a
resolved prepared JSON into Chai's native M8 path, job-local CIF directory, lookup mode,
and restraint path. It performs no model work or downloads. The older
`prepare_msa_bundle()` remains as a compatibility wrapper with template-server search
disabled.

Planned work is intentionally gated and will be implemented one stage at a time:

1. prepared JSON schema and workflow foundation (implemented in stage 1);
2. paired/unpaired A3M.zst persistence and private Parquet reconstruction (implemented
   in stage 2);
3. template and restraint handoff (implemented in stage 3);
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

- 32 tests plus 7 parameterized subtests passed in an isolated Python 3.12 environment.
- Coverage includes local plain/gzip/zstd template CIFs, canonical job-local CIF output,
  server M8 ID remapping, entity-name query validation, download fallback injection,
  declared/server conflict rejection, native restraint parsing, both restraint chain-ID
  conventions, and resolved inference-resource mapping.
- Focused Ruff checks and formatting, `python3 -m compileall -q chai_lab tests`, and
  `git diff --check` passed. Full model/GPU validation remains stage 6.
