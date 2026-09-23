# EnsembleFold wrapper stage status — 2026-09-23

This stage preserves Chai's own scientific processing and public prepared JSON.
Defaults: `write_input_json=true`, `compress_fold_input=false`,
`compress_full_confidence=false`. Prepared resources are external plain text by
default; detailed confidence is JSON, or compressed NPZ when requested.

Included fixes: snapshot all entities' template sources before in-place export;
keep complete-template sequence indices and translate them for Chai internally.
CPU/offline regression covers 141 tests and 20 subtests; no full-model/GPU
equivalence claim is made. See `ENSEMBLEFOLD_CHANGES.md` for usage.

The proposed AF3-style mandatory template declaration was cancelled. Existing
template omission/null handling is retained, including the inference-only
write-dependent behavior. Failure during replacement can still leave mixed old
and new sample files or a partially updated prepared bundle; this is not fixed by
the compression switches. Skip remains existence/non-empty checking, not content
validation or input-condition matching. These limitations remain for later work.
