# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

"""Side-effect-free planning skeleton for the EnsembleFold Chai-1 workflow."""

from dataclasses import dataclass
from pathlib import Path

from chai_lab.data.io.prepared_input import (
    PreparedInput,
    PreparedInputError,
    load_prepared_input,
    prepared_input_path,
    validate_target_name,
)


@dataclass(frozen=True)
class WorkflowPlan:
    """Resolved locations and stage choices for one wrapper invocation."""

    input_path: Path
    output_dir: Path
    name: str
    job_dir: Path
    prepared_path: Path
    predictions_dir: Path
    run_data_pipeline: bool
    run_inference: bool
    prepared_input: PreparedInput | None


def build_workflow_plan(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    run_data_pipeline: bool,
    run_inference: bool,
    validate_resources: bool = True,
) -> WorkflowPlan:
    """Validate an invocation and describe where its stages will read and write.

    This is deliberately a planning API in stage 1. It neither runs Chai-1 nor writes
    data-pipeline artifacts. Later stages will execute this stable plan.
    """
    if not run_data_pipeline and not run_inference:
        raise ValueError(
            "At least one of run_data_pipeline or run_inference must be true."
        )

    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input path does not exist or is not a file: {source}")
    output_root = Path(output_dir).expanduser().resolve()

    prepared: PreparedInput | None = None
    if run_data_pipeline:
        if source.suffix.lower() == ".json":
            raise PreparedInputError(
                "The data pipeline expects a FASTA input, not a prepared JSON."
            )
        name = validate_target_name(source.stem)
        manifest = prepared_input_path(output_root, name)
    else:
        if source.suffix.lower() != ".json":
            raise PreparedInputError(
                "Inference-only mode expects a prepared *_data.json input."
            )
        prepared = load_prepared_input(source)
        if validate_resources:
            prepared = prepared.validate_resources(source)
        name = prepared.name
        manifest = source

    job_dir = output_root / name
    return WorkflowPlan(
        input_path=source,
        output_dir=output_root,
        name=name,
        job_dir=job_dir,
        prepared_path=manifest,
        predictions_dir=job_dir / "predictions",
        run_data_pipeline=run_data_pipeline,
        run_inference=run_inference,
        prepared_input=prepared,
    )
