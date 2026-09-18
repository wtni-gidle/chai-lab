# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

"""Plan and execute the EnsembleFold prepared-input Chai-1 workflow."""

import logging
import os
import secrets
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import torch

from chai_lab.data.io.prepared_input import (
    PreparedInput,
    PreparedInputError,
    load_prepared_input,
    prepared_input_path,
)

logger = logging.getLogger(__name__)


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
    prepared_input: PreparedInput


@dataclass(frozen=True)
class WorkflowResult:
    """Published resources and predictions from one wrapper invocation."""

    prepared_path: Path
    seeds: tuple[int, ...]
    prediction_paths: tuple[Path, ...]


def build_workflow_plan(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    run_data_pipeline: bool,
    run_inference: bool,
    validate_resources: bool = True,
) -> WorkflowPlan:
    """Validate an invocation and describe where its stages will read and write.

    This planning API is side-effect free. ``run_prepared_workflow`` executes the
    returned locations and stage choices.
    """
    if not run_data_pipeline and not run_inference:
        raise ValueError(
            "At least one of run_data_pipeline or run_inference must be true."
        )

    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input path does not exist or is not a file: {source}")
    output_root = Path(output_dir).expanduser().resolve()

    if source.suffix.lower() != ".json":
        raise PreparedInputError(
            "The wrapper expects a self-contained JSON input for every stage."
        )

    prepared = load_prepared_input(source)
    if validate_resources:
        prepared = prepared.validate_resources(source)
    name = prepared.name
    manifest = prepared_input_path(output_root, name) if run_data_pipeline else source

    job_dir = output_root / name
    return WorkflowPlan(
        input_path=source,
        output_dir=output_root,
        name=name,
        job_dir=job_dir,
        prepared_path=manifest,
        predictions_dir=job_dir,
        run_data_pipeline=run_data_pipeline,
        run_inference=run_inference,
        prepared_input=prepared,
    )


def normalize_seeds(
    seeds: str | int | Sequence[int] | None,
) -> tuple[int, ...]:
    """Normalize one or more seeds while preserving the requested order."""
    if seeds is None or (isinstance(seeds, str) and not seeds.strip()):
        return (secrets.randbelow(2**32),)
    if isinstance(seeds, str):
        try:
            values = tuple(int(item.strip()) for item in seeds.split(","))
        except ValueError as error:
            raise ValueError(
                "seeds must be a comma-separated list of integers"
            ) from error
        if any(not item.strip() for item in seeds.split(",")):
            raise ValueError("seeds must not contain empty items")
    elif isinstance(seeds, int):
        values = (seeds,)
    else:
        values = tuple(seeds)

    if not values:
        raise ValueError("At least one seed is required for inference")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in values):
        raise ValueError("Every seed must be an integer in the uint32 range")
    if len(values) != len(set(values)):
        raise ValueError("Seeds must be unique within one invocation")
    return values


def make_prepared_feature_context(
    prepared: PreparedInput,
    *,
    msa_directory: Path,
    esm_device: torch.device,
    use_esm_embeddings: bool = True,
    constraint_path: Path | None = None,
    fasta_names_as_cif_chains: bool = False,
):
    """Build Chai's native feature-context boundary from a resolved prepared JSON."""
    from chai_lab.data.collate.utils import AVAILABLE_MODEL_SIZES
    from chai_lab.data.dataset.all_atom_feature_context import (
        MAX_MSA_DEPTH,
        AllAtomFeatureContext,
    )
    from chai_lab.data.dataset.constraints.restraint_context import (
        RestraintContext,
        load_manual_restraints_for_chai1,
    )
    from chai_lab.data.dataset.embeddings.embedding_context import EmbeddingContext
    from chai_lab.data.dataset.embeddings.esm import get_esm_embedding_context
    from chai_lab.data.dataset.inference_dataset import load_chains_from_raw
    from chai_lab.data.dataset.msas.load import get_msa_contexts
    from chai_lab.data.dataset.msas.msa_context import MSAContext
    from chai_lab.data.dataset.structure.all_atom_structure_context import (
        AllAtomStructureContext,
    )
    from chai_lab.data.dataset.structure.bond_utils import (
        get_atom_covalent_bond_pairs_from_constraints,
    )
    from chai_lab.data.io.prepared_templates import get_prepared_template_context
    from chai_lab.data.parsing.restraints import parse_pairwise_table

    chains = load_chains_from_raw(
        prepared.to_chai_inputs(),
        identifier=prepared.name,
        entity_name_as_subchain=fasta_names_as_cif_chains,
    )
    merged_context = AllAtomStructureContext.merge(
        [chain.structure_context for chain in chains]
    )
    n_actual_tokens = merged_context.num_tokens
    if n_actual_tokens > max(AVAILABLE_MODEL_SIZES):
        raise ValueError(
            f"Too many tokens in input: {n_actual_tokens} > "
            f"{max(AVAILABLE_MODEL_SIZES)}"
        )

    if any(msa_directory.glob("*.aligned.pqt")):
        msa_context, msa_profile_context = get_msa_contexts(
            chains, msa_directory=msa_directory
        )
    else:
        msa_context = MSAContext.create_empty(
            n_tokens=n_actual_tokens, depth=MAX_MSA_DEPTH
        )
        msa_profile_context = MSAContext.create_empty(
            n_tokens=n_actual_tokens, depth=MAX_MSA_DEPTH
        )
    if msa_context.num_tokens != n_actual_tokens:
        raise ValueError(
            "Discrepant tokens in prepared input and MSA: "
            f"{n_actual_tokens} != {msa_context.num_tokens}"
        )

    template_context = get_prepared_template_context(chains, prepared)
    if use_esm_embeddings:
        embedding_context = get_esm_embedding_context(chains, device=esm_device)
    else:
        embedding_context = EmbeddingContext.empty(n_tokens=n_actual_tokens)

    if constraint_path is None:
        restraint_context = RestraintContext.empty()
    else:
        constraints = parse_pairwise_table(constraint_path)
        restraint_context = load_manual_restraints_for_chai1(
            chains,
            crop_idces=None,
            provided_constraints=constraints,
        )
        covalent_a, covalent_b = get_atom_covalent_bond_pairs_from_constraints(
            provided_constraints=constraints,
            token_residue_index=merged_context.token_residue_index,
            token_residue_name=merged_context.token_residue_name,
            token_subchain_id=merged_context.subchain_id,
            token_asym_id=merged_context.token_asym_id,
            atom_token_index=merged_context.atom_token_index,
            atom_ref_name=merged_context.atom_ref_name,
        )
        if covalent_a.numel() > 0 and covalent_b.numel() > 0:
            original_a, original_b = merged_context.atom_covalent_bond_indices
            if original_a.numel() == original_b.numel() == 0:
                merged_context.atom_covalent_bond_indices = (
                    covalent_a,
                    covalent_b,
                )
            else:
                merged_context.atom_covalent_bond_indices = (
                    torch.concatenate([original_a, covalent_a]),
                    torch.concatenate([original_b, covalent_b]),
                )
            if (
                merged_context.atom_covalent_bond_indices[0].numel()
                != merged_context.atom_covalent_bond_indices[1].numel()
            ):
                raise ValueError("Covalent restraint atom indices are inconsistent")

    merged_context.drop_glycan_leaving_atoms_inplace()
    return AllAtomFeatureContext(
        chains=chains,
        structure_context=merged_context,
        msa_context=msa_context,
        profile_msa_context=msa_profile_context,
        template_context=template_context,
        embedding_context=embedding_context,
        restraint_context=restraint_context,
    )


def _temporary_parent() -> Path | None:
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if slurm_tmpdir:
        path = Path(slurm_tmpdir).expanduser()
        if path.is_dir():
            return path
        logger.warning("Ignoring missing SLURM_TMPDIR: %s", path)
    return None


def _run_folding(*args, **kwargs):
    from chai_lab.chai1 import run_folding_on_context

    return run_folding_on_context(*args, **kwargs)


def run_prepared_workflow(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    run_data_pipeline: bool = True,
    run_inference: bool = True,
    use_msa_server: bool = False,
    use_templates_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    max_template_date: str | date = "2099-01-01",
    seeds: str | int | Sequence[int] | None = None,
    recycle_msa_subsample: int = 0,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    num_diffn_samples: int = 5,
    device: str | None = None,
    low_memory: bool = True,
    use_esm_embeddings: bool = True,
    constraint_path: str | Path | None = None,
    fasta_names_as_cif_chains: bool = False,
    skip: bool = False,
) -> WorkflowResult:
    """Execute data preparation and/or inference for one prepared JSON target."""
    if num_diffn_samples <= 0:
        raise ValueError("num_diffn_samples must be positive")
    if num_trunk_recycles <= 0:
        raise ValueError("num_trunk_recycles must be positive")
    if num_diffn_timesteps <= 0:
        raise ValueError("num_diffn_timesteps must be positive")

    plan = build_workflow_plan(
        input_path,
        output_dir,
        run_data_pipeline=run_data_pipeline,
        run_inference=run_inference,
    )
    manifest_path = plan.input_path
    if run_data_pipeline:
        from chai_lab.data.io.prepared_msas import prepare_data_bundle

        prepare_data_bundle(
            plan.input_path,
            plan.prepared_path,
            use_msa_server=use_msa_server,
            use_templates_server=use_templates_server,
            msa_server_url=msa_server_url,
            max_template_date=max_template_date,
        )
        manifest_path = plan.prepared_path

    if not run_inference:
        return WorkflowResult(
            prepared_path=manifest_path,
            seeds=(),
            prediction_paths=(),
        )

    normalized_seeds = normalize_seeds(seeds)
    logger.info("Running Chai-1 seeds: %s", ", ".join(map(str, normalized_seeds)))
    from chai_lab.data.io.prepared_outputs import (
        expected_seed_samples,
        publish_structure_candidates,
        seed_outputs_complete,
    )

    pending_seeds = tuple(
        seed
        for seed in normalized_seeds
        if not skip
        or not seed_outputs_complete(
            plan.predictions_dir,
            seed=seed,
            sample_count=num_diffn_samples,
        )
    )
    skipped_seeds = tuple(
        seed for seed in normalized_seeds if seed not in pending_seeds
    )
    if skipped_seeds:
        logger.info("Skipping complete seeds: %s", ", ".join(map(str, skipped_seeds)))
    expected_model_paths = tuple(
        sample.model_path
        for seed in normalized_seeds
        for sample in expected_seed_samples(
            plan.predictions_dir,
            seed=seed,
            sample_count=num_diffn_samples,
        )
    )
    if not pending_seeds:
        return WorkflowResult(
            prepared_path=manifest_path,
            seeds=normalized_seeds,
            prediction_paths=expected_model_paths,
        )

    prepared = load_prepared_input(manifest_path).validate_resources(manifest_path)
    resolved_constraint_path = None
    if constraint_path is not None:
        resolved_constraint_path = Path(constraint_path).expanduser().resolve()
        if not resolved_constraint_path.is_file():
            raise FileNotFoundError(
                f"constraint_path does not exist or is not a file: "
                f"{resolved_constraint_path}"
            )
    torch_device = torch.device(device if device is not None else "cuda:0")

    from chai_lab.data.io.prepared_msas import build_private_msa_directory

    with tempfile.TemporaryDirectory(
        prefix=f"chai_{plan.name}_", dir=_temporary_parent()
    ) as temporary:
        temporary_root = Path(temporary)
        private_msa_directory = build_private_msa_directory(
            prepared, temporary_root / "msas"
        )
        feature_context = make_prepared_feature_context(
            prepared,
            msa_directory=private_msa_directory,
            esm_device=torch_device,
            use_esm_embeddings=use_esm_embeddings,
            constraint_path=resolved_constraint_path,
            fasta_names_as_cif_chains=fasta_names_as_cif_chains,
        )
        for seed in pending_seeds:
            logger.info("Running seed %d", seed)
            candidates = _run_folding(
                feature_context,
                output_dir=temporary_root / f"seed-{seed}",
                recycle_msa_subsample=recycle_msa_subsample,
                num_trunk_recycles=num_trunk_recycles,
                num_diffn_timesteps=num_diffn_timesteps,
                num_diffn_samples=num_diffn_samples,
                entity_names_as_chain_names_in_output_cif=fasta_names_as_cif_chains,
                seed=seed,
                device=torch_device,
                low_memory=low_memory,
            )
            published = publish_structure_candidates(
                candidates,
                predictions_dir=plan.predictions_dir,
                seed=seed,
            )
            if len(published) != num_diffn_samples:
                raise ValueError(
                    f"Seed {seed} published {len(published)} samples; "
                    f"expected {num_diffn_samples}"
                )

    return WorkflowResult(
        prepared_path=manifest_path,
        seeds=normalized_seeds,
        prediction_paths=expected_model_paths,
    )
