# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

"""Command line interface."""

import logging
from pathlib import Path

import typer

from chai_lab.data.parsing.msas.aligned_pqt import merge_a3m_in_directory
from chai_lab.workflow import run_prepared_workflow

logging.basicConfig(level=logging.INFO)

CITATION = """
@article{Chai-1-Technical-Report,
	title        = {Chai-1: Decoding the molecular interactions of life},
	author       = {{Chai Discovery}},
	year         = 2024,
	journal      = {bioRxiv},
	publisher    = {Cold Spring Harbor Laboratory},
	doi          = {10.1101/2024.10.10.615955},
	url          = {https://www.biorxiv.org/content/early/2024/10/11/2024.10.10.615955},
	elocation-id = {2024.10.10.615955},
	eprint       = {https://www.biorxiv.org/content/early/2024/10/11/2024.10.10.615955.full.pdf}
}
""".strip()


def citation():
    """Print citation information"""
    typer.echo(CITATION)


def _boolean_option(value: str, option_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise typer.BadParameter(f"{option_name} must be true or false")


def fold(
    input_path: Path = typer.Argument(  # noqa: B008
        ..., exists=True, dir_okay=False
    ),
    output_dir: Path = typer.Argument(..., file_okay=False),  # noqa: B008
    run_data_pipeline: str = typer.Option(
        "true", "-D", "--run-data-pipeline", help="Prepare MSA/template resources."
    ),
    run_model_inference: str = typer.Option(
        "true", "-P", "--run-inference", help="Run model inference."
    ),
    seeds: str | None = typer.Option(
        None, "-r", "--seeds", help="One seed or comma-separated seeds."
    ),
    diffusion_samples: int = typer.Option(
        5, "-n", "--diffusion-samples", help="Diffusion samples per seed."
    ),
    recycling_steps: int = typer.Option(
        3, "-c", "--recycling-steps", help="Trunk recycling steps."
    ),
    sampling_steps: int = typer.Option(
        200, "-p", "--sampling-steps", help="Diffusion sampling steps."
    ),
    use_msa_server: str = typer.Option(
        "false", "-M", "--use-msa-server", help="Search missing protein MSAs."
    ),
    use_templates_server: str = typer.Option(
        "false", "-T", "--use-templates-server", help="Search missing templates."
    ),
    max_template_date: str = typer.Option(
        "2099-01-01",
        "--max-template-date",
        help="Latest allowed searched-template release date (YYYY-MM-DD).",
    ),
    msa_server_url: str = typer.Option("https://api.colabfold.com", "--msa-server-url"),
    recycle_msa_subsample: int = typer.Option(0, "--recycle-msa-subsample"),
    device: str | None = typer.Option(None, "--device"),
    low_memory: str = typer.Option("true", "--low-memory"),
    use_esm_embeddings: str = typer.Option(
        "true",
        "--use-esm-embeddings",
        help="Generate ESM embeddings during inference.",
    ),
    constraint_path: Path | None = typer.Option(
        None,
        "--constraint-path",
        exists=True,
        dir_okay=False,
        help="Optional native Chai restraint CSV used during inference.",
    ),
    fasta_names_as_cif_chains: str = typer.Option(
        "false",
        "--fasta-names-as-cif-chains",
        help="Use input entity IDs as internal subchain and output CIF chain names.",
    ),
    skip: str = typer.Option(
        "false", "-S", "--skip", help="Skip seeds with every expected output file."
    ),
) -> None:
    """Run the prepared-JSON data pipeline and/or Chai-1 inference."""
    try:
        result = run_prepared_workflow(
            input_path,
            output_dir,
            run_data_pipeline=_boolean_option(run_data_pipeline, "--run-data-pipeline"),
            run_inference=_boolean_option(run_model_inference, "--run-inference"),
            use_msa_server=_boolean_option(use_msa_server, "--use-msa-server"),
            use_templates_server=_boolean_option(
                use_templates_server, "--use-templates-server"
            ),
            msa_server_url=msa_server_url,
            max_template_date=max_template_date,
            seeds=seeds,
            recycle_msa_subsample=recycle_msa_subsample,
            num_trunk_recycles=recycling_steps,
            num_diffn_timesteps=sampling_steps,
            num_diffn_samples=diffusion_samples,
            device=device,
            low_memory=_boolean_option(low_memory, "--low-memory"),
            use_esm_embeddings=_boolean_option(
                use_esm_embeddings, "--use-esm-embeddings"
            ),
            constraint_path=constraint_path,
            fasta_names_as_cif_chains=_boolean_option(
                fasta_names_as_cif_chains,
                "--fasta-names-as-cif-chains",
            ),
            skip=_boolean_option(skip, "--skip"),
        )
    except (ValueError, FileNotFoundError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Prepared input: {result.prepared_path}")
    if result.seeds:
        typer.echo(f"Seeds: {','.join(map(str, result.seeds))}")
        typer.echo(f"Published models: {len(result.prediction_paths)}")


def build_app() -> typer.Typer:
    """Build the command tree for console use and CLI tests."""
    app = typer.Typer()
    app.command("fold", help="Run prepared-JSON Chai-1 workflow.")(fold)
    app.command(
        "a3m-to-pqt",
        help="Convert all a3m files in a directory for a *single sequence* into a aligned parquet file",
    )(merge_a3m_in_directory)
    app.command("citation", help="Print citation information")(citation)
    return app


def cli():
    build_app()()


if __name__ == "__main__":
    cli()
