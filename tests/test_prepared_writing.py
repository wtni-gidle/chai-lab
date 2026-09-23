"""Independent snapshot publication: real preparation/consumers, no model/network."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from chai_lab import workflow
from chai_lab.data.io import prepared_outputs
from chai_lab.data.io.compression import read_text_auto
from chai_lab.main import build_app

PAIRED = ">q\nACDE\n>paired\nAC-E\n"
UNPAIRED = ">q\nACDE\n>old\nA-DE\n"
REPLACEMENT = ">q\nACDE\n>new\nACD-\n"


def request(tmp_path, *, template=False):
    source = tmp_path / "input.json"
    (tmp_path / "unpaired.a3m").write_text(UNPAIRED)
    protein = {
        "id": ["A"], "sequence": "ACDE", "pairedMsa": PAIRED,
        "unpairedMsaPath": "unpaired.a3m", "templates": [],
    }
    if template:
        from tests.test_prepared_templates import PreparedTemplateFeatureRoundTripTest

        cif = PreparedTemplateFeatureRoundTripTest._protein_mmcif()
        (tmp_path / "template.cif").write_text(cif)
        protein["templates"] = [{
            "mmcifPath": "template.cif",
            "queryIndices": [0, 1, 2, 3], "templateIndices": [0, 1, 2, 3],
        }]
    source.write_text(json.dumps({
        "version": 1, "name": "job", "sequences": [{"protein": protein}],
    }))
    return source


def completed_seed(output):
    # Deliberately non-parseable content: the established skip contract is nonempty.
    for sample in prepared_outputs.expected_seed_samples(output / "job", seed=7, sample_count=1):
        for path in (sample.model_path, sample.summary_path, sample.pae_path,
                     sample.pde_path, sample.plddt_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"complete")


@pytest.mark.parametrize("data,inference", [(True, False), (True, True), (False, True)])
@pytest.mark.parametrize("write", [None, False, True])
def test_publication_is_independent_and_updates_even_when_seed_skips(
    tmp_path, monkeypatch, data, inference, write
):
    source = request(tmp_path)
    source_bytes = source.read_bytes()
    output = tmp_path / "out"
    completed_seed(output)
    snapshot = output / "job/job_data.json"
    snapshot.write_bytes(b"old snapshot")
    stale = output / "job/msas/job__A_unpairedmsa.a3m.zst"
    stale.parent.mkdir()
    stale.write_bytes(b"old resource")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))

    def unexpected(*args, **kwargs):
        pytest.fail("completed seed / data-only must not build model features")

    monkeypatch.setattr(workflow, "make_prepared_feature_context", unexpected)
    result = workflow.run_prepared_workflow(
        source, output, run_data_pipeline=data, run_inference=inference,
        write_input_json=write, seeds=7, num_diffn_samples=1, skip=True,
    compress_fold_input=True)
    publish = True if write is None else write
    if publish:
        protein = json.loads(snapshot.read_text())["sequences"][0]["protein"]
        assert protein["unpairedMsaPath"] == "msas/job__A_unpairedmsa.a3m.zst"
        assert read_text_auto(stale) == UNPAIRED
        assert read_text_auto(snapshot.parent / protein["pairedMsaPath"]) == PAIRED
        assert result.prepared_path == snapshot
    else:
        assert snapshot.read_bytes() == b"old snapshot"
        assert stale.read_bytes() == b"old resource"
        assert result.prepared_path == (None if data else source)
    assert source.read_bytes() == source_bytes
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("failure", [False, True])
def test_private_preparation_lives_through_real_features_and_is_cleaned(
    tmp_path, monkeypatch, failure
):
    source = request(tmp_path, template=True)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))
    seen = []

    def fold(context, *, output_dir, **kwargs):
        # Real native MSA and template consumers have already run at this point.
        rows = {context.msa_context.ith_sequence(i)
                for i in range(context.msa_context.depth)
                if context.msa_context.mask[i].any()}
        assert {"ACDE", "AC-E", "A-DE"} <= rows
        assert context.template_context.template_pseudo_beta_mask.any()
        assert list(scratch.rglob("*_data.json"))
        assert list(scratch.rglob("*.cif.zst"))
        seen.append(True)
        if failure:
            raise RuntimeError("model boundary failed")
        output_dir.mkdir()
        cif = output_dir / "sample.cif"
        cif.write_text("data_prediction\n")
        return SimpleNamespace(
            cif_paths=[cif], ranking_data=[None],
            pae=np.zeros((1, 4, 4)), pde=np.zeros((1, 4, 4)),
            plddt=np.zeros((1, 4)),
        )

    monkeypatch.setattr(workflow, "_run_folding", fold)
    monkeypatch.setattr(prepared_outputs, "get_scores", lambda _: {})
    output = tmp_path / "out"
    kwargs = dict(write_input_json=False, device="cpu", use_esm_embeddings=False,
                  seeds=7, num_diffn_samples=1)
    if failure:
        with pytest.raises(RuntimeError, match="model boundary failed"):
            workflow.run_prepared_workflow(source, output, **kwargs, compress_fold_input=True)
    else:
        result = workflow.run_prepared_workflow(source, output, **kwargs, compress_fold_input=True)
        assert result.prepared_path is None
        assert result.prediction_paths[0].is_file()
    assert seen == [True]
    assert list(scratch.iterdir()) == []
    assert not (output / "job/job_data.json").exists()
    assert not (output / "job/msas").exists()


@pytest.mark.parametrize("data", [False, True])
@pytest.mark.parametrize("write", [False, True])
def test_same_path_unpaired_replacement_reaches_native_context_without_changing_template(
    tmp_path, monkeypatch, data, write
):
    source = request(tmp_path, template=True)
    output = tmp_path / "out"
    observed = []

    def fold(context, *, output_dir, **kwargs):
        msa = context.msa_context
        rows = {msa.ith_sequence(i) for i in range(msa.depth) if msa.mask[i].any()}
        # Compare actual native template tensors across the two conditions.
        observed.append((rows, context.template_context.template_distances.clone()))
        output_dir.mkdir()
        cif = output_dir / "sample.cif"
        cif.write_text("data_prediction\n")
        return SimpleNamespace(cif_paths=[cif], ranking_data=[None],
                               pae=np.zeros((1, 4, 4)), pde=np.zeros((1, 4, 4)),
                               plddt=np.zeros((1, 4)))

    monkeypatch.setattr(workflow, "_run_folding", fold)
    monkeypatch.setattr(prepared_outputs, "get_scores", lambda _: {})
    settings = dict(device="cpu", use_esm_embeddings=False, seeds=7, num_diffn_samples=1)
    workflow.run_prepared_workflow(source, output, write_input_json=True, **settings, compress_fold_input=True)
    before = {p: p.read_bytes() for p in (output / "job").rglob("*")
              if p.is_file() and p.parent.name not in {"models", "full_data", "summary_confidences"}}
    (tmp_path / "unpaired.a3m").write_text(REPLACEMENT)
    workflow.run_prepared_workflow(source, output, run_data_pipeline=data,
                                  write_input_json=write, **settings, compress_fold_input=True)
    assert "A-DE" in observed[0][0] and "ACD-" not in observed[0][0]
    assert "ACD-" in observed[1][0] and "A-DE" not in observed[1][0]
    assert "AC-E" in observed[0][0] & observed[1][0]
    np.testing.assert_array_equal(observed[0][1].numpy(), observed[1][1].numpy())
    if not write:
        assert {p: p.read_bytes() for p in before} == before
    else:
        snapshot = output / "job/job_data.json"
        protein = json.loads(snapshot.read_text())["sequences"][0]["protein"]
        assert read_text_auto(snapshot.parent / protein["unpairedMsaPath"]) == REPLACEMENT
        assert read_text_auto(snapshot.parent / protein["pairedMsaPath"]) == PAIRED
        assert read_text_auto(snapshot.parent / protein["templates"][0]["mmcifPath"]) == (tmp_path / "template.cif").read_text()
        assert protein["templates"][0]["templateIndices"] == [0, 1, 2, 3]


def test_inference_only_publication_cannot_enable_search(tmp_path, monkeypatch):
    from chai_lab.data.dataset.msas import colabfold

    source = request(tmp_path)
    payload = json.loads(source.read_text())
    payload["sequences"][0]["protein"] = {"id": ["A"], "sequence": "ACDE"}
    source.write_text(json.dumps(payload))
    output = tmp_path / "out"
    completed_seed(output)

    def forbidden(*args, **kwargs):
        pytest.fail("D=false must not search even with server flags true")

    monkeypatch.setattr(colabfold, "generate_colabfold_a3ms", forbidden)
    workflow.run_prepared_workflow(
        source, output, run_data_pipeline=False, write_input_json=True,
        use_msa_server=True, use_templates_server=True, seeds=7, skip=True,
        num_diffn_samples=1,
    compress_fold_input=True)
    assert (output / "job/job_data.json").is_file()


@pytest.mark.parametrize("value", [0, "false", [], {}])
def test_non_boolean_write_rejected_before_output(tmp_path, value):
    source = request(tmp_path)
    with pytest.raises(ValueError, match="write_input_json"):
        workflow.run_prepared_workflow(source, tmp_path / "out", run_inference=False,
                                      write_input_json=value, compress_fold_input=True)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("flag", ["-J", "--write-input-json", "--write_input_json"])
def test_cli_can_prepare_without_publication(tmp_path, flag):
    source = request(tmp_path)
    output = tmp_path / "out"
    result = CliRunner().invoke(build_app(), ["fold", str(source), str(output),
                                           "-P", "false", flag, "false"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert not output.exists()
    assert "Prepared input: None" not in result.output


def test_shell_forwards_explicit_write_option(tmp_path):
    source = request(tmp_path)
    executable = tmp_path / "chai-lab"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    executable.chmod(0o755)
    shell = Path(__file__).resolve().parents[1] / "run_chai1.sh"
    result = subprocess.run([str(shell), "-i", str(source), "-o", str(tmp_path / "out"),
                             "-D", "true", "-P", "false", "-J", "false"],
                            env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert args[args.index("--write-input-json") + 1] == "false"
