"""Prepared publication and detailed confidence compression contract."""
import json
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
from typer.testing import CliRunner

from chai_lab import workflow
from chai_lab.data.io import prepared_outputs
from chai_lab.data.io.compression import read_text_auto
from chai_lab.data.io.prepared_input import load_prepared_input
from chai_lab.data.io.prepared_msas import prepare_data_bundle
from chai_lab.main import build_app
from tests.test_prepared_writing import request, PAIRED, UNPAIRED


@pytest.mark.parametrize("compressed", [None, False, True])
def test_bundle_reads_zstd_and_publishes_selected_external_text(tmp_path, compressed):
    source = request(tmp_path, template=True)
    old = tmp_path / "old/job_data.json"
    prepare_data_bundle(source, old, use_msa_server=False, compress_fold_input=True)
    target = tmp_path / "new/job_data.json"
    options = {} if compressed is None else {"compress_fold_input": compressed}
    prepare_data_bundle(old, target, use_msa_server=False, **options)
    protein = json.loads(target.read_text())["sequences"][0]["protein"]
    suffix = ".zst" if compressed else ""
    for field, stem, expected in (("pairedMsaPath", "pairedmsa", PAIRED), ("unpairedMsaPath", "unpairedmsa", UNPAIRED)):
        assert protein[field] == f"msas/job__A_{stem}.a3m{suffix}"
        assert read_text_auto(target.parent / protein[field]) == expected
    template = protein["templates"][0]
    assert template["mmcifPath"].endswith(".cif" + suffix)
    assert template["queryIndices"] == [0, 1, 2, 3]
    assert read_text_auto(target.parent / template["mmcifPath"]) == (tmp_path / "template.cif").read_text()
    assert load_prepared_input(target).validate_resources(target).name == "job"


@pytest.mark.parametrize("compressed", [None, False, True])
def test_confidence_shapes_current_format_resume_and_switch_cleanup(tmp_path, monkeypatch, compressed):
    cif = tmp_path / "native.cif"
    cif.write_text("data_native\n#\n")
    candidates = SimpleNamespace(cif_paths=[cif], ranking_data=[None], pae=np.array([[[0.25]]], dtype=np.float32), pde=np.array([[[0.75]]], dtype=np.float32), plddt=np.array([[0.5]], dtype=np.float32))
    monkeypatch.setattr(prepared_outputs, "get_scores", lambda _: {"score": 0.5})
    output = tmp_path / "out"
    embedded = output / "embeddings/untouched.npz"
    embedded.parent.mkdir(parents=True)
    embedded.write_bytes(b"embeddings")
    for selection in (not bool(compressed), compressed):
        options = {} if selection is None else {"compress_full_confidence": selection}
        sample = prepared_outputs.publish_structure_candidates(candidates, predictions_dir=output, seed=7, **options)[0]
        for kind, path, expected in (("pae", sample.pae_path, [[0.25]]), ("pde", sample.pde_path, [[0.75]]), ("plddt", sample.plddt_path, [0.5])):
            assert path.name == f"{kind}_seed-7_sample-0." + ("npz" if selection else "json")
            assert not path.with_suffix(".json" if selection else ".npz").exists()
            if selection:
                with ZipFile(path) as archive:
                    assert all(item.compress_type == ZIP_DEFLATED for item in archive.infolist())
                with np.load(path) as archive:
                    actual = dict(archive)
                    assert actual[kind].dtype == np.dtype("float32")
            else:
                actual = json.loads(path.read_text())
            assert set(actual) == {kind}
            np.testing.assert_array_equal(actual[kind], expected)
        assert prepared_outputs.seed_outputs_complete(output, seed=7, sample_count=1, **options)
    sample.pae_path.unlink()
    assert not prepared_outputs.seed_outputs_complete(output, seed=7, sample_count=1, **options)
    assert embedded.read_bytes() == b"embeddings"


@pytest.mark.parametrize("write", [None, False, True])
def test_cli_inference_only_skip_refreshes_default_plain_snapshot(tmp_path, monkeypatch, write):
    source = request(tmp_path)
    output = tmp_path / "out"
    for relative in ("models/seed-7_sample-0_model.cif", "summary_confidences/seed-7_sample-0_summary_confidences.json", "full_data/pae_seed-7_sample-0.json", "full_data/pde_seed-7_sample-0.json", "full_data/plddt_seed-7_sample-0.json"):
        path = output / "job" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("complete")
    def forbidden(*args, **kwargs):
        pytest.fail("skip or snapshot publication must not load a model or search")
    monkeypatch.setattr(workflow, "make_prepared_feature_context", forbidden)
    args = ["fold", str(source), str(output), "--run-data-pipeline", "false", "--seeds", "7", "--diffusion-samples", "1", "--skip", "true"]
    if write is not None:
        args += ["--write-input-json", str(write).lower()]
    result = CliRunner().invoke(build_app(), args)
    assert result.exit_code == 0, result.output
    snapshot = output / "job/job_data.json"
    assert snapshot.exists() is (write is not False)
    if snapshot.exists():
        resource = output / "job/msas/job__A_unpairedmsa.a3m"
        assert resource.read_text() == UNPAIRED
        (tmp_path / "unpaired.a3m").write_text(">q\nACDE\n>new\nACD-\n")
        result = CliRunner().invoke(build_app(), args)
        assert result.exit_code == 0, result.output
        assert resource.read_text().endswith(">new\nACD-\n")
