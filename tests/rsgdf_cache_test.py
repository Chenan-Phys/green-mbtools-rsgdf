"""Cache reuse must honor the requested producer and physical settings."""
import hashlib
import os
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from green_mbtools.mint import common_utils as comm, integral_utils as iu
from green_mbtools.mint.pyscf_init import pyscf_pbc_init


def parameters(backend, output):
    return ["--a", "4 0 0\n0 4 0\n0 0 4", "--atom", "H 0 0 0\nH 0 0 0.74",
            "--basis", "sto3g", "--auxbasis", "weigend", "--nk", "2", "1", "1",
            "--df_backend", backend, "--grid_only", "true", "--space_symm", "false",
            "--tr_symm", "true", "--output_path", str(output)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    result = {}
    previous = Path.cwd()
    try:
        for backend in ("ccgdf", "rsgdf"):
            root = tmp_path_factory.mktemp("cache-" + backend)
            os.chdir(root)
            system = pyscf_pbc_init(comm.init_pbc_params(parameters(backend, root / "input.h5")))
            system.mean_field_input()
            result[backend] = root / "cderi.h5"
    finally:
        os.chdir(previous)
    return result


def prepared(caches, tmp_path, monkeypatch, backend, cached_backend=None):
    monkeypatch.chdir(tmp_path)
    cache = tmp_path / "cderi.h5"
    shutil.copy2(caches[cached_backend or backend], cache)
    system = pyscf_pbc_init(comm.init_pbc_params(parameters(backend, tmp_path / "input.h5")))
    return system, cache


@pytest.mark.parametrize("backend", ["ccgdf", "rsgdf"])
def test_matching_cache_reused(caches, tmp_path, monkeypatch, backend):
    system, cache = prepared(caches, tmp_path, monkeypatch, backend)
    before = digest(cache)
    monkeypatch.setattr(iu.GreenGDF, "_make_j3c", lambda *a, **k: pytest.fail("Unexpected rebuild"))
    system.mean_field_input()
    assert (tmp_path / "input.h5").is_file()
    assert digest(cache) == before


@pytest.mark.parametrize("cached_backend", ["ccgdf", "rsgdf"])
@pytest.mark.parametrize("entry", ["initialization", "export"])
def test_backend_switch_preserves_cache_and_outputs(caches, tmp_path, monkeypatch, cached_backend, entry):
    backend = "rsgdf" if cached_backend == "ccgdf" else "ccgdf"
    system, cache = prepared(caches, tmp_path, monkeypatch, backend, cached_backend)
    before = digest(cache)
    folder = tmp_path / "export"
    folder.mkdir()
    output = tmp_path / "input.h5" if entry == "initialization" else folder / "VQ_0.h5"
    output.write_bytes(b"original output")
    monkeypatch.setattr(iu.GreenGDF, "_make_j3c", lambda *a, **k: pytest.fail("Mismatch must not rebuild"))
    with pytest.raises(RuntimeError, match="changed backend"):
        if entry == "initialization":
            system.mean_field_input()
        else:
            iu.compute_integrals(system.args, system.cell, system.df_object(), system.kmesh,
                                 system.cell.nao_nr(), basename=str(folder), keep_after=True)
    assert digest(cache) == before
    assert output.read_bytes() == b"original output"


@pytest.mark.parametrize("backend", ["ccgdf", "rsgdf"])
@pytest.mark.parametrize("setting", ["precision", "geometry", "basis", "auxbasis", "kpts", "mesh",
                                     "eta", "threshold", "decomposition", "space_symm", "tr_symm"])
def test_changed_settings_rejected(caches, tmp_path, monkeypatch, backend, setting):
    system, cache = prepared(caches, tmp_path, monkeypatch, backend)
    mydf = system.df_object()
    before = digest(cache)
    assert iu.df_cache.validate_df_cache(mydf, cache) == ("_CCGDFBuilder" if backend == "ccgdf" else "_RSGDFBuilder")
    if setting == "precision":
        mydf.cell.precision *= 0.1
    elif setting == "geometry":
        mydf.cell._atom[0][1][0] += 0.01
    elif setting == "basis":
        mydf.cell._basis["H"][0][1][0] *= 1.001
    elif setting == "auxbasis":
        mydf.auxbasis = {"H": [[0, [1.0, 1.0]]]}
    elif setting == "kpts":
        mydf.kpts = mydf.kpts + np.array([0.01, 0, 0])
    elif setting == "mesh":
        mydf.mesh = [15, 15, 15]
    elif setting == "eta":
        mydf.eta = 0.3
    elif setting == "threshold":
        mydf.linear_dep_threshold *= 10
    elif setting == "decomposition":
        mydf.use_j2c_eig_decomposition = not mydf.use_j2c_eig_decomposition
    else:
        setattr(mydf, setting, not getattr(mydf, setting))
    with pytest.raises(RuntimeError, match="changed"):
        iu.df_cache.validate_df_cache(mydf, cache)
    assert digest(cache) == before


def test_incomplete_actual_frames_rejected(caches, tmp_path, monkeypatch):
    system, cache = prepared(caches, tmp_path, monkeypatch, "rsgdf")
    with h5py.File(cache, "a") as f:
        del f["j2c/metric_factors/0"]
    before = digest(cache)
    with pytest.raises(RuntimeError, match="incomplete actual RSGDF frame"):
        system.mean_field_input()
    assert digest(cache) == before


@pytest.mark.parametrize("allow", [False, True])
def test_untagged_ccgdf_requires_explicit_compatibility(caches, tmp_path, monkeypatch, caplog, allow):
    system, cache = prepared(caches, tmp_path, monkeypatch, "ccgdf")
    system.args = comm.init_pbc_params(parameters("ccgdf", tmp_path / "input.h5")
                                      + ["--allow_legacy_df_cache", str(allow).lower()])
    with h5py.File(cache, "a") as f:
        del f[iu.df_cache.GROUP]
    before = digest(cache)
    if allow:
        system.mean_field_input()
        assert "provenance is unverifiable" in caplog.text
    else:
        with pytest.raises(RuntimeError, match="untagged cache"):
            system.mean_field_input()
    assert digest(cache) == before


@pytest.mark.parametrize("requested", ["ccgdf", "rsgdf"])
def test_untagged_rsgdf_never_uses_legacy_ccgdf_option(caches, tmp_path, monkeypatch, requested):
    system, cache = prepared(caches, tmp_path, monkeypatch, requested, "rsgdf")
    system.args.allow_legacy_df_cache = True
    with h5py.File(cache, "a") as f:
        del f[iu.df_cache.GROUP]
    with pytest.raises(RuntimeError, match="untagged cache"):
        system.mean_field_input()


def test_ordinary_file_cannot_be_a_correction(caches, tmp_path, monkeypatch):
    system, cache = prepared(caches, tmp_path, monkeypatch, "rsgdf")
    folder = tmp_path / "correlation"
    with pytest.raises(RuntimeError, match="changed.*role"):
        iu.compute_integrals(system.args, system.cell, system.df_object(), system.kmesh,
                             system.cell.nao_nr(), basename=str(folder), cderi_name2=str(cache))
    assert not folder.exists()


@pytest.mark.parametrize("damage", ["schema", "checksum", "producer"])
def test_invalid_provenance_rejected(caches, tmp_path, monkeypatch, damage):
    system, cache = prepared(caches, tmp_path, monkeypatch, "rsgdf")
    with h5py.File(cache, "a") as f:
        group = f[iu.df_cache.GROUP]
        if damage == "schema":
            group.attrs["schema_version"] = 999
        elif damage == "checksum":
            group.attrs["request_sha256"] = "invalid"
        else:
            group.attrs["producer"] = "_CCGDFBuilder"
    with pytest.raises(RuntimeError, match="DF cache mismatch"):
        system.mean_field_input()
