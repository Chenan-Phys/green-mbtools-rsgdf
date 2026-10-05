"""Physical regressions for exports with retained ordinary/Ewald files."""
import hashlib
from pathlib import Path
import shutil
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from pyscf.pbc import df, gto

from green_mbtools.mint import common_utils as comm, integral_utils as iu


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block(reader, pair):
    pieces = []
    for real, imag, sign in reader.sr_loop(pair, compact=False):
        assert sign == 1
        pieces.append(real + 1j * imag)
    return np.concatenate(pieces)


def gram(left, right):
    return left.T @ right.conj()


@pytest.fixture(scope="module", params=[True, False], ids=["eigenvalue", "cholesky"])
def saved_integrals(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("retained-integrals")
    cell = gto.Cell(a=np.eye(3) * 4, atom="H 0 0 0; H 0 0 0.74",
                    basis="sto3g", precision=1e-10, verbose=0).build()
    kpts = cell.make_kpts([2, 1, 1])
    args = SimpleNamespace(df_backend="rsgdf", auxbasis="weigend", beta=None, Nk=0,
                           space_symm=False, tr_symm=True,
                           use_j2c_eig_decomposition=request.param, orth="none", memory=1)
    mydf = comm.construct_gdf(args, cell, kpts)
    mydf._cderi_to_save = str(root / "cderi.h5")
    mydf.build()
    iu.build_legacy_ewald(mydf, cell, kpts, str(root / "cderi_ewald.h5"))
    return root, cell, kpts, args


def export_blocks(folder):
    with h5py.File(folder / "meta.h5") as f:
        starts = f["chunk_indices"][()]
    result = []
    for start in starts:
        with h5py.File(folder / f"VQ_{start}.h5") as f:
            values = f[str(start)][()].view(np.complex128)
            result.extend(values.reshape(values.shape[0], values.shape[1], -1))
    return result


def test_repeated_hf_and_correlation_exports(saved_integrals, tmp_path, monkeypatch):
    root, cell, kpts, args = saved_integrals
    monkeypatch.chdir(tmp_path)
    for name in ("cderi.h5", "cderi_ewald.h5"):
        shutil.copy2(root / name, tmp_path / name)
    hashes = {name: digest(tmp_path / name) for name in ("cderi.h5", "cderi_ewald.h5")}
    ordinary = comm.construct_gdf(args, cell, kpts)
    ordinary._cderi = "cderi.h5"
    corrected = df.GDF(cell, kpts)
    corrected._cderi = "cderi_ewald.h5"
    reference_hf = [block(ordinary, (ki, ki)) for ki in kpts]
    reference_corr = [block(corrected, (ki, ki)) for ki in kpts]
    assert np.linalg.norm(gram(reference_hf[0], reference_hf[0]) -
                          gram(reference_corr[0], reference_corr[0])) > 1e-5
    monkeypatch.setattr(ordinary, "build", lambda: pytest.fail("Matching cache should be reused"))
    for repeat in range(2):
        for kind, correction, refs in (("hf", None, reference_hf),
                                       ("corr", "cderi_ewald.h5", reference_corr)):
            folder = tmp_path / kind
            _, _, irre, pairs, _ = iu.compute_integrals(
                args, cell, ordinary, kpts, cell.nao_nr(), basename=str(folder),
                keep_after=True, cderi_name2=correction)
            values = export_blocks(folder)
            diag = {int(pairs[i, 0]): values[p]
                    for p, i in enumerate(irre) if pairs[i, 0] == pairs[i, 1]}
            for i in diag:
                for j in diag:
                    np.testing.assert_allclose(gram(diag[i], diag[j]), gram(refs[i], refs[j]),
                                               atol=1e-8, rtol=1e-7)
        assert {name: digest(tmp_path / name) for name in hashes} == hashes


def test_missing_explicit_correction_preserves_export(saved_integrals, tmp_path):
    root, cell, kpts, args = saved_integrals
    ordinary = comm.construct_gdf(args, cell, kpts)
    folder = tmp_path / "existing"
    folder.mkdir()
    sentinel = folder / "VQ_0.h5"
    sentinel.write_bytes(b"preserved output")
    with pytest.raises(FileNotFoundError, match="Requested Ewald"):
        iu.compute_integrals(args, cell, ordinary, kpts, cell.nao_nr(),
                             basename=str(folder), cderi_name=str(root / "cderi.h5"),
                             cderi_name2=str(tmp_path / "missing.h5"))
    assert sentinel.read_bytes() == b"preserved output"


def test_correction_shape_check():
    reader = SimpleNamespace(sr_loop=lambda *args, **kwargs: [(np.zeros((2, 1)), np.zeros((2, 1)), 1)])
    with pytest.raises(ValueError, match="AO dimensions"):
        iu._validate_corrected_blocks(reader, np.zeros((1, 3)), 2, 22)
