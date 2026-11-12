"""Stable eigenwhitening and absolute rank under auxiliary scale separation."""
from types import SimpleNamespace
import numpy as np
import pytest
import scipy.linalg as la
from green_mbtools.mint.df_metric import scaled_eigen_metric


@pytest.mark.parametrize("complex_metric", [False, True])
def test_known_coulomb_contraction_with_core_diffuse_scale_separation(complex_metric):
    rng = np.random.default_rng(802)
    a = rng.normal(size=(10,10))
    if complex_metric:
        a = a + 1j*rng.normal(size=a.shape)
    s = a @ a.conj().T + .2*np.eye(10)
    scales = np.logspace(-5,5,10)
    j = scales[:,None]*s*scales[None,:]
    z = rng.normal(size=(10,4)) + 1j*rng.normal(size=(10,4))
    three_center = scales[:,None]*(s@z)
    w, r = scaled_eigen_metric(j,1e-14)
    transformed = w@three_center
    # Analytic contraction: (D S z)† (D S D)^-1 (D S z) = z† S z.
    np.testing.assert_allclose(transformed.conj().T@transformed,z.conj().T@s@z,atol=1e-9,rtol=1e-10)
    np.testing.assert_allclose(w@r,np.eye(10),atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(r@r.conj().T,j,atol=1e-10,rtol=1e-10)


def test_absolute_rank_is_independent_of_equilibration():
    j = np.diag([1e-12,1.,2.])
    w,r = scaled_eigen_metric(j,1e-10)
    ws,rs = scaled_eigen_metric(j,1e-10,np.array([1.,10.,.1]))
    assert w.shape == ws.shape == (2,3)
    np.testing.assert_allclose(w@r,np.eye(2),atol=1e-12)
    rng=np.random.default_rng(2)
    c=rng.normal(size=(3,5))
    np.testing.assert_allclose((w@c).conj().T@(w@c),(ws@c).conj().T@(ws@c),atol=1e-10)


@pytest.mark.parametrize("fixed_scale",[False,True])
def test_absolute_spectrum_preserves_truncated_subspace_under_rotation(fixed_scale):
    scales=np.array([1e-4,1e-4,1e4])
    s=np.diag([2e-12,1.,3.])
    j=scales[:,None]*s*scales[None,:]
    angle=.71
    rotation=np.array([[np.cos(angle),-np.sin(angle),0.],
                       [np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    reference=scales if fixed_scale else None
    w,r=scaled_eigen_metric(j,1e-10,reference)
    wr,rr=scaled_eigen_metric(rotation@j@rotation.T,1e-10,reference)
    assert len(w)==len(wr)==2
    np.testing.assert_allclose(rr@wr,rotation@(r@w)@rotation.T,atol=1e-12,rtol=1e-12)


def test_fixed_positive_reference_allows_a_periodic_null_mode():
    w,r=scaled_eigen_metric(np.diag([0.,2.]),1e-10,np.ones(2))
    assert w.shape==(1,2)
    np.testing.assert_allclose(w@r,np.ones((1,1)),atol=1e-12)


@pytest.mark.parametrize("matrix,error", [
    (np.array([[1.,2.],[2.,1.]]),la.LinAlgError),
    (np.zeros((2,2)),la.LinAlgError),
    (np.array([[1.,.5],[0.,1.]]),ValueError),
    (np.array([[np.nan]]),ValueError),
])
def test_invalid_positive_metrics_fail(matrix,error):
    with pytest.raises(error):scaled_eigen_metric(matrix,1e-10)


@pytest.mark.parametrize("backend", ["ccgdf","rsgdf"])
def test_ge_gamma_eigen_interactions_match_cholesky(tmp_path,backend):
    from pyscf.pbc import gto,df
    from green_mbtools.mint import common_utils as comm
    a=5.7
    cell=gto.Cell(a=np.array([[0,a/2,a/2],[a/2,0,a/2],[a/2,a/2,0]]),
        atom=f"Ge 0 0 0; Ge {a/4} {a/4} {a/4}",basis="x2c-svpall",
        precision=1e-10,verbose=0,max_memory=2000).build()
    auxiliary=df.aug_etb(cell,beta=2.)
    columns=np.unique(np.r_[np.random.default_rng(381).choice(cell.nao**2,12,replace=False),[0,65,130,4095]])
    grams=[]
    for eig in (False,True):
        args=SimpleNamespace(df_backend=backend,auxbasis=auxiliary,beta=None,Nk=0,
            space_symm=False,tr_symm=False,x2c=1,use_j2c_eig_decomposition=eig)
        mydf=comm.construct_gdf(args,cell,cell.make_kpts([1,1,1]))
        mydf._cderi_to_save=str(tmp_path/f"{backend}-{eig}.h5")
        mydf.build()
        blocks=[(re+1j*im)[:,columns] for re,im,sign in mydf.sr_loop(compact=False)]
        factor=np.concatenate(blocks)
        grams.append(factor.T@factor.conj())
    np.testing.assert_allclose(*grams,atol=1e-8,rtol=1e-7)
