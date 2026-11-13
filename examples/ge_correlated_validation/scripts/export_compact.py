"""Tractable Ge valence model for the complete GREEN correlated matrix.

PySCF HF supplies initialization only. GREEN GF2/GW supply every accepted
scientific energy. This model is deliberately separate from all-electron Ge.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import resource
import time
import numpy as np
from pyscf import lib
from pyscf.pbc import df,scf
from pyscf.pbc.scf import chkfile
from green_mbtools.mint import common_utils as comm,integral_utils as iu
from green_mbtools.mint.pyscf_init import pyscf_pbc_init

def run(options):
    assert Path(iu.__file__).resolve().is_relative_to(Path(os.environ['FIX_SRC']).resolve())
    lib.num_threads(4)
    out=options.output.resolve();out.mkdir(parents=True,exist_ok=False);os.chdir(out)
    a=options.a
    params=['--a',f'0 {a/2} {a/2}\n{a/2} 0 {a/2}\n{a/2} {a/2} 0',
            '--atom',f'Ge 0 0 0\nGe {a/4} {a/4} {a/4}',
            '--basis','gth-szv','--pseudo','gth-pbe','--nk','2',
            '--restricted','true','--x2c','0','--space_symm','false','--tr_symm','false',
            '--df_backend',options.backend,'--use_j2c_eig_decomposition','true',
            '--keep_cderi','true','--memory','256','--orth','none',
            '--output_path',str(out/'input.h5'),'--int_path',str(out/'df_int'),
            '--hf_int_path',str(out/'df_hf_int')]
    args=comm.init_pbc_params(params)
    system=pyscf_pbc_init(args)
    system.cell.precision=1e-10
    system.cell.mesh=None;system.cell.rcut=None
    system.cell.build(False,False)
    args.auxbasis=df.aug_etb(system.cell,beta=2.0);args.beta=None
    mydf=system.df_object();mydf.max_memory=2500
    mydf._cderi_to_save=str(out/'cderi.h5')
    start=time.monotonic();mydf.build();ordinary=time.monotonic()-start
    mf=scf.KRHF(system.cell,system.kmesh)
    mf.with_df=mydf;mf.exxdiv='ewald'
    mf.conv_tol=1e-11;mf.conv_tol_grad=1e-6;mf.max_cycle=100;mf.chkfile=str(out/'hf-seed.chk')
    common_seed=out.parent/'ge-ccgdf'/'hf-seed.chk'
    seed_reused=options.backend=='rsgdf' and common_seed.exists()
    if seed_reused:
        _,saved=chkfile.load_scf(str(common_seed))
        for key in ('mo_coeff','mo_occ','mo_energy'): setattr(mf,key,saved[key])
        mf.e_tot=json.loads((common_seed.parent/'generation.json').read_text())['hf_seed_energy_hartree']
        mf.converged=True
    else:
        mf.kernel()
    assert mf.converged,'HF initialization failed; no correlated input accepted'
    overlap=mf.get_ovlp();hcore=mf.get_hcore();density=mf.make_rdm1()
    fock=mf.get_fock(dm=density)
    mf.get_ovlp=lambda *a,**k:overlap
    mf.get_hcore=lambda *a,**k:hcore
    mf.get_veff=lambda *a,**k:fock-hcore
    mf.get_fock=lambda *a,**k:fock
    system.mf_object=lambda *a,**k:mf
    system.mean_field_input(mydf)
    result={'status':'EXPORTED','model':'Ge valence GTH-SZV / GTH-PBE, 2x2x2',
            'a_angstrom':a,'volume_angstrom3':a**3/4,'backend':options.backend,
            'basis':'gth-szv','pseudo':'gth-pbe','x2c':0,'diffuse_cutoff':None,
            'nao':system.cell.nao_nr(),'naux':mydf.auxcell.nao_nr(),'nk':2,
            'nelectron':system.cell.nelectron,'precision':system.cell.precision,
            'hf_seed_energy_hartree':mf.e_tot,'hf_seed_used_as_scientific_result':False,
            'hf_seed_energy_tolerance':1e-11,'hf_seed_gradient_tolerance':1e-6,
            'hf_seed_reused_from_ccgdf':seed_reused,
            'pyscf_mean_field_kernel_executed':not seed_reused,
            'producer_revision':'0f101cb945b4d13cb71c9926fa023942ec14b60f',
            'ordinary_builder':mydf._green_df_builder_name,'ordinary_build_seconds':ordinary,
            'seed_spectrum_min_hartree':float(np.min(mf.mo_energy)),
            'seed_spectrum_max_hartree':float(np.max(mf.mo_energy)),
            'export_wall_seconds':time.monotonic()-start,
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out/'generation.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPACT_GREEN_EXPORT_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--a',type=float,required=True)
    parser.add_argument('--backend',choices=['ccgdf','rsgdf'],required=True);parser.add_argument('--output',type=Path,required=True)
    logging.basicConfig(level=logging.INFO);run(parser.parse_args())
