"""Backend/device comparisons in the physical orbital space, not DF frames."""
import argparse
import json
import os
from pathlib import Path
import h5py
import numpy as np
from run_green import complex_data

ENERGY_ATOL=1e-7
ATOL=1e-7
RTOL=1e-6

def array_error(left,right):
    error=np.abs(left-right)
    limit=ATOL+RTOL*np.abs(right)
    return {'max_abs':float(error.max()),'frobenius':float(np.linalg.norm(error)),
            'max_tolerance_ratio':float((error/limit).max()),
            'failed_entries':int(np.count_nonzero(error>limit))}

def compare(left,right):
    a=json.loads((left/'run.json').read_text());b=json.loads((right/'run.json').read_text())
    assert a['status']==b['status']=='PASS' and a['converged'] and b['converged']
    assert a['method']==b['method'] and a['a_angstrom']==b['a_angstrom']
    assert a['binary_sha256']==b['binary_sha256'] and a['grid_sha256']==b['grid_sha256']
    checks={};failures=[]
    for key in ('Energy_1b','Energy_HF','Energy_2b','total_energy_hartree'):
        delta=a['numerical']['last'][key]-b['numerical']['last'][key]
        checks[key]={'difference':delta,'atol':ENERGY_ATOL,'pass':abs(delta)<=ENERGY_ATOL}
        if not checks[key]['pass']: failures.append(key)
    for key in ('mu','electron_number'):
        av=a['numerical']['last']['mu'] if key=='mu' else a['numerical'][key]
        bv=b['numerical']['last']['mu'] if key=='mu' else b['numerical'][key]
        checks[key]={'difference':av-bv,'atol':1e-7,'pass':abs(av-bv)<=1e-7}
        if not checks[key]['pass']: failures.append(key)
    manifest=Path(os.environ['STUDY_RUNS'])/'inputs'/f"a{a['a_angstrom']:.6f}"/'preparation.json'
    prep=json.loads(manifest.read_text())
    with h5py.File(prep['frozen']['ccgdf']['path']) as file:
        overlap=complex_data(file['HF/S-k'])
    eig,vec=np.linalg.eigh(overlap)
    assert eig.min()>0
    root=(vec*np.sqrt(eig)[...,None,:])@vec.conj().swapaxes(-1,-2)
    inverse=(vec/np.sqrt(eig)[...,None,:])@vec.conj().swapaxes(-1,-2)
    matrices={}
    with h5py.File(left/'results.h5') as af,h5py.File(right/'results.h5') as bf:
        ai=int(af['iter'][()]);bi=int(bf['iter'][()])
        np.testing.assert_array_equal(af[f'iter{ai}/G_tau/mesh'][()],bf[f'iter{bi}/G_tau/mesh'][()])
        for label in ('Sigma1','Selfenergy/data','G_tau/data'):
            aa=complex_data(af[f'iter{ai}/{label}']);bb=complex_data(bf[f'iter{bi}/{label}'])
            assert aa.shape==bb.shape
            transform=root if label=='G_tau/data' else inverse
            raw=array_error(aa,bb)
            if aa.ndim==4:
                normalized=array_error(transform@aa@transform,transform@bb@transform)
            else:
                normalized={'max_abs':0.,'frobenius_squared':0.,'max_tolerance_ratio':0.,'failed_entries':0}
                for x,y in zip(aa,bb):
                    err=array_error(transform@x@transform,transform@y@transform)
                    for key in ('max_abs','max_tolerance_ratio'): normalized[key]=max(normalized[key],err[key])
                    normalized['frobenius_squared']+=err['frobenius']**2
                    normalized['failed_entries']+=err['failed_entries']
                normalized['frobenius']=float(np.sqrt(normalized.pop('frobenius_squared')))
            matrices[label]={'raw_ao':raw,'overlap_normalized':normalized}
            if normalized['failed_entries']: failures.append(label)
        early={}
        for label in ('Sigma1','Selfenergy/data','G_tau/data'):
            early[label]=array_error(complex_data(af['iter1/'+label]),complex_data(bf['iter1/'+label]))
    return {'left':str(left),'right':str(right),'method':a['method'],
            'status':'PASS' if not failures else 'FAIL','failures':failures,
            'atol':ATOL,'rtol':RTOL,'scalar_checks':checks,'matrix_checks':matrices,
            'first_iteration_diagnostic':early,'iterations':[a['numerical']['iteration'],b['numerical']['iteration']]}

def volume(a,stage='production'):
    root=Path(os.environ['STUDY_RUNS'])/stage/f'a{a:.6f}'
    pairs=[];missing=[]
    for method,modes in [('gw',['cpu','gpu']),('gf2',['cpu','hybrid'])]:
        paths={(mode,backend):root/f'{method}-{mode}-{backend}' for mode in modes for backend in ('ccgdf','rsgdf')}
        requests=[(paths[mode,'ccgdf'],paths[mode,'rsgdf']) for mode in modes]
        requests +=[(paths[modes[0],backend],paths[modes[1],backend]) for backend in ('ccgdf','rsgdf')]
        for left,right in requests:
            if not all((path/'run.json').exists() and json.loads((path/'run.json').read_text())['status']=='PASS' for path in (left,right)):
                missing.append([str(left),str(right)]);continue
            pairs.append(compare(left,right))
    result={'a_angstrom':a,'status':('FAIL' if any(p['status']=='FAIL' for p in pairs) else 'INCOMPLETE' if missing else 'PASS'),
            'comparisons':pairs,'pending_or_failed_pairs':missing}
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print('VOLUME_COMPARISON',a,result['status'],len(pairs),'comparisons',flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--a',type=float,required=True);parser.add_argument('--stage',default='production')
    opt=parser.parse_args();out=volume(opt.a,opt.stage)
    if out['status']=='FAIL': raise SystemExit(1)
