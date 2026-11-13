"""Produce, validate, migrate and freeze matched inputs for one Ge volume."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import h5py
import numpy as np
from upstream_migrate import migrate

SCRIPTS=Path(__file__).resolve().parent

def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(8*1024**2): result.update(block)
    return result.hexdigest()

def complex_data(dataset):
    data=dataset[()]
    if np.iscomplexobj(data): return data
    assert data.dtype==np.float64 and data.shape[-1]==2
    return data.view(np.complex128).reshape(data.shape[:-1])

def prepare(a):
    root=Path(os.environ['STUDY_RUNS'])/'inputs'/f'a{a:.6f}'
    root.mkdir(parents=True,exist_ok=True)
    completed=root/'preparation.json'
    if completed.exists():
        old=json.loads(completed.read_text())
        assert old['status']=='PASS'
        for backend,item in old['frozen'].items():
            assert digest(item['path'])==item['sha256']
        return old
    for backend in ('ccgdf','rsgdf'):
        target=root/('ge-'+backend)
        if (target/'generation.json').exists(): continue
        assert not target.exists(),f'Preserving incomplete export: {target}'
        assert shutil.disk_usage(root).free>80*1024**3,'Free scratch below 80 GiB'
        assert os.environ.get('STUDY_MODEL')=='compact', 'This portable package covers the compact model'
        exporter='export_compact.py'
        cmd=[sys.executable,str(SCRIPTS/exporter),'--a',str(a),'--backend',backend,'--output',str(target)]
        with (root/f'{backend}-export.log').open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=5400)
    checker=Path(os.environ['FIX_SRC'])/'examples/rsgdf_validation/compare-fixtures.py'
    with (root/'interaction-validation.log').open('w') as log:
        subprocess.run([sys.executable,str(checker),str(root),'--fixtures','ge'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    native=root/'native'
    native.mkdir(exist_ok=True)
    converted={}
    for backend in ('ccgdf','rsgdf'):
        legacy=root/('ge-'+backend)
        folder=native/backend
        folder.mkdir(exist_ok=False)
        archives=[]
        for label in ('df_hf_int','df_int'):
            output=folder/label
            output.mkdir()
            for path in (legacy/label).iterdir():
                if path.name=='meta.h5': shutil.copy2(path,output/path.name)
                else: os.link(path,output/path.name)
            archives.append(str(output))
        original_input_sha=digest(legacy/'input.h5')
        migrate(str(legacy/'input.h5'),output=str(folder/'input.h5'),int_paths=archives,
                dm_file=str(legacy/'dm.h5'),target='1.1.0')
        assert digest(legacy/'input.h5')==original_input_sha
        with h5py.File(legacy/'input.h5') as old,h5py.File(folder/'input.h5') as new:
            assert new.attrs['__green_version__']=='1.1.0'
            for key in ('HF/H-k','HF/S-k','HF/Fock-k'):
                np.testing.assert_array_equal(complex_data(old[key]),complex_data(new[key]))
        converted[backend]={'input':str(folder/'input.h5'),'hf_integrals':archives[0],
                            'correlation_integrals':archives[1],'legacy_input_sha256':original_input_sha,
                            'migration_preserved_one_body_arrays_bitwise':True}
    differences={}
    with h5py.File(converted['ccgdf']['input']) as cc,h5py.File(converted['rsgdf']['input']) as rs:
        for key,tolerance in [('HF/H-k',1e-9),('HF/S-k',1e-13)]:
            left,right=complex_data(cc[key]),complex_data(rs[key])
            differences[key]=float(abs(left-right).max())
            np.testing.assert_allclose(left,right,atol=tolerance,rtol=0)
        differences['seed_Fock_max_abs']=float(abs(complex_data(cc['HF/Fock-k'])-complex_data(rs['HF/Fock-k'])).max())
        np.testing.assert_allclose(cc['HF/Energy_nuc'][()],rs['HF/Energy_nuc'][()],atol=1e-10,rtol=0)
    frozen={}
    for backend in ('ccgdf','rsgdf'):
        out=native/(backend+'-frozen.h5')
        shutil.copy2(converted[backend]['input'],out)
        if backend=='rsgdf':
            with h5py.File(converted['ccgdf']['input']) as cc,h5py.File(out,'a') as rs:
                del rs['HF']
                cc.copy('HF',rs)
        frozen[backend]={'path':str(out),'sha256':digest(out)}
    with h5py.File(frozen['ccgdf']['path']) as cc,h5py.File(frozen['rsgdf']['path']) as rs:
        for key in ('HF/H-k','HF/S-k','HF/Fock-k','HF/Energy_nuc'):
            np.testing.assert_array_equal(cc[key][()],rs[key][()])
    generated=json.loads((root/'ge-ccgdf/generation.json').read_text())
    result={'status':'PASS','a_angstrom':a,'volume_angstrom3':a**3/4,
            'basis':generated['basis'],'model':generated.get('model','all-electron sfX2C Ge'),
            'nao':generated['nao'],'nk':generated['nk'],'naux':generated['naux'],
            'converted':converted,'frozen':frozen,'independent_export_differences':differences,
            'interaction_validation':json.loads((root/'fixture-comparison.json').read_text()),
            'identical_starting_one_body_and_hf_state':True,
            'migration_source':json.loads((SCRIPTS/'upstream-migration-source.json').read_text())}
    completed.write_text(json.dumps(result,indent=2)+'\n')
    print('INPUTS_READY',a,flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--a',type=float,required=True)
    prepare(p.parse_args().a)
