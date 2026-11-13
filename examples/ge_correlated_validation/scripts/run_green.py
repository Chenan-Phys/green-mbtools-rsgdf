"""One bounded, audited GREEN correlated calculation with truthful dispatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import time

import h5py
import numpy as np

def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(8*1024**2): result.update(block)
    return result.hexdigest()

def complex_data(ds):
    data=ds[()]
    return data if np.iscomplexobj(data) else data.view(np.complex128).reshape(data.shape[:-1])

def memory_tree(root_pid):
    rows={}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            info=(entry/'stat').read_text().rsplit(')',1)[1].split()
            rows[int(entry.name)]=(int(info[1]),int(info[21])*os.sysconf('SC_PAGE_SIZE'))
        except (OSError,ValueError,IndexError): pass
    selected={root_pid}
    while True:
        expanded=selected|{pid for pid,(parent,_) in rows.items() if parent in selected}
        if expanded==selected: break
        selected=expanded
    return sum(rows.get(pid,(0,0))[1] for pid in selected)

def summarize(path, input_file):
    with h5py.File(path) as result,h5py.File(input_file) as inp:
        iteration=int(result['iter'][()])
        history=[]
        for index in range(1,iteration+1):
            if f'iter{index}/Energy_HF' not in result: continue
            group=result[f'iter{index}']
            row={key:float(group[key][()]) for key in ('Energy_1b','Energy_HF','Energy_2b','mu')}
            row.update(iteration=index,total_energy_hartree=row['Energy_HF']+row['Energy_2b'])
            history.append(row)
        assert len(history)>=1
        group=result[f'iter{iteration}']
        overlap=complex_data(inp['HF/S-k'])
        endpoint=complex_data(group['G_tau/data'])[-1]
        ns=int(inp['params/ns'][()]);nao=int(inp['params/nao'][()]);nso=int(inp['params/nso'][()])
        spin=2 if ns==1 and nao==nso else 1
        assert overlap.shape==endpoint.shape
        nk=int(inp['params/nk'][()])
        assert int(inp['symmetry/k/ink'][()])==nk,'This study uses the full k mesh'
        electrons=float(-spin*np.einsum('skij,skji->',overlap,endpoint).real/nk)
        target=float(inp['params/nel_cell'][()])
        delta=None
        if len(history)>1:
            delta=sum(abs(history[-1][key]-history[-2][key]) for key in ('Energy_1b','Energy_HF','Energy_2b'))
        return {'iteration':iteration,'history':history,'last':history[-1],
                'independent_energy_change_hartree':delta,'electron_number':electrons,
                'target_electron_number':target,'electron_number_error':electrons-target,
                'G_shape':list(group['G_tau/data'].shape),'spin_prefactor':spin}

def compact(raw, output, summary):
    keep={1,max(1,summary['iteration']-1),summary['iteration']}
    with h5py.File(raw) as source,h5py.File(output,'w') as target:
        for key,value in source.attrs.items(): target.attrs[key]=value
        for key in source:
            if not re.fullmatch(r'iter\d+',key) or int(key[4:]) in keep:
                source.copy(key,target)
        target.attrs['study_retained_matrix_iterations']=json.dumps(sorted(keep))
        target.attrs['study_full_scalar_history']='run.json: numerical.history'
    # Reopen every retained matrix before removing this run's uncompressed raw
    # history. Scalar histories for every iteration remain in the JSON record.
    with h5py.File(raw) as source,h5py.File(output) as target:
        for index in keep:
            if f'iter{index}' not in source: continue
            for key in ('Sigma1','Selfenergy/data','G_tau/data'):
                a=source[f'iter{index}/{key}'];b=target[f'iter{index}/{key}']
                assert a.shape==b.shape and a.dtype==b.dtype
                if a.ndim:
                    for i in range(a.shape[0]): np.testing.assert_array_equal(a[i],b[i])
                else: np.testing.assert_array_equal(a[()],b[()])
    raw.unlink()
    return sorted(keep)

def run(options):
    home=Path(os.environ['STUDY_HOME']);root=Path(os.environ['STUDY_RUNS'])
    if options.stage=='production' and (root/'superseded-loose-number-tolerance.json').exists():
        raise RuntimeError('Loose-number-tolerance phase is superseded; preserve it and use production-tight')
    preparation=json.loads((root/'inputs'/f'a{options.a:.6f}'/'preparation.json').read_text())
    assert preparation['status']=='PASS'
    backend=options.backend
    input_file=Path(preparation['frozen'][backend]['path'])
    assert digest(input_file)==preparation['frozen'][backend]['sha256']
    converted=preparation['converted'][backend]
    manifest='fixed-mu-build.json' if options.fixed_mu is not None else 'build-provenance.json'
    build=json.loads((home/'evidence'/manifest).read_text())
    executable=Path(build['binary'])
    assert digest(executable)==build['binary_sha256']
    with h5py.File(input_file) as file: assert file.attrs['__green_version__']=='1.1.0'
    case=root/options.stage/f'a{options.a:.6f}'/f'{options.method.lower()}-{options.mode}-{backend}'
    case.mkdir(parents=True,exist_ok=False)
    kernel='GPU' if options.mode in ('gpu','hybrid') else 'CPU'
    assert (options.method=='GF2' and options.mode in ('cpu','hybrid')) or (options.method=='GW' and options.mode in ('cpu','gpu'))
    dispatch=('GPU HF + CPU GF2 correlation' if options.mode=='hybrid' else
              'GPU HF + GPU GW correlation' if options.mode=='gpu' else f'CPU HF + CPU {options.method} correlation')
    grid=Path(os.environ['GREEN_GRID_FILE'])
    raw=case/'results-raw.h5'
    seed_record=None
    if options.seed_results is not None:
        seed=options.seed_results.resolve()
        with h5py.File(seed) as source,h5py.File(raw,'w') as target:
            index=int(source['iter'][()])
            for key,value in source.attrs.items(): target.attrs[key]=value
            source.copy(f'iter{index}',target,name='iter0')
            target['iter']=np.uint64(0)
        seed_record={'path':str(seed),'sha256':digest(seed),'source_iteration':index,
                     'restart_iteration_rebased_to_zero':True}
    command=[str(executable),'--input_file',str(input_file),
             '--dfintegral_hf_file',converted['hf_integrals'],'--dfintegral_file',converted['correlation_integrals'],
             '--scf_type',options.method,'--kernel',kernel,'--jobs','SC','--BETA','1000',
             '--grid_file',str(grid),'--itermax',str(options.itermax),'--threshold',str(options.energy_threshold),'--tol',str(options.density_tolerance),
             '--restart',str(options.seed_results is not None).lower(),'--diis_restart','false','--results_file',str(raw),'--diis_file',str(case/'diis.h5'),
             '--mixing_type',options.mixing_type,'--diis_start',str(options.diis_start),'--diis_size','8','--mixing_weight',str(options.mixing_weight),
             '--P_sp','false','--Sigma_sp','false','--cuda_low_cpu_memory','true','--cuda_low_gpu_memory','true']
    if options.fixed_mu is not None:
        command += ['--benchmark_mu',str(options.fixed_mu),'--const_density','false']
    ranks=options.ranks or (4 if options.method=='GW' and options.mode=='cpu' else 1)
    if ranks>1: command=[shutil.which('mpirun'),'--bind-to','none','-np',str(ranks)]+command
    execution_environment=os.environ.copy()
    if ranks>1: execution_environment['OMP_NUM_THREADS']='1'
    record={'status':'RUNNING','a_angstrom':options.a,'volume_angstrom3':options.a**3/4,
            'method':options.method,'mode':options.mode,'backend':backend,'dispatch':dispatch,
            'model':preparation.get('model','all-electron sfX2C Ge'),'basis':preparation['basis'],'nao':preparation['nao'],
            'command':command,'input_sha256':digest(input_file),'binary_sha256':build['binary_sha256'],
            'grid_sha256':digest(grid),'beta_hartree_inverse':1000,'temperature_kelvin':315.775024804,
            'stage':options.stage,'itermax':options.itermax,'timeout_seconds':options.timeout,
            'relative_particle_number_tolerance':options.density_tolerance,
            'energy_threshold_hartree':options.energy_threshold,
            'mixing_type':options.mixing_type,'mixing_weight':options.mixing_weight,
            'fixed_chemical_potential':options.fixed_mu,
            'consumer_kind':'public-API fixed-mu diagnostic with unchanged kernels' if options.fixed_mu is not None else 'unmodified upstream main executable',
            'correlated_initial_state':seed_record,
            'mpi_ranks':ranks,'omp_threads_per_rank':int(execution_environment.get('OMP_NUM_THREADS','1')),
            'energy_kind':'GREEN correlated internal energy including nuclear repulsion',
            'peak_process_tree_rss_bytes':0,'converged':False}
    (case/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    start=time.monotonic();reason=None;process=None
    assert shutil.disk_usage(root).free>80*1024**3,'Free scratch below 80 GiB'
    with (case/'run.log').open('w') as log:
        process=subprocess.Popen(command,cwd=case,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=execution_environment)
        record['pid']=process.pid
        (case/'run.json').write_text(json.dumps(record,indent=2)+'\n')
        while process.poll() is None:
            rss=memory_tree(process.pid)
            record['peak_process_tree_rss_bytes']=max(record['peak_process_tree_rss_bytes'],rss)
            if rss>14*1024**3: reason='process-tree RSS exceeded 14 GiB'
            if shutil.disk_usage(root).free<50*1024**3: reason='scratch free space below 50 GiB'
            if time.monotonic()-start>options.timeout: reason='wall-time limit'
            if reason:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                break
            time.sleep(2)
    record.update(exit_code=process.returncode,wall_seconds=time.monotonic()-start,stop_reason=reason)
    text=(case/'run.log').read_text(errors='replace')
    record['explicit_convergence_message']='Simulation Converged' in text
    leakages={}
    for name,value in re.findall(r'Leakage of ([^:\n]+):\s*([0-9.eE+\-]+)',text):
        try: leakages[name]=float(value)
        except ValueError: pass
    record['last_transform_leakages']=leakages
    try:
        if raw.exists(): record['numerical']=summarize(raw,input_file)
        values=record.get('numerical',{})
        residual=values.get('independent_energy_change_hartree')
        converged=(process.returncode==0 and not reason and record['explicit_convergence_message']
                   and residual is not None and residual<options.energy_threshold and abs(values['electron_number_error'])<1e-7)
        if options.fixed_mu is not None:
            assert all(row['mu']==options.fixed_mu for row in values['history']),'Chemical potential changed in controlled run'
        record['converged']=converged
        if 'pilot' in options.stage: record['status']='PILOT_COMPLETE' if process.returncode==0 and not reason else 'PILOT_LIMIT_OR_FAILURE'
        else: record['status']='PASS' if converged else 'NOT_CONVERGED_OR_FAILED'
        if process.returncode==0 and raw.exists():
            record['retained_matrix_iterations']=compact(raw,case/'results.h5',values)
            record['results_sha256']=digest(case/'results.h5')
    except Exception as error:
        record['summary_error']=repr(error)
        record['status']='SUMMARY_FAILED'
    (case/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    print('GREEN_CASE',options.a,options.method,options.mode,backend,record['status'],record['wall_seconds'],flush=True)
    return record

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--a',type=float,required=True)
    parser.add_argument('--method',choices=['GF2','GW'],required=True)
    parser.add_argument('--mode',choices=['cpu','gpu','hybrid'],required=True)
    parser.add_argument('--backend',choices=['ccgdf','rsgdf'],required=True)
    parser.add_argument('--stage',default='production')
    parser.add_argument('--itermax',type=int,default=100)
    parser.add_argument('--timeout',type=float,default=7200)
    parser.add_argument('--density-tolerance',type=float,default=1e-12)
    parser.add_argument('--ranks',type=int,default=None)
    parser.add_argument('--fixed-mu',type=float,default=None)
    parser.add_argument('--mixing-type',default='CDIIS')
    parser.add_argument('--mixing-weight',type=float,default=0.3)
    parser.add_argument('--diis-start',type=int,default=2)
    parser.add_argument('--energy-threshold',type=float,default=1e-9)
    parser.add_argument('--seed-results',type=Path)
    options=parser.parse_args()
    outcome=run(options)
    if outcome['status'] not in ('PASS','PILOT_COMPLETE'): raise SystemExit(1)
