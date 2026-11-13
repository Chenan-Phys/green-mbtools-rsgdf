"""Converge physical references, then compare matched GREEN continuations.

References use the unmodified main executable. Controlled continuations use
the same unmodified kernel libraries and identical mu/G/Sigma initialization.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from run_controlled import run as compare_continuations

root=Path(os.environ['STUDY_RUNS']);scripts=Path(__file__).resolve().parent
STAGE='canonical-refined'
CONTROLS=json.loads((scripts/'reference-controls.json').read_text())

def reference(a,method,seed):
    case=root/STAGE/f'a{a:.6f}'/f'{method.lower()}-cpu-ccgdf'
    if (case/'run.json').exists():
        record=json.loads((case/'run.json').read_text());assert record['status']=='PASS',case
        return case/'results.h5'
    control=CONTROLS['overrides'].get(f'{method}@{a:.6f}',CONTROLS['default'])
    command=[sys.executable,str(scripts/'run_green.py'),'--a',str(a),'--method',method,'--mode','cpu',
             '--backend','ccgdf','--stage',STAGE,'--density-tolerance',str(control['relative_number_tolerance']),
             '--energy-threshold',str(control['energy_threshold_hartree']),'--mixing-type','SIGMA_MIXING',
             '--mixing-weight','0.15' if method=='GF2' else '0.3',
             '--itermax',str(control['itermax']),'--timeout',str(control['timeout_seconds']),'--seed-results',str(seed)]
    if method=='GF2':command+=['--ranks',os.environ.get('REFERENCE_GF2_RANKS','1')]
    subprocess.run(command,check=True)
    return case/'results.h5'

def point(a,seeds):
    manifest=root/'inputs'/f'a{a:.6f}'/'preparation.json'
    assert json.loads(manifest.read_text())['status']=='PASS'
    next_seeds={method:reference(a,method,seeds[method]) for method in ('GW','GF2')}
    comparison=compare_continuations(a,STAGE,STAGE,'matched-refined')
    return next_seeds,comparison

def branch(values,central_seeds):
    seeds=central_seeds.copy();records=[]
    for a in values:
        try:
            seeds,result=point(a,seeds)
            records.append(result)
            (root/f'branch-{values[0]:.6f}.json').write_text(json.dumps(records,indent=2)+'\n')
            print('CORRELATED_VOLUME_PASS',a,flush=True)
        except Exception as error:
            record={'a_angstrom':a,'status':'FAIL','error':repr(error),'traceback':traceback.format_exc()}
            records.append(record)
            (root/f'branch-{values[0]:.6f}.json').write_text(json.dumps(records,indent=2)+'\n')
            raise
    return records

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--branch',choices=['both','compression','expansion'],default='both')
    options=parser.parse_args()
    initial={method:Path(os.environ[f'INITIAL_{method}_REFERENCE']) for method in ('GW','GF2')}
    assert all(path.exists() for path in initial.values())
    central_seeds,central=point(5.7,initial)
    print('CENTRAL_MATCHED_MATRIX_PASS',json.dumps(central),flush=True)
    branches={'compression':[5.6125,5.525,5.4375,5.35],
              'expansion':[5.7875,5.875,5.9625,6.05]}
    selected=list(branches.values()) if options.branch=='both' else [branches[options.branch]]
    errors=[];completed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futures=[pool.submit(branch,values,central_seeds) for values in selected]
        for future in concurrent.futures.as_completed(futures):
            try: completed+=future.result()
            except Exception as error: errors.append(repr(error))
    ledger='correlated-scan-ledger.json' if options.branch=='both' else f'correlated-scan-{options.branch}-ledger.json'
    (root/ledger).write_text(json.dumps({'central':central,'completed':completed,'errors':errors},indent=2)+'\n')
    subprocess.run([sys.executable,str(scripts/'analyze_eos.py')],check=True)
    assert not errors,errors
    print('CORRELATED_SCAN_EXECUTED',options.branch,flush=True)
