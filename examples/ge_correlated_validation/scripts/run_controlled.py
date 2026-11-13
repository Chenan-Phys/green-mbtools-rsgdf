"""Compare all existing GREEN kernels at the same reference mu, then check N/E."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from compare_green import volume

scripts=Path(__file__).resolve().parent
root=Path(os.environ['STUDY_RUNS'])

def run(a,reference_stage='production-tight',gf2_reference_stage=None,stage='continued'):
    bridges=[]
    for method,modes in [('GW',['gpu','cpu']),('GF2',['cpu','hybrid'])]:
        chosen_stage=gf2_reference_stage if method=='GF2' and gf2_reference_stage else reference_stage
        reference=root/chosen_stage/f'a{a:.6f}'/f'{method.lower()}-cpu-ccgdf'/'run.json'
        started=time.monotonic()
        while not reference.exists() or json.loads(reference.read_text())['status']=='RUNNING':
            if time.monotonic()-started>7200: raise TimeoutError(f'Canonical reference not ready: {reference}')
            time.sleep(5)
        baseline=json.loads(reference.read_text())
        assert baseline['status']=='PASS' and baseline['converged'],reference
        mu=baseline['numerical']['last']['mu']
        for mode in modes:
            for backend in ('ccgdf','rsgdf'):
                case=root/stage/f'a{a:.6f}'/f'{method.lower()}-{mode}-{backend}'
                if (case/'run.json').exists():
                    record=json.loads((case/'run.json').read_text())
                    assert record['status']=='PASS' and record['fixed_chemical_potential']==mu
                else:
                    command=[sys.executable,str(scripts/'run_green.py'),'--a',str(a),'--method',method,
                             '--mode',mode,'--backend',backend,'--stage',stage,'--fixed-mu',str(mu),
                             '--density-tolerance','1e-12','--itermax','100','--timeout','3600',
                             '--seed-results',str(reference.parent/'results.h5')]
                    if method=='GF2':
                        command += ['--mixing-type','SIGMA_MIXING','--mixing-weight','0.15','--itermax','300']
                    elif stage in ('matched-linear','matched-refined'):
                        command += ['--mixing-type','SIGMA_MIXING','--mixing-weight','0.3']
                    subprocess.run(command,check=True)
                    record=json.loads((case/'run.json').read_text())
                difference=record['numerical']['last']['total_energy_hartree']-baseline['numerical']['last']['total_energy_hartree']
                bridge={'method':method,'mode':mode,'backend':backend,'reference':str(reference),
                        'fixed_mu_hartree':mu,'energy_difference_to_canonical_hartree':difference,
                        'electron_number_error':record['numerical']['electron_number_error'],
                        'status':'PASS' if abs(difference)<1e-7 and abs(record['numerical']['electron_number_error'])<1e-7 else 'FAIL'}
                bridges.append(bridge)
                (root/stage/f'a{a:.6f}'/'canonical-bridges.json').write_text(json.dumps(bridges,indent=2)+'\n')
                assert bridge['status']=='PASS',bridge
    checks=volume(a,stage)
    assert checks['status']=='PASS',checks['status']
    return {'a_angstrom':a,'status':'PASS','canonical_bridges':bridges,'comparison_count':len(checks['comparisons'])}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--a',type=float,required=True)
    parser.add_argument('--reference-stage',default='production-tight');parser.add_argument('--gf2-reference-stage')
    parser.add_argument('--stage',default='continued')
    options=parser.parse_args()
    result=run(options.a,options.reference_stage,options.gf2_reference_stage,options.stage);print('CONTROLLED_VOLUME_COMPLETE',result,flush=True)
