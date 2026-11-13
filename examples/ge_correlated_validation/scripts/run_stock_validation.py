"""Validate matched starts through stock main with its normal density search."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from compare_green import volume

root=Path(os.environ['STUDY_RUNS']);scripts=Path(__file__).resolve().parent

def run(a):
    bridges=[]
    for method,modes in [('GW',['gpu','cpu']),('GF2',['cpu','hybrid'])]:
        reference=root/'canonical-refined'/f'a{a:.6f}'/f'{method.lower()}-cpu-ccgdf'
        baseline=json.loads((reference/'run.json').read_text())
        assert baseline['status']=='PASS' and baseline['converged']
        for mode in modes:
            for backend in ('ccgdf','rsgdf'):
                case=root/'stock-validation'/f'a{a:.6f}'/f'{method.lower()}-{mode}-{backend}'
                if (case/'run.json').exists():
                    result=json.loads((case/'run.json').read_text());assert result['status']=='PASS'
                else:
                    command=[sys.executable,str(scripts/'run_green.py'),'--a',str(a),
                        '--method',method,'--mode',mode,'--backend',backend,'--stage','stock-validation',
                        '--seed-results',str(reference/'results.h5'),'--density-tolerance',str(min(1e-12,baseline['relative_particle_number_tolerance'])),
                        '--energy-threshold','1e-11','--mixing-type','SIGMA_MIXING',
                        '--mixing-weight','0.15' if method=='GF2' else '0.3','--itermax','100','--timeout','1800']
                    subprocess.run(command,check=True)
                    result=json.loads((case/'run.json').read_text())
                assert result['fixed_chemical_potential'] is None
                assert result['consumer_kind']=='unmodified upstream main executable'
                delta=result['numerical']['last']['total_energy_hartree']-baseline['numerical']['last']['total_energy_hartree']
                nerror=result['numerical']['electron_number_error']
                bridge={'method':method,'mode':mode,'backend':backend,
                    'energy_difference_to_canonical_hartree':delta,'electron_number_error':nerror,
                    'mu_difference_to_canonical_hartree':result['numerical']['last']['mu']-baseline['numerical']['last']['mu'],
                    'status':'PASS' if abs(delta)<1e-7 and abs(nerror)<1e-7 else 'FAIL'}
                bridges.append(bridge)
                (case.parent/'canonical-bridges.json').write_text(json.dumps(bridges,indent=2)+'\n')
                assert bridge['status']=='PASS',bridge
    comparison=volume(a,'stock-validation')
    print('STOCK_VOLUME_COMPLETED',a,comparison['status'],flush=True)
    return comparison

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--a',type=float,required=True)
    run(parser.parse_args().a)
