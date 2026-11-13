"""Finish the common grid and add a bounded, predeclared bracket if needed."""
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from analyze_eos import analyze
from prepare_inputs import prepare
from run_correlated_scan import point

root=Path(os.environ['STUDY_RUNS'])
initial=[5.35,5.4375,5.525,5.6125,5.7,5.7875,5.875,5.9625,6.05]
start=time.monotonic()
while True:
    ready=True
    for a in initial:
        path=root/'matched-refined'/f'a{a:.6f}'/'comparison.json'
        if not path.exists():ready=False;continue
        record=json.loads(path.read_text())
        assert record['status']=='PASS',record
    if ready:break
    if time.monotonic()-start>10800:raise TimeoutError('Initial common window remains incomplete')
    time.sleep(10)

window=root/'eos-window.json'
if window.exists():
    config=json.loads(window.read_text())
    extra_lower=sorted([a for a in config['added_a_angstrom'] if a<initial[0]],reverse=True)
    extra_upper=sorted(a for a in config['added_a_angstrom'] if a>initial[-1])
else:
    result=analyze()
    assert not result['missing_or_failed']
    assert not result['physical_comparison_failures'] and not result['canonical_bridge_failures']
    lower=upper=False
    for fit in result['fits']:
        values=[p['total_energy_hartree'] for p in fit['points']]
        index=int(np.argmin(values))
        lower=lower or index==0
        upper=upper or index==len(values)-1
    extra_lower=[5.2625,5.175] if lower else []
    extra_upper=[6.1375,6.225] if upper else []
if not window.exists() and (extra_lower or extra_upper):
    shutil.copy2(root/'analysis/correlated-eos-summary.json',root/'analysis/initial-window-summary.json')
    config={'a_angstrom':sorted(initial+extra_lower+extra_upper),
        'reason':'Initial common window did not bracket every method minimum; bounded extension uses the same points for all backends/devices.',
        'initial_a_angstrom':initial,'added_a_angstrom':sorted(extra_lower+extra_upper),
        'predeclared_extension_step_angstrom':0.0875,'maximum_added_points_per_edge':2,
        'gf2_reference_mpi_ranks':int(os.environ.get('REFERENCE_GF2_RANKS','1')),
        'acceptance_bounds_changed':False}
    (root/'eos-window.json').write_text(json.dumps(config,indent=2)+'\n')
    print('COMMON_WINDOW_EXTENSION',json.dumps(config),flush=True)
for values,seed_a in ((extra_lower,initial[0]),(extra_upper,initial[-1])):
    seeds={m:root/'canonical-refined'/f'a{seed_a:.6f}'/f'{m.lower()}-cpu-ccgdf/results.h5' for m in ('GW','GF2')}
    for a in values:
        prepare(a)
        seeds,comparison=point(a,seeds)
result=analyze()
assert result['status']=='PASS',result['status']
print('COMMON_WINDOW_QUALIFIED',result['point_count'],flush=True)
