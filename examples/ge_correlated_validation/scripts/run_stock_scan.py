"""Follow accepted references with the stock executable, one volume at a time."""
import json
import os
from pathlib import Path
import time
from analyze_eos import analyze
from run_stock_validation import run

root=Path(os.environ['STUDY_RUNS']);started=time.monotonic()
initial=[5.35,5.4375,5.525,5.6125,5.7,5.7875,5.875,5.9625,6.05]
while True:
    window=root/'eos-window.json'
    values=json.loads(window.read_text())['a_angstrom'] if window.exists() else initial
    completed=[]
    for a in values:
        folder=root/'stock-validation'/f'a{a:.6f}'
        check=folder/'comparison.json'
        if check.exists():
            record=json.loads(check.read_text());assert record['status']=='PASS',record
            completed.append(a);continue
        primary=root/'matched-refined'/f'a{a:.6f}'/'comparison.json'
        if not primary.exists():continue
        assert json.loads(primary.read_text())['status']=='PASS'
        # Wait for active owned cases, but resume a partially completed volume.
        existing=[json.loads(p.read_text()) for p in folder.glob('*/run.json')]
        if any(r['status']=='RUNNING' for r in existing):continue
        assert all(r['status']=='PASS' for r in existing),folder
        record=run(a);assert record['status']=='PASS',record
        completed.append(a)
    summary=root/'analysis/correlated-eos-summary.json'
    primary=json.loads(summary.read_text()) if summary.exists() else {}
    if primary.get('status')=='PASS' and len(completed)==len(values):
        result=analyze('stock-validation')
        assert result['status']=='PASS',result['status']
        (root/'stock-validation-ledger.json').write_text(json.dumps({'status':'PASS','a_angstrom':values,
            'stock_point_count':result['point_count'],'fixed_mu':False},indent=2)+'\n')
        print('STOCK_SCAN_PASS',result['point_count'],flush=True)
        break
    if time.monotonic()-started>10800:raise TimeoutError('Stock validation remains incomplete')
    time.sleep(10)
