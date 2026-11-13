"""Require unchanged physical outputs before adopting another MPI layout."""
import json
import os
from pathlib import Path
from compare_green import compare
root=Path(os.environ['STUDY_RUNS'])
left=root/'canonical-refined/a5.700000/gf2-cpu-ccgdf'
right=root/'canonical-mpi-check/a5.700000/gf2-cpu-ccgdf'
result=compare(left,right)
(right.parent/'comparison-to-one-rank.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['status']=='PASS',result['failures']
print('GF2_MPI_REFERENCE_CHECK_PASS',flush=True)
