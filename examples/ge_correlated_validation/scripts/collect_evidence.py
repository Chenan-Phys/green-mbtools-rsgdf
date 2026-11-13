"""Archive small scientific evidence; leave large calculation arrays remote."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

home=Path(os.environ['STUDY_HOME']);runs=Path(os.environ['STUDY_RUNS'])
compact=runs/'compact-gth-szv-2x2x2-v2'
target=home/'evidence/study-evidence.tar.gz'
items=[];seen=set()
def add(path,name):
    if name in seen:return
    seen.add(name)
    items.append((path,name))
for label in ('analysis','analysis-stock'):
    for path in sorted((compact/label).glob('*')):
        if path.is_file(): add(path,label+'/'+path.name)
for path in sorted(home.joinpath('scripts').glob('*')):
    if path.is_file() and path.suffix in ('.py','.cpp','.sh','.json'):add(path,'scripts/'+path.name)
for path in sorted(home.joinpath('evidence').glob('*')):
    if path.is_file() and path.suffix in ('.json','.log') and path.name!='study-evidence-manifest.json':
        add(path,'workstation/'+path.name)
for path in sorted(compact.rglob('*')):
    if path.is_file() and path.suffix in ('.json','.csv','.log'):
        add(path,'runs/compact/'+path.relative_to(compact).as_posix())
for folder in (runs/'inputs',runs/'all-electron-pilot'):
    if not folder.exists():continue
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.log','.csv'):
            add(path,'runs/all-electron/'+path.relative_to(runs).as_posix())
manifest=[]
for path,name in items:
    manifest.append({'archive_path':name,'source_path':str(path),'bytes':path.stat().st_size,
                     'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
provenance=[]
for path in (Path(os.environ['FIX_SRC']),home/'green-mbpt',home/'green-gpu'):
    provenance.append({'path':str(path),'sha':subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),
        'status':subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip()})
assert not any(p['status'] for p in provenance),provenance
record={'files':manifest,'repositories':provenance,'large_arrays_retained_on_workstation':True}
manifest_path=home/'evidence/study-evidence-manifest.json'
manifest_path.write_text(json.dumps(record,indent=2)+'\n')
with tarfile.open(target,'w:gz') as archive:
    archive.add(manifest_path,arcname='manifest.json')
    for path,name in items:archive.add(path,arcname=name,recursive=False)
print('EVIDENCE_ARCHIVE',target,target.stat().st_size,hashlib.sha256(target.read_bytes()).hexdigest(),flush=True)
