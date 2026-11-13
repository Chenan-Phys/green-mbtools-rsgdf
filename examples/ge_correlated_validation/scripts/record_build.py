"""Record the isolated consumer and dependency revisions without changing them."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

home = Path(os.environ['STUDY_HOME'])
repositories = [home/'green-mbpt', home/'green-gpu', Path(os.environ['FIX_SRC'])]
repositories += sorted((home/'build/_deps').glob('*-src'))
records = []
for root in repositories:
    if not (root/'.git').exists():
        continue
    sha = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    status = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()
    records.append({'path': str(root), 'sha': sha, 'status': status})
    assert not status, (root, status)
executable = home/'build/mbpt.exe'
record = {'repositories': records, 'binary': str(executable),
          'binary_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
          'old_installed_binary_sha256': (hashlib.sha256(Path(os.environ['PRODUCTION_BINARY']).read_bytes()).hexdigest() if os.environ.get('PRODUCTION_BINARY') else None)}
(home/'evidence/build-provenance.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2), flush=True)
