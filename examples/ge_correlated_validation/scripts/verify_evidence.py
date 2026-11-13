"""Extract the small evidence archive only after checking paths and hashes."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile

parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('output',type=Path)
options=parser.parse_args();root=options.output.resolve();root.mkdir(parents=True,exist_ok=True)
with tarfile.open(options.archive,'r:gz') as archive:
    members={m.name:m for m in archive.getmembers()}
    assert len(members)==len(archive.getmembers()),'Duplicate archive names'
    manifest=json.load(archive.extractfile('manifest.json'))
    expected={r['archive_path']:r for r in manifest['files']}
    assert set(members)==set(expected)|{'manifest.json'}
    assert not any(r['status'] for r in manifest['repositories'])
    for name,member in members.items():
        posix=PurePosixPath(name)
        assert not posix.is_absolute() and '..' not in posix.parts and ':' not in name and '\\' not in name
        assert member.isfile() and not member.issym() and not member.islnk()
        path=(root/Path(*posix.parts)).resolve();assert path.is_relative_to(root)
        data=archive.extractfile(member).read()
        if name!='manifest.json':
            record=expected[name]
            assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],name
        if path.exists():assert path.read_bytes()==data,f'Refusing to overwrite changed evidence: {path}'
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
record={'status':'PASS','archive_sha256':hashlib.sha256(options.archive.read_bytes()).hexdigest(),
        'verified_files':len(expected),'output':str(root),'repositories':manifest['repositories']}
(root.parent/'evidence-verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
