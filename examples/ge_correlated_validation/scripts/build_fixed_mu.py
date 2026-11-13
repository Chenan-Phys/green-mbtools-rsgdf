"""Compile a public-API diagnostic against the exact existing static libraries."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

home=Path(os.environ['STUDY_HOME']);build=home/'build'
folder=home/'fixed-mu-harness';folder.mkdir(exist_ok=True)
source=home/'scripts/fixed_mu.cpp';obj=folder/'fixed_mu.cpp.o';binary=folder/'fixed-mu.exe'
flags={}
for line in (build/'CMakeFiles/mbpt.exe.dir/flags.make').read_text().splitlines():
    if ' = ' in line:
        key,value=line.split(' = ',1);flags[key]=value
link=shlex.split((build/'CMakeFiles/mbpt.exe.dir/link.txt').read_text())
compiler=link[0]
compile_command=[compiler]+sum([shlex.split(flags[key]) for key in ('CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS')],[])
compile_command += ['-c',str(source),'-o',str(obj)]
link_command=[]
for item in link:
    if item=='CMakeFiles/mbpt.exe.dir/main.cpp.o': item=str(obj)
    elif item=='mbpt.exe': item=str(binary)
    elif item.startswith('-Wl,--dependency-file='): item='-Wl,--dependency-file='+str(folder/'link.d')
    link_command.append(item)
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
main_before=digest(build/'mbpt.exe')
subprocess.run(compile_command,cwd=build,check=True)
subprocess.run(link_command,cwd=build,check=True)
assert digest(build/'mbpt.exe')==main_before
libraries={item:digest(build/item if not item.startswith('/') else item)
           for item in link if item.endswith('.a') and 'lib' in item}
result={'binary':str(binary),'binary_sha256':digest(binary),'source':str(source),
        'source_sha256':digest(source),'upstream_main_binary_sha256':main_before,
        'compile_command':compile_command,'link_command':link_command,
        'static_library_sha256':libraries,'upstream_sources_modified':False,
        'purpose':'Set the same public Dyson mu() value for all paired cases; existing SC loop and all HF/GF2/GW CPU/GPU kernels remain unchanged.'}
(home/'evidence/fixed-mu-build.json').write_text(json.dumps(result,indent=2)+'\n')
print('FIXED_MU_HARNESS_READY',result['binary_sha256'],flush=True)
