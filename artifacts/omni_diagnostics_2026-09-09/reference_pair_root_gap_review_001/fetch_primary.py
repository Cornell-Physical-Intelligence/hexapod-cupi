"""Read installed primary Python source via SSH; no SDK imports or GPU calls."""
from pathlib import Path
import datetime, hashlib, json, subprocess
ROOT = Path(__file__).resolve().parent
paths = [
 '/home/orionh/IsaacLab/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py',
 '/home/orionh/IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py',
 '/home/orionh/IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/kernels.py',
]
script = "from pathlib import Path\nimport json\npaths=" + repr(paths) + "\nprint(json.dumps({p:Path(p).read_text() for p in paths}))\n"
r = subprocess.run(['ssh', 'spark', 'python3', '-'], input=script, text=True, capture_output=True, check=True)
sources = json.loads(r.stdout)
out = ROOT/'installed_primary'; out.mkdir(exist_ok=True)
receipt = {'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'scope':'Installed Python getters and conversion kernels; native PhysX solver internals not inspected.', 'sources':[]}
for i, path in enumerate(paths):
 name = ('base_articulation_data.py','physx_articulation_data.py','physx_shared_kernels.py')[i]
 data = sources[path].encode(); (out/name).write_bytes(data)
 receipt['sources'].append({'remote_path':path,'local_path':'installed_primary/'+name,'sha256':hashlib.sha256(data).hexdigest()})
(ROOT/'PRIMARY_SOURCE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
