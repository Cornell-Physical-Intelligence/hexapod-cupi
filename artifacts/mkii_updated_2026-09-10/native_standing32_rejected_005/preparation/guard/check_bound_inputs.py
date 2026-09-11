"""Exercise the frozen guard-to-host preflight seam locally with external calls forbidden."""
from pathlib import Path
from unittest.mock import patch
import importlib.util,json,subprocess,sys
sys.dont_write_bytecode=True
root=Path(sys.argv[1]).resolve();here=Path(__file__).resolve().parent
standing_input=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else root/'tmp/canonical_native_standing_terminal_006/run/standing'

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def forbidden(*a,**kw):raise AssertionError('External process forbidden during CPU preflight')

with patch.object(subprocess,'run',side_effect=forbidden),patch.object(subprocess,'Popen',side_effect=forbidden),patch.object(subprocess,'check_output',side_effect=forbidden):
 guard=load('_bound_canonical_guard',here/'launch_guarded_remote.py')
 hostpath=root/'tmp/canonical_native_standing_host_007/launch_standing_spark.py'
 host=load('_bound_canonical_host',hostpath)
 with patch.multiple(guard,HOST=hostpath,SOURCE=root/'tmp/updated_native_standing_005',ADMISSION=root/'tmp/canonical_native_actuation_terminal_001/run/actuation',ASSET=root/'artifacts/mkii_updated_2026-09-10/usd_002/rs05_mass_corrected',SUPERVISOR_SOURCE=root/'tmp/reference_physics_adapter_009/source_009',OUTPUT=root/'tmp/canonical_standing32_guard_uncreated_preflight_004',STANDING_ONE=standing_input):
  captured=json.loads((here/'inputs/reservation_readback.json').read_text())['files']
  localpins={str(here/'inputs/reservation_files'/str(i)):v['sha256'] for i,(name,v) in enumerate(captured.items())}
  with patch.object(guard,'verify_reservation',return_value={'reservation_files_independently_hashed_locally':True}):
   for path,bound in localpins.items():assert guard.sha(Path(path))==bound
   identity=guard.validate_standing_inputs(host)
 assert not any(k in sys.modules for k in ('numpy','torch','isaaclab'))
print(json.dumps({'passed':True,'external_calls':0,'GPU_calls':0,'identity':identity,'host_freeze_sha256':guard.HOST_FREEZE_SHA256,'scope':'Actual frozen inputs and guard-host API; local standard-library preflight only'},indent=2))
