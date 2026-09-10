"""Exercise the frozen guard-to-host preflight seam locally with external calls forbidden."""
from pathlib import Path
from unittest.mock import patch
import importlib.util,json,subprocess,sys
sys.dont_write_bytecode=True
root=Path(sys.argv[1]).resolve();here=Path(__file__).resolve().parent

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def forbidden(*a,**kw):raise AssertionError('External process forbidden during CPU preflight')

with patch.object(subprocess,'run',side_effect=forbidden),patch.object(subprocess,'Popen',side_effect=forbidden),patch.object(subprocess,'check_output',side_effect=forbidden):
 guard=load('_bound_canonical_guard',here/'launch_guarded_remote.py')
 hostpath=root/'tmp/canonical_native_query_host_001/launch_query_spark.py'
 host=load('_bound_canonical_host',hostpath)
 with patch.multiple(guard,HOST=hostpath,SOURCE=root/'tmp/updated_native_sdf_query_002',ADMISSION=root/'tmp/canonical_native_inspection_terminal_003/run/inspection',ASSET=root/'artifacts/mkii_updated_2026-09-10/usd_002/rs05_mass_corrected',SUPERVISOR_SOURCE=root/'tmp/reference_physics_adapter_009/source_009',OUTPUT=root/'tmp/canonical_query_guard_uncreated_preflight_001'):
  identity=guard.validate_query_inputs(host)
 assert not any(k in sys.modules for k in ('numpy','torch','isaaclab'))
print(json.dumps({'passed':True,'external_calls':0,'GPU_calls':0,'identity':identity,'host_freeze_sha256':guard.HOST_FREEZE_SHA256,'scope':'Actual frozen inputs and guard-host API; local standard-library preflight only'},indent=2))
