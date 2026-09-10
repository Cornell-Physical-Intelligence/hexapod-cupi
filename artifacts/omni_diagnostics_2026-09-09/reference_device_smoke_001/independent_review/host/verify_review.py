"""Read-only independent frozen-host inventory and tiny successor-delta receipt."""
from pathlib import Path
import hashlib,json,difflib
HERE=Path(__file__).resolve().parent;TMP=HERE.parent
PINNED={'reference_device_smoke_launch_001':'139f9680aa5a92f5694fbd0350d22945e36da3d715e78ee113b220985d519a80','reference_device_smoke_launch_002':'adbb60b3d77b8d6b268dcf3060232fcf3c493361259a0997582e694d62a35658'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    verified={}
    for name,expected in PINNED.items():
        p=TMP/name;assert sha(p/'FREEZE_SHA256.json')==expected
        inventory=json.loads((p/'FREEZE_SHA256.json').read_text());actual={str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file() and f.name!='FREEZE_SHA256.json'}
        assert actual==inventory
        verified[name]={'freeze_sha256':expected,'files_verified':len(inventory),'runtime_sha256':{f:sha(p/f) for f in ['launch_device_smoke_spark.py','numeric_evidence.py','test_launcher.py']}}
    old=TMP/'reference_device_smoke_launch_001';new=TMP/'reference_device_smoke_launch_002'
    expected=(old/'launch_device_smoke_spark.py').read_text().replace('from numeric_evidence import numeric\n','from numeric_evidence import numeric,require_eight_float32_ticks\n').replace("    if not all(numeric(clocks,'all_sensors_valid'", "    require_eight_float32_ticks(current,n*14)\n    if not all(numeric(clocks,'all_sensors_valid'")
    assert expected==(new/'launch_device_smoke_spark.py').read_text()
    supervisor=TMP/'reference_physics_adapter_009/source_009/tools/launch_reference_physics_spark.py'
    assert sha(supervisor)=='9ebabaf254cc77fabad7d4759cc80fc26d7efd3912ea84bd55304594c88ed561'
    result={'inventories':verified,'launcher_delta_only_import_and_clock_check':True,'source009_supervisor_sha256':sha(supervisor),'no_frozen_bytes_edited':True,'GPU_launches':0}
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
