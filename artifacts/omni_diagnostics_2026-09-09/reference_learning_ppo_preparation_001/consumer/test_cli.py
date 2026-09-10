"""Actual subprocess preflight with complete late AppLauncher flags, no app/GPU."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'tmp/reference_physics_adapter_009/source_009').is_dir())


class Tests(unittest.TestCase):
    def test_full_host_command_before_and_after_late_device_flags(self):
        # An isolated copied fixture gets its own immutable inventory. This
        # permits the preflight test before the owner candidate is frozen.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'consumer'
            shutil.copytree(HERE,source,ignore=shutil.ignore_patterns('__pycache__','FREEZE_SHA256.json'))
            inventory={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
            (source/'FREEZE_SHA256.json').write_text(json.dumps(inventory,indent=2)+'\n')
            flags=['--headless','--device','cuda:0','--info','--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true']
            for first,mode in ((False,'calibrate'),(True,'calibrate'),(False,'learning_recovery_32')):
                phase='calibration' if mode=='calibrate' else mode
                output=root/(('before' if first else 'after')+'_'+mode)/phase
                args=['--mode',mode,'--source-root',str(ROOT/'tmp/reference_physics_adapter_009/source_009'),
                    '--run',str(ROOT/'tmp/reference_device_smoke_launch_001_preflight_inputs'),
                    '--device-run',str(ROOT/'tmp/reference_device_smoke_actual001_review/raw/reference_device_smoke_001'),
                    '--bridge',str(ROOT/'tmp/reference_device_smoke_001'),
                    '--observation-bundle',str(ROOT/'tmp/reference_policy_observation_005_001'),
                    '--package',str(ROOT/'tmp/reference_device_smoke_launch_001_preflight_inputs/inputs/study'),
                    '--standing',str(output.parent/'standing/admission.json'),'--output',str(output),'--preflight-only']
                result=subprocess.run([sys.executable,'-S',str(source/'run_moving_ppo.py')]+(flags+args if first else args+flags),
                    cwd=root,capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr)
                identity=json.loads(result.stdout)
                self.assertEqual(identity['maximum_PPO_updates'],25)
                self.assertEqual(identity['initial_allocation_updates'],10)
                self.assertEqual(identity['device_admission_sha256'],'1e22a3b292be40501d97c6fd4a35e5a015c873fb773c2a9dca1ae5d28d8833b1')
                self.assertIsNone(identity['standing_admission_sha256'])
                self.assertNotIn('REFERENCE_SCREEN_APP_START',result.stdout)
                self.assertFalse(output.exists())

if __name__=='__main__':unittest.main()
