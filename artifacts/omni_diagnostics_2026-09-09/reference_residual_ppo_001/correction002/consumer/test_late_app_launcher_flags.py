"""Actual CLI preflight with the host's full late-added AppLauncher flags.

These subprocesses return before AppLauncher import and do not allocate a GPU.
The preserved001 command must reproduce the proof-path overwrite;002 must keep
all proof paths for either position of the exact device flag.
"""
from pathlib import Path
import json,subprocess,sys,tempfile,unittest
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'tmp/reference_physics_adapter_009/source_009').is_dir())

class LateAppFlagsTests(unittest.TestCase):
    def command(self,script,output,*,flags_first=False):
        flags=['--headless','--device','cuda:0','--info',
            '--kit_args=--/exts/omni.kit.telemetry/skipDeferredStartup=true --/app/extensions/excluded/4=omni.kit.telemetry']
        args=['--mode','smoke','--source-root',str(ROOT/'tmp/reference_physics_adapter_009/source_009'),
            '--run',str(ROOT/'tmp/reference_device_smoke_launch_001_preflight_inputs'),
            '--device-run',str(ROOT/'tmp/reference_device_smoke_actual001_review/raw/reference_device_smoke_001'),
            '--bridge',str(ROOT/'tmp/reference_device_smoke_001'),
            '--observation-bundle',str(ROOT/'tmp/reference_policy_observation_005_001'),
            '--package',str(ROOT/'tmp/reference_device_smoke_launch_001_preflight_inputs/inputs/study'),
            '--standing',str(output.parent/'standing/admission.json'),'--output',str(output),'--preflight-only']
        return [sys.executable,str(script)]+(flags+args if flags_first else args+flags)
    def test_exact_old_command_reproduces_failure_before_app(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'smoke';old=ROOT/'tmp/reference_residual_ppo_source_001/run_residual_ppo.py'
            r=subprocess.run(self.command(old,output),cwd=tmp,capture_output=True,text=True,timeout=30)
            self.assertNotEqual(r.returncode,0)
            self.assertIn('cuda:0/campaign.json',r.stderr)
            self.assertNotIn('REFERENCE_SCREEN_APP_START',r.stdout)
            self.assertFalse(output.exists())
    def test_corrected_exact_command_preserves_device_proof_and_all_flags(self):
        for first in (False,True):
            with self.subTest(flags_first=first),tempfile.TemporaryDirectory() as tmp:
                output=Path(tmp)/'smoke'
                r=subprocess.run(self.command(HERE/'run_residual_ppo.py',output,flags_first=first),cwd=tmp,capture_output=True,text=True,timeout=30)
                self.assertEqual(r.returncode,0,r.stderr)
                result=json.loads(r.stdout)
                self.assertEqual(result['device_admission_sha256'],'1e22a3b292be40501d97c6fd4a35e5a015c873fb773c2a9dca1ae5d28d8833b1')
                self.assertIsNone(result['standing_admission_sha256'])
                self.assertFalse(result['Stage2_complete'])
                self.assertNotIn('REFERENCE_SCREEN_APP_START',r.stdout)
                self.assertFalse(output.exists())
if __name__=='__main__':unittest.main()
