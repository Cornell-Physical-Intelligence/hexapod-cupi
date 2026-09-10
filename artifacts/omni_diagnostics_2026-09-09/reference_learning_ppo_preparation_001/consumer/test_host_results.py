"""Stdlib host checks bind complete phase receipts, not only success booleans."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from campaign_contract import validate_result,save,sha,RUNTIME,RUNTIME_SCHEMA
from moving_runner import contract,BINDING_KEYS

HERE=Path(__file__).resolve().parent


class Tests(unittest.TestCase):
    def fixture(self,root):
        calibration=root/'calibration';calibration.mkdir()
        # Preserve an actual passed32-replica calibration's per-row bounds.
        actual=HERE/'inputs/calibration002.json'
        calibration.joinpath('calibration.json').write_bytes(actual.read_bytes())
        bindings={k:'0'*64 for k in BINDING_KEYS};bindings['observation_schema_sha256']=RUNTIME_SCHEMA
        bindings['calibration_sha256']=sha(calibration/'calibration.json')
        policy=contract('admitted_bounded_moving_PPO',bindings)
        identity={k:bindings[k] for k in BINDING_KEYS if k not in ('calibration_sha256','consumer_source_sha256','observation_schema_sha256')}
        identity['consumer_freeze_sha256']='0'*64
        (calibration/'initial.pt').write_bytes(b'synthetic checkpoint bytes; host does not unpickle')
        save(calibration/'initial.pt.json',{**policy,'checkpoint_sha256':sha(calibration/'initial.pt')})
        output=root/'profile_32';raw=output/'raw';raw.mkdir(parents=True)
        session={'controls':712,'failure':None,'physical_admission':False,'evaluation_gates_changed':False}
        save(raw/'session.json',session)
        save(raw/'physics_substep_review.json',{'completed_controls':712,'samples_including_initial':5697,'error':None,'method_restored':True})
        save(raw/'episodes.json',[])
        for name in ('trace.npz','sensor_clocks.npz','reference_states.npz','physics_substeps.npz'):(raw/name).write_bytes(b'synthetic raw file placeholder; numerical replay separate')
        state={'status':'completed','mode':'profile_32','identity':identity,'source_inputs_unchanged':True,'runtime_binding':{'runtime_tree_sha256':RUNTIME},
            'Stage2_complete':False,'policy_identity':policy,'input_checkpoint_sha256':sha(calibration/'initial.pt'),
            'session':session,'replicas':32,'profile_controls':512,'profile_passed':True,'profile_control_seconds':[.01]*512,
            'profile_wall_s':5.2,
            'active_transitions':16384,'PPO_updates_completed':0,'policy_training_started':False}
        save(output/'state.json',state)
        return output,identity,state

    def test_actual_calibration_bound_profile_and_rejections(self):
        with tempfile.TemporaryDirectory() as d:
            out,identity,state=self.fixture(Path(d))
            result=validate_result(out,identity)
            self.assertEqual(result['phase'],'profile_32');self.assertEqual(result['controls'],712)
            for key,value in [('active_transitions',16383),('input_checkpoint_sha256','f'*64),('profile_passed',False),('policy_training_started',True),('source_inputs_unchanged',False)]:
                bad=copy.deepcopy(state);bad[key]=value;save(out/'state.json',bad)
                with self.assertRaises(ValueError):validate_result(out,identity)
            save(out/'state.json',state)
            save(out/'raw/episodes.json',[{'ended_rows':[0]}])
            with self.assertRaises(ValueError):validate_result(out,identity)

    def test_host_modules_are_standard_library_only(self):
        code='import sys;sys.path.insert(0,'+repr(str(HERE))+');import campaign_contract,run_moving_ppo;assert not any(x=="torch" or x=="numpy" or x.startswith("rsl_rl") for x in sys.modules)'
        subprocess.run([sys.executable,'-S','-c',code],check=True,capture_output=True,text=True)

if __name__=='__main__':unittest.main()
