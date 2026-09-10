"""Actual009 scorer parity and bounded-consumer admission/receipt regressions."""
import copy,json,tempfile,unittest,sys
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from physical_contract import _quiet_pass,verify_standing,SOURCE
HERE=Path(__file__).parent
ROOT=next(p for p in HERE.resolve().parents if (p/'tmp/reference_physics_adapter_009/source_009').is_dir())
sys.path.insert(0,str(ROOT/'tmp/reference_physics_adapter_009/source_009/tools'))
sys.path.insert(0,str(ROOT/'tmp/reference_policy_observation_005_001'))
from observation import ObservationBuilder
from batch_wave import MODES
from physical_scores import quiet_trial,forward_stop_trial

class PhysicalScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures={}
        for phase in ('standing','wave'):
            z=np.load(HERE/'inputs'/(phase+'_scoring_fixture.npz'));arrays={k:z[k] for k in z.files}
            cls.fixtures[phase]=(tuple(arrays['joint_names']),[{k:a[i].copy() for k,a in arrays.items() if k!='joint_names'} for i in range(len(arrays['time_s']))])
    def test_actual009_all32_quiet_matches_original_acceptance(self):
        names,rows=self.fixtures['standing'];result=quiet_trial(rows,names,200,1000)
        self.assertTrue(result['passed']);self.assertEqual(len(result['per_environment']),32)
        _quiet_pass(result,32)
        changed=copy.deepcopy(rows)
        for row in changed[200:]:row['joint_velocity_rad_s'][6,17]=.031
        bad=quiet_trial(changed,names,200,1000)
        self.assertFalse(bad['passed']);self.assertFalse(bad['per_environment'][6]['pass'])
        self.assertEqual(sum(r['pass'] for r in bad['per_environment']),31)
        with self.assertRaises(ValueError):_quiet_pass(bad,32)
    def test_actual009_full_contact_forward_stop_matches_original(self):
        names,rows=self.fixtures['wave'];end=json.loads((HERE/'inputs/wave_final_state.json').read_text())
        session=NS(rows=rows,names=names,wave=NS(s={'mode':torch.tensor([MODES.index(end['mode'])]),
            'quiet_valid':torch.tensor([True]),'quiet_time':torch.tensor([end['quiet_time']],dtype=torch.float64),
            'touchdowns':torch.tensor([end['touchdowns']])}))
        report=forward_stop_trial(session)
        self.assertTrue(report['passed']);self.assertEqual(report['completed_legs'],list(range(6)))
        self.assertAlmostEqual(report['independent_progress']['measured_forward_displacement_m'],.11524266,places=6)
        self.assertGreaterEqual(report['final_quiet']['per_environment'][0]['window_duration_s'],10.)
        session.wave.s['mode'][0]=MODES.index('awaiting_landing_support')
        self.assertFalse(forward_stop_trial(session)['passed'])
    def test_false_or_nonfinite_single_replica_cannot_hide_in_positive_aggregate(self):
        names,rows=self.fixtures['standing'];good=quiet_trial(rows,names,200,1000)
        for change in ('pass','nan','missing'):
            bad=copy.deepcopy(good)
            if change=='pass':bad['per_environment'][5]['pass']=False
            elif change=='nan':bad['per_environment'][5]['max_joint_velocity_rms_rad_s']=float('nan')
            else:bad['per_environment'].pop()
            with self.assertRaises(ValueError):_quiet_pass(bad,32)
    def test_fresh_standing_rejects_different_source_or_campaign(self):
        # The local copy is only a CPU identity fixture, never new physical proof.
        source=HERE/'inputs/standing_admission.json'
        with tempfile.TemporaryDirectory() as tmp:
            campaign=Path(tmp);folder=campaign/'standing';folder.mkdir();path=folder/'admission.json'
            path.write_bytes(source.read_bytes());(folder/'state.json').write_bytes(source.read_bytes())
            self.assertEqual(len(verify_standing(path,campaign)),64)
            with self.assertRaises(ValueError):verify_standing(path,campaign/'other')
            changed=json.loads(path.read_text());changed['identity']['source_manifest_sha256']='0'*64
            for p in (path,folder/'state.json'):p.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):verify_standing(path,campaign)
if __name__=='__main__':unittest.main()
