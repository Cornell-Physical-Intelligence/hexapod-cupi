import hashlib
import json
from unittest.mock import patch
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import torch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'tools/omni_repair_training.py').exists())
sys.path.insert(0,str(ROOT/'tools'))
from omni_repair_training import repair_options, load_repair_checkpoint, stand_raw_action_cost
from launch_omni_repair_pair_spark import assert_source_pair, checked_report

CHECKPOINT=None
OPTIONS={'checkpoint_sha256':'1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8',
         'exploration_std':.1,'entropy_coef':0.,'optimizer':'reset','learning_rate':5e-5}

class Model:
    def __init__(self): self.values={}
    def state_dict(self): return self.values
    def load(self, value):
        self.values={k:v.clone() for k,v in value.items()}
        if 'distribution.std_param' in self.values:
            self.distribution=SimpleNamespace(std_type='scalar',std_param=self.values['distribution.std_param'])

class Runner:
    def __init__(self):
        self.alg=SimpleNamespace(actor=Model(),critic=Model(),optimizer=SimpleNamespace(state={},param_groups=[{'lr':3e-4}]),schedule='adaptive')
        self.current_learning_iteration=0;self.calls=0;self.device="cpu"
    def load(self,path,load_cfg,strict,map_location):
        assert load_cfg=={'actor':True,'critic':True,'optimizer':False,'iteration':True,'rnd':False};assert strict
        self.calls+=1;s=torch.load(path,map_location='cpu',weights_only=False)
        self.alg.actor.load(s['actor_state_dict']);self.alg.critic.load(s['critic_state_dict']);self.current_learning_iteration=s['iter']

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global CHECKPOINT
        cls.temporary=tempfile.TemporaryDirectory()
        CHECKPOINT=Path(cls.temporary.name)/'checkpoint.pt'
        actor={'mlp.0.weight':torch.arange(2*315,dtype=torch.float32).reshape(2,315),
               'obs_normalizer._mean':torch.full((1,315),.2),
               'distribution.std_param':torch.full((18,),.4)}
        critic={'mlp.0.weight':torch.arange(2*318,dtype=torch.float32).reshape(2,318),
                'obs_normalizer._mean':torch.full((1,318),.3)}
        torch.save({'actor_state_dict':actor,'critic_state_dict':critic,'iter':1847},CHECKPOINT)
        OPTIONS['checkpoint_sha256']=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()
    def test_options_fail_closed(self):
        for change in ({'exploration_std':0},{'optimizer':'resume'},{'extra':1},{'entropy_coef':.01},{'learning_rate':float('nan')}):
            with self.assertRaises(ValueError):repair_options({**OPTIONS,**change})
    def test_wrong_checkpoint_no_resume(self):
        r=Runner()
        with self.assertRaises(ValueError):load_repair_checkpoint(r,CHECKPOINT,{**OPTIONS,'checkpoint_sha256':'0'*64})
        self.assertEqual(r.calls,0)
    def test_load_preserves_weights_and_normalizers(self):
        r=Runner();report=load_repair_checkpoint(r,CHECKPOINT,OPTIONS)
        self.assertEqual(report['optimizer_state_entries'],0)
        self.assertTrue(report['observation_normalizers_preserved'])
        self.assertTrue(torch.allclose(r.alg.actor.distribution.std_param,torch.full((18,),.1)))
        self.assertEqual(r.alg.optimizer.param_groups[0]['lr'],5e-5)
        self.assertEqual(r.alg.entropy_coef,0.)
        with self.assertRaises(RuntimeError):
            r.alg.optimizer.state['fake']=1;load_repair_checkpoint(r,CHECKPOINT,OPTIONS)
    def test_corrupt_normalizer_rejected(self):
        r=Runner();old=r.load
        def corrupted(*a,**kw):
            old(*a,**kw);r.alg.actor.values['obs_normalizer._mean']+=1
        r.load=corrupted
        with self.assertRaises(RuntimeError):load_repair_checkpoint(r,CHECKPOINT,OPTIONS)
    def test_raw_penalty_unclipped_and_only_quiet(self):
        cmd=torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,0.,.2],[.03,0.,.05]])
        raw=torch.full((4,18),2.)
        self.assertEqual(stand_raw_action_cost(cmd,raw).tolist(),[4.,0.,0.,4.])
        self.assertEqual(stand_raw_action_cost(cmd,torch.zeros_like(raw)).tolist(),[0.]*4)
    def test_capture_precedes_parent_clipping(self):
        code=(ROOT/'tools/omni_flat_env.py').read_text().split('def _pre_physics_step',1)[1].split('def _get_observations',1)[0]
        self.assertLess(code.index('self.omni_raw_policy_action = actions.clone()'),code.index('super()._pre_physics_step(actions)'))

class PairEvidenceTests(unittest.TestCase):
    def test_code_and_asset_differences_cannot_masquerade_as_reward_comparison(self):
        plan = 'robot/hexapod_mkii_length_study/training_plan.json'
        left = {plan: 'a', 'tools/env.py': 'same', 'robot/asset.urdf': 'same'}
        right = {**left, plan: 'b'}
        assert_source_pair([left, right])
        for bad in ({**right, 'tools/env.py': 'other'},
                    {**right, 'robot/asset.urdf': 'other'},
                    {plan: 'b', 'tools/env.py': 'same'}, left):
            with self.assertRaises(ValueError):
                assert_source_pair([left, bad])

    def test_nonfinite_applied_torque_never_reaches_allocation_screen(self):
        report = {'checkpoint_sha256': 'identity', 'complete': True,
                  'observation_audit': {'actor_width': 315, 'critic_width': 318,
                      'max_same_step_repeat_difference': 0, 'max_command_slice_difference': 0,
                      'max_history_shift_difference': 0},
                  'scenarios': [{'windows': {'all': {'applied_torque_abs_max_nm': 0}}}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'diagnostics.json'
            for bad in (float('nan'), float('inf'), -0.01, 1.7):
                report['scenarios'][0]['windows']['all']['applied_torque_abs_max_nm'] = bad
                path.write_text(json.dumps(report))
                with patch('launch_omni_repair_pair_spark.metrics') as screen:
                    with self.assertRaises(RuntimeError):
                        checked_report(path.parent, 'identity')
                    screen.assert_not_called()


if __name__=='__main__': unittest.main()
