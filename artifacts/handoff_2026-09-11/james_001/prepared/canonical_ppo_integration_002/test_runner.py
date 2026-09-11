"""Real RSL CPU learner on a labeled synthetic405/408 fixture, never native physics."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
from test_support import ROOT,TEST_DEPS
# Test dependency only; the runtime uses its installed, source-bound RSL package.
sys.path.insert(0,str(TEST_DEPS))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from canonical_direct_ppo.runner import learn_smoke,checked_observation,strict_reload,construct,equal_tree
from canonical_direct_ppo.smoke_config import protocol,runner_config

class FakeEnv:
    num_envs=32;num_actions=18;device='cpu'
    def __init__(self,fail_at=None):
        self.n=0;self.fail_at=fail_at;self.raw=[]
        self.obs=torch.zeros(32,405);self.obs[:,5]=-1
    def get_observations(self):
        return {'policy':self.obs,'critic':torch.cat([self.obs,torch.zeros(32,3)],dim=-1)}
    def step(self,action):
        self.n+=1;self.raw.append(action.detach().clone())
        if self.n==self.fail_at:raise RuntimeError('injected native safety failure')
        self.obs[:,-18:]=action*.1
        reward=-action.square().mean(-1)
        return self.get_observations(),reward,torch.zeros(32,dtype=torch.bool),{}
    def export_runtime_state(self):return {'CPU_FIXTURE_ONLY':True,'n':self.n,'obs':self.obs.clone(),'array':np.arange(3)}
    def export_audit(self,output):
        return {'CPU_FIXTURE_ONLY':True,'actual_attempted_controls':self.n,'stored_raw_rows':len(self.raw)}

class RunnerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_real_two_updates_and_every_checkpoint_strict_reload(self):
        env=FakeEnv()
        with tempfile.TemporaryDirectory()as d:
            receipt=learn_smoke(env,{'CPU_FIXTURE_ONLY':True},Path(d)/'run',device='cpu')
            self.assertEqual(receipt['status'],'completed');self.assertEqual(receipt['updates_completed'],2)
            self.assertEqual(receipt['controls_completed'],48);self.assertEqual(receipt['optimizer_steps_completed'],40)
            self.assertEqual(receipt['transitions'],1536);self.assertFalse(receipt['quality_admitted'])
            for update in receipt['updates']:
                self.assertTrue(update['reload']['passed']);self.assertTrue(update['reload']['strict_model_normalizer_optimizer'])
                self.assertGreater(update['reload']['optimizer_entries'],0)
    def test_failure_preserves_actual_attempted_native_prefix_without_second_update(self):
        env=FakeEnv(fail_at=27)
        with tempfile.TemporaryDirectory()as d:
            out=Path(d)/'run'
            with self.assertRaisesRegex(RuntimeError,'native safety'):learn_smoke(env,{'CPU_FIXTURE_ONLY':True},out,device='cpu')
            r=json.loads((out/'state.json').read_text());self.assertEqual(r['status'],'failed')
            self.assertEqual(r['controls_completed'],26);self.assertEqual(r['controls_attempted'],27)
            self.assertEqual(r['native_audit']['stored_raw_rows'],27);self.assertEqual(r['updates_completed'],1)
            self.assertFalse((out/'decision_002.pt').exists())
    def test_observation_clone_and_privilege_boundary(self):
        env=FakeEnv();obs=checked_observation(env.get_observations());env.obs.fill_(7)
        self.assertFalse(bool((obs['policy']==7).all()))
        wrong=env.get_observations();wrong['critic'][:,0]+=1
        with self.assertRaises(ValueError):checked_observation(wrong)
    def test_protocol_not_old_task_and_explicit_smoke_parameters(self):
        p=protocol();self.assertEqual(p['actor_width'],405);self.assertEqual(p['critic_width'],408)
        self.assertIsNone(p['checkpoint_input']);self.assertFalse(p['auto_reset']);self.assertEqual(p['executed_target_slew_rad_per20ms'],.04)
        self.assertEqual(runner_config()['actor']['hidden_dims'],[512,256,128])
        self.assertTrue(equal_tree({'x':np.array([1])},{'x':np.array([1])}))

if __name__=='__main__':unittest.main()
