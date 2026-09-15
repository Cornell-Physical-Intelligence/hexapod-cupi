"""Synthetic CPU preparation only; final v3 frozen-loader proof is separate."""
import copy
from dataclasses import field, fields, make_dataclass
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

import migrate as m

ROOT = Path(__file__).resolve().parents[3]
OLD = m.load_module(ROOT/'artifacts/restart_2026-09-14/paper_walk_execution_001/source_017/learner.py', 'migration_test_old')
SyntheticConfig = make_dataclass('SyntheticConfig', [(k,type(v),field(default=v)) for k,v in m.ADDITIONS.items()], bases=(OLD.Config,))
NEW = SimpleNamespace(SCHEMA=m.NEW_SCHEMA, Config=SyntheticConfig)


class MigrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.prior = self.out/'prior.npz'
        np.savez(self.prior, states=np.zeros((3,61),np.float32), next_states=np.ones((3,61),np.float32))
        c = OLD.Config(num_envs=2, rollout_steps=2, minibatches=1, target_kl=.02,
            actor_hidden=(8,), memory_hidden=(8,), estimator_hidden=(8,), critic_hidden=(8,), discriminator_hidden=(8,))
        self.learner = OLD.PPOLearner(None,self.prior,self.out/'learner',c,device='cpu')
        for optimizer, params in ((self.learner.optimizer,self.learner.model.parameters()), (self.learner.discriminator_optimizer,self.learner.amp.discriminator.parameters())):
            for parameter in params: parameter.grad=torch.ones_like(parameter)*.01
            optimizer.step()
        self.learner.updates=17
        self.learner.transitions=1234
        self.learner.optimizer_steps=9
        self.learner.discriminator_steps=11
        self.learner.episodes=3
        self.learner.bc_steps=4
        self.learner.last_metrics={'retained':3.25,'nested':{'accepted':[1,2]}}
        self.path=self.out/'old.pt'
        self.learner.save(self.path)
        self.payload=torch.load(self.path,map_location='cpu',weights_only=False)
        # Explicitly test preservation of GPU RNG bytes on the CPU-only host.
        self.payload['rng']['cuda']=[torch.tensor([17,23,91],dtype=torch.uint8)]

    def tearDown(self):
        self.temp.cleanup()

    def transform(self,payload=None,new=NEW):
        return m.transform(self.payload if payload is None else payload,OLD,new,m.sha(OLD.__file__),'f'*64)

    def test_transform_preserves_nonempty_adam_rng_counters_metrics_and_config(self):
        original=copy.deepcopy(self.payload)
        result=self.transform()
        m.verify_preserved(self.payload,result)
        self.assertTrue(m.exact(self.payload,original))
        self.assertTrue(self.payload['optimizer']['state'])
        self.assertTrue(self.payload['discriminator_optimizer']['state'])
        self.assertEqual(result['counters'][m.NEW_COUNTER],0)
        self.assertEqual(result['schema'],m.NEW_SCHEMA)
        self.assertTrue(m.exact(result['rng']['cuda'],self.payload['rng']['cuda']))
        key=next(iter(result['model']))
        result['model'][key].add_(1)
        self.assertTrue(m.exact(self.payload,original))

    def test_unexpected_identity_or_existing_configuration_is_rejected(self):
        for key,value in [('schema','wrong'),('learner_sha256','wrong'),('simulation_resume_supported',True),('stage2_complete',True)]:
            p=copy.deepcopy(self.payload);p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.transform(p)
        for mutation in ['missing','target','already']:
            p=copy.deepcopy(self.payload)
            if mutation=='missing':del p['config']['entropy_coefficient']
            if mutation=='target':p['config']['target_kl']=.03
            if mutation=='already':p['counters'][m.NEW_COUNTER]=0
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.transform(p)

    def test_unreviewed_successor_config_field_rejected(self):
        cfg=make_dataclass('UnexpectedConfig',[('unexpected',int,field(default=0))],bases=(SyntheticConfig,))
        with self.assertRaisesRegex(ValueError,'schema delta'):
            self.transform(new=SimpleNamespace(SCHEMA=m.NEW_SCHEMA,Config=cfg))

    def test_preservation_proof_rejects_corruption_in_every_state_family(self):
        for key in ['model','amp','optimizer','discriminator_optimizer','rng','metrics']:
            result=self.transform();result[key]={}
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Preserved payload'):
                m.verify_preserved(self.payload,result)
        result=self.transform();result['counters']['updates']=0
        with self.assertRaisesRegex(ValueError,'Existing counter'):m.verify_preserved(self.payload,result)
        result=self.transform();result['counters'][m.NEW_COUNTER]=False
        with self.assertRaisesRegex(ValueError,'explicitly'):m.verify_preserved(self.payload,result)
        result=self.transform();result['config']['learning_rate']=.001
        with self.assertRaisesRegex(ValueError,'Existing Config'):m.verify_preserved(self.payload,result)

    def test_old_loader_rejects_successor_without_mutation(self):
        p=self.out/'new.pt';torch.save(self.transform(),p)
        self.assertIn('identity',m.rejected_without_mutation(self.learner,p))

    def test_inference_checks_actor_estimator_critic_discriminator_and_rng(self):
        second=OLD.PPOLearner(None,self.prior,self.out/'second',self.learner.config,device='cpu')
        second.load(self.path)
        obs=np.linspace(-.2,.2,231*5,dtype=np.float32).reshape(5,1,231)
        critic=np.concatenate((obs,np.zeros((5,1,3),np.float32)),axis=-1)
        p=self.out/'synthetic_trace.npz';np.savez(p,policy_observation=obs,critic_observation=critic,
            amp_state_before=np.zeros((5,1,61),np.float32),amp_state_after=np.ones((5,1,61),np.float32))
        rng=OLD.PPOLearner._rng();receipt=m.probe_outputs(self.learner,second,p)
        self.assertEqual(receipt['rows'],5)
        self.assertTrue(m.exact(rng,OLD.PPOLearner._rng()))
        with torch.no_grad():next(second.model.estimator.parameters()).add_(.1)
        with self.assertRaisesRegex(ValueError,'inference differs'):m.probe_outputs(self.learner,second,p)

    def test_unapproved_execution_rejected_before_any_output(self):
        p=self.out/'disabled.json';p.write_text('{"root_authorized_execution":false}')
        with self.assertRaisesRegex(ValueError,'not authorized'):m.execute(p)

    def test_source_freeze_rejects_unlisted_or_changed_files(self):
        source=self.out/'source';source.mkdir()
        (source/'learner.py').write_text('pass\n')
        manifest=source/'FREEZE_SHA256.json'
        manifest.write_text(json.dumps({'learner.py':m.sha(source/'learner.py')}))
        digest=m.sha(manifest)
        self.assertEqual(m.verify_source(source,digest),(source/'learner.py').resolve())
        (source/'unexpected.py').write_text('pass\n')
        with self.assertRaisesRegex(ValueError,'unlisted'):m.verify_source(source,digest)
        (source/'unexpected.py').unlink()
        (source/'learner.py').write_text('changed\n')
        with self.assertRaisesRegex(ValueError,'member differs'):m.verify_source(source,digest)


if __name__=='__main__':unittest.main()
