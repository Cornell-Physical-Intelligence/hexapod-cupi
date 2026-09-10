"""CPU contract tests use real Torch state/checkpoint bytes, without simulator mocks."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import hashlib
import json
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
HERE=Path(__file__).parent;ROOT=HERE.parents[1]
sys.path.append(str(ROOT/'tools'))
from velocity_action import VelocityActionConfig,JointTargetVelocity
from candidate_runner import *
from omni_quiet_review import QUIET_GATES

IDENTITY={'variant':'f050_t060','urdf_sha256':'e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c',
          'plan_sha256':'b'*64,'stance_index':0,'source_sha256':'c'*64}
class Model(torch.nn.Module):
    def __init__(self,width,output):
        super().__init__();self.mlp=torch.nn.Sequential(torch.nn.Linear(width,8),torch.nn.Tanh(),torch.nn.Linear(8,output))
        self.register_buffer('obs_normalizer',torch.zeros(width))
        if output==18:
            self.distribution=torch.nn.Module();self.distribution.std_type='scalar'
            self.distribution.std_param=torch.nn.Parameter(torch.ones(18))
class Runner:
    def __init__(self):
        self.alg=SimpleNamespace(actor=Model(495,18),critic=Model(498,1),learning_rate=1e-4,schedule='adaptive',entropy_coef=0.)
        self.alg.optimizer=torch.optim.Adam(list(self.alg.actor.parameters())+list(self.alg.critic.parameters()),lr=1e-4)
        self.current_learning_iteration=0;self.device='cpu';self.loads=0;self.saves=0
        self.logger=SimpleNamespace(cfg={'logger':'tensorboard'},log_dir='test')
    def save(self,path,infos):
        self.saves+=1;torch.save({'actor_state_dict':self.alg.actor.state_dict(),'critic_state_dict':self.alg.critic.state_dict(),
                                'infos':infos,'iter':self.current_learning_iteration},path)
    def load(self,path,strict,map_location):
        self.loads+=1;s=torch.load(path,weights_only=False)
        self.alg.actor.load_state_dict(s['actor_state_dict'],strict=strict);self.alg.critic.load_state_dict(s['critic_state_dict'],strict=strict)

class RunnerTests(unittest.TestCase):
    def test_corrective_scope_is_exact_and_episode_matched(self):
        self.assertEqual(INITIAL_STD,.005)
        self.assertEqual(SAMPLED_CALIBRATION_STEPS,1000)
        self.assertEqual(SAMPLED_CALIBRATION_STEPS*.02,20.)
        source=(HERE/'candidate_runner.py').read_text()
        self.assertIn("('sampled_std_0p005',True,SAMPLED_CALIBRATION_STEPS)",source)
        self.assertIn("mismatch",source)
    def test_zero_mean_initialization_and_explicit_std(self):
        r=Runner();report=initialize_scratch(r,VelocityActionConfig('formal_004'))
        self.assertTrue(torch.equal(r.alg.actor.mlp(torch.randn(32,495)),torch.zeros(32,18)))
        self.assertTrue(torch.allclose(r.alg.actor.distribution.std_param,torch.full((18,),.005)))
        self.assertEqual(report['optimizer_state_entries'],0)
        r.current_learning_iteration=1
        with self.assertRaises(ValueError):initialize_scratch(r,VelocityActionConfig('formal_004'))
    def test_early_local_logger_sentinel_preserves_existing_writer(self):
        r=Runner();prepare_local_logger_for_early_checkpoint(r);self.assertIsNone(r.logger.writer)
        existing=object();r.logger.writer=existing;prepare_local_logger_for_early_checkpoint(r)
        self.assertIs(r.logger.writer,existing)
        r=Runner();r.logger.cfg['logger']='wandb'
        with self.assertRaises(ValueError):prepare_local_logger_for_early_checkpoint(r)
    def test_checkpoint_roundtrip_immutable_and_inference_buffers(self):
        c=VelocityActionConfig('formal_004');r=Runner()
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.pt';save_candidate(r,path,c,IDENTITY)
            with torch.inference_mode():r.alg.actor.obs_normalizer=torch.zeros(495)
            report=load_candidate(r,path,c,IDENTITY)
            self.assertFalse(r.alg.actor.obs_normalizer.is_inference());self.assertEqual(r.loads,1)
            self.assertIn('actor.obs_normalizer',report['inference_buffers_made_writable'])
            with self.assertRaises(FileExistsError):save_candidate(r,path,c,IDENTITY)
            self.assertEqual(r.saves,1)
            path.unlink()
            with self.assertRaises(FileExistsError):save_candidate(r,path,c,IDENTITY)
    def test_reserved_identity_wrong_profile_hash_and_embedded_metadata(self):
        c=VelocityActionConfig('formal_004');r=Runner()
        with self.assertRaises(ValueError):checkpoint_metadata(c,{**IDENTITY,'profile':'diagnostic_003'})
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'x.pt';save_candidate(r,path,c,IDENTITY)
            with self.assertRaises(ValueError):load_candidate(r,path,VelocityActionConfig('diagnostic_003'),IDENTITY)
            with self.assertRaises(ValueError):load_candidate(r,path,c,{**IDENTITY,'source_sha256':'d'*64})
            saved=torch.load(path,weights_only=False);saved['infos']['candidate_contract']['profile']='old_absolute'
            torch.save(saved,path)
            with self.assertRaises(ValueError):load_candidate(r,path,c,IDENTITY)
            side=path.with_suffix('.pt.json');meta=json.loads(side.read_text());meta['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();side.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'embedded'):load_candidate(r,path,c,IDENTITY)
            self.assertEqual(r.loads,0)
    def test_inference_rollout_can_reset_outside_inference_mode(self):
        names=[str(i) for i in range(18)];c=JointTargetVelocity(names,{n:-1 for n in names},{n:1 for n in names},2,VelocityActionConfig('formal_004'),dtype=torch.float32)
        c.reset(torch.zeros(2,18))
        with torch.inference_mode():result=c.step(torch.ones(2,18))
        self.assertFalse(c.position.is_inference());self.assertFalse(result.clipped_action.is_inference())
        c.reset(torch.zeros(2,18));self.assertTrue((c.velocity==0).all())
    def test_per_replica_per_joint_saturation_and_nonfinite_fail(self):
        shape=(150,2,18);data={k:np.zeros(shape) for k in ['applied_torque_nm','computed_torque_nm','joint_position_rad','joint_velocity_rad_s','joint_target_rad','executable_target_velocity_rad_s','executable_target_acceleration_rad_s2']}
        data.update(position_world_m=np.zeros((150,2,3)),quaternion_world_wxyz=np.tile([1.,0,0,0],(150,2,1)),
                    terminated=np.zeros((150,2),bool),truncated=np.zeros((150,2),bool),reward_term_nonfoot=np.zeros((150,2)))
        c=VelocityActionConfig('formal_004');names=list(map(str,range(18)))
        self.assertTrue(summarize_calibration(data,names,.02,c,quiet_required=True)['passed'])
        data['computed_torque_nm'][100,1,0]=1.61
        report=summarize_calibration(data,names,.02,c,quiet_required=False)
        self.assertTrue(report['per_environment'][0]['passed']);self.assertFalse(report['per_environment'][1]['passed'])
        data['applied_torque_nm'][0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):summarize_calibration(data,names,.02,c,quiet_required=False)
    def test_preflight_rejects_resume_before_reading_plan(self):
        args=SimpleNamespace(mode='train',checkpoint=Path('old.pt'))
        with self.assertRaisesRegex(ValueError,'never resumes'):preflight_candidate(args)
    def test_preflight_actual_c_asset_and_matched_admission(self):
        with TemporaryDirectory() as tmp:
            package=Path(tmp);(package/'urdf').mkdir()
            manifest=json.loads((ROOT/'robot/hexapod_mkii_length_study/manifest.json').read_text())
            record=next(r for r in manifest['variants'] if r['variant']=='f050_t060')
            (package/record['urdf']).write_bytes((ROOT/'robot/hexapod_mkii_length_study'/record['urdf']).read_bytes())
            (package/'manifest.json').write_text(json.dumps(manifest))
            plan=json.loads((ROOT/'robot/hexapod_mkii_length_study/training_plan.json').read_text())
            plan.update(training_num_envs=1024,evaluation_num_envs=48,training_iterations=50)
            stance=plan['variants']['f050_t060']['stances'][0]
            stance['joint_positions_rad']={manifest['link_joint_mapping'][leg]['joints'][part]:float(np.deg2rad(deg))
                for leg in ('lf','lm','lr','rf','rm','rr') for part,deg in [('coxa',0),('femur',40),('tibia',120)]}
            stance['root_height_at_contact_m']=.13053251856352807;stance['suggested_reset_root_height_m']=.13653251856352808
            plan['omni']={'architecture':LINEAGE,'initial_exploration_std':.005,'sampled_calibration_control_steps':1000,'velocity_candidate':{'profile':'formal_004','max_acceleration_rad_s2':8.},'overrides':{'observation_noise_scale':1.,'target_filter_time_constant_s':0.}}
            (package/'training_plan.json').write_text(json.dumps(plan))
            args=SimpleNamespace(mode='validate',checkpoint=None,package=package,output=package/'result',iterations=None,stance_index=0,variant='f050_t060',admission=None)
            with patch('candidate_runner.source_digest',return_value='c'*64):
                identity,config=preflight_candidate(args)
                self.assertEqual(identity['urdf_sha256'],IDENTITY['urdf_sha256'])
                plan['omni']['initial_exploration_std']=.05
                (package/'training_plan.json').write_text(json.dumps(plan))
                with self.assertRaisesRegex(ValueError,'explicit std'):preflight_candidate(args)
                plan['omni']['initial_exploration_std']=.005
                (package/'training_plan.json').write_text(json.dumps(plan))
                args.mode='probe';args.admission=package/'admission.json'
                args.admission.write_text(json.dumps({**identity,'gate':{'passed':True}}))
                preflight_candidate(args)
                args.admission.write_text(json.dumps({**identity,'source_sha256':'d'*64,'gate':{'passed':True}}))
                with self.assertRaisesRegex(ValueError,'matching source'):preflight_candidate(args)
    def test_candidate_diagnostics_schema_and_state_audit_present(self):
        source=(HERE/'candidate_diagnostics.py').read_text()
        self.assertNotIn('315',source);self.assertNotIn('318',source)
        self.assertIn('max_executable_state_difference',source);self.assertIn('per_replica',source)
        compile(source,'candidate_diagnostics','exec')
        compile((HERE/'train_velocity_candidate.py').read_text(),'candidate_entrypoint','exec')

if __name__=='__main__':unittest.main()
