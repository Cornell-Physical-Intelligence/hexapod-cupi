from pathlib import Path
import ast,hashlib,importlib.util,json,sys,unittest
from types import SimpleNamespace
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from reset_pose_override import ResetPoseOverride,arm_targets
PLAN=json.loads((HERE/'plan.json').read_text())
path=ROOT/'tmp/reference_moving_ppo_003/recovery.py'
spec=importlib.util.spec_from_file_location('bound_recovery',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
RecoveryTargets=mod.RecoveryTargets

def make_env():
    q=torch.tensor([0.]*6+[float(torch.tensor(40*torch.pi/180))]*6+[float(torch.tensor(120*torch.pi/180))]*6).expand(32,-1).clone()
    limits=torch.stack([torch.full_like(q,-4),torch.full_like(q,4)],-1)
    data=SimpleNamespace(default_joint_pos=q.clone(),soft_joint_pos_limits=limits,joint_pos=q.clone(),joint_vel=torch.zeros_like(q),joint_pos_target=q.clone(),joint_vel_target=torch.zeros_like(q),root_link_pos_w=torch.randn(32,3),root_link_quat_w=torch.tensor([0.,0.,0.,1.]).expand(32,-1).clone(),root_com_lin_vel_w=torch.zeros(32,3),root_link_ang_vel_w=torch.zeros(32,3))
    controller=SimpleNamespace(reference_position=q.double().clone(),reference_velocity=q.double()*0,residual_position=q.double()*0,residual_velocity=q.double()*0)
    def reset(q,r,env_ids):
        controller.reference_position[env_ids]=r;controller.reference_velocity[env_ids]=0;controller.residual_position[env_ids]=0;controller.residual_velocity[env_ids]=0
    controller.reset=reset
    robot=SimpleNamespace(joint_names=PLAN['joint_names_runtime'],data=data)
    robot.write_joint_position_to_sim_index=lambda *,position,env_ids: data.joint_pos.__setitem__(env_ids,position)
    robot.set_joint_position_target_index=lambda *,target,env_ids: data.joint_pos_target.__setitem__(env_ids,target)
    env=SimpleNamespace(num_envs=32,step_dt=.02,device='cpu',_sim_step_counter=0,_robot=robot,reference_residual_controller=controller,_previous_processed_joint_target=q.clone(),_processed_actions=q.clone(),_actions=torch.zeros_like(q),_has_previous_processed_joint_target=torch.ones(32,dtype=torch.bool),_reference_fresh=torch.zeros(32,dtype=torch.bool),reset_calls=0)
    def original(ids):
        ids=torch.arange(32) if ids is None else ids
        env.reset_calls+=1
        resetq=data.default_joint_pos[ids]+.015
        data.joint_pos[ids]=resetq;data.joint_vel[ids]=0;data.joint_pos_target[ids]=resetq
        env._previous_processed_joint_target[ids]=resetq;env._processed_actions[ids]=resetq
        reset(resetq.double(),resetq.double(),ids)
    env._reset_idx=original
    return env

class ResetTests(unittest.TestCase):
    def test_failed_vector_named_reorder(self):
        e=make_env();names=list(reversed(e._robot.joint_names))
        q=arm_targets(PLAN,'recorded_failed_pose',names,e._robot.data.default_joint_pos)
        expected=torch.tensor(list(reversed(PLAN['recorded_failure']['reset_q_rad'])))
        self.assertTrue(torch.equal(q[8],expected))
    def test_exact_canonical_and_recorded_first_last_knots(self):
        e=make_env();nom=e._robot.data.default_joint_pos.double()
        for arm in ('recorded_failed_pose','canonical_pose'):
            q=arm_targets(PLAN,arm,e._robot.joint_names,nom).double()
            r=RecoveryTargets(q,nom,e._robot.data.soft_joint_pos_limits)
            self.assertTrue(torch.equal(r._position(torch.zeros(32,dtype=torch.long)),q))
            vmax=amax=0.
            for i in range(200):
                s=r.sample_next();vmax=max(vmax,float(s['discrete_v_ref'].abs().max()));amax=max(amax,float(s['discrete_a_ref'].abs().max()))
                completed=r.advanced();self.assertEqual(bool(completed.all()),i==199)
                if i>=99:self.assertTrue(torch.equal(s['q_ref'],nom))
            self.assertLessEqual(vmax,1.75);self.assertLessEqual(amax,6)
            self.assertTrue(torch.equal(r.sample_next()['discrete_v_ref'],torch.zeros_like(nom)))
            self.assertTrue(torch.equal(r.sample_next()['discrete_a_ref'],torch.zeros_like(nom)))
    def test_full_mixed_root_and_unselected_preserved(self):
        for arm in ('recorded_failed_pose','canonical_pose'):
            e=make_env();original=e._reset_idx;roots=e._robot.data.root_link_pos_w.clone()
            override=ResetPoseOverride(e,PLAN,arm)
            with override.installed():
                for epoch in PLAN['epochs']:
                    e._sim_step_counter=epoch['at_control']*8
                    ids=override.arm_next();e._reset_idx(ids)
                    self.assertTrue(override.records[-1]['passed'])
                    self.assertTrue(torch.equal(roots,e._robot.data.root_link_pos_w))
            self.assertEqual(e.reset_calls,5);self.assertEqual(e._reset_idx,original)
    def test_unarmed_and_wrong_time_fail_before_write(self):
        e=make_env();o=ResetPoseOverride(e,PLAN,'canonical_pose')
        with o.installed():
            with self.assertRaisesRegex(RuntimeError,'Unarmed'):e._reset_idx(torch.arange(32))
            o.arm_next();e._sim_step_counter=8
            with self.assertRaisesRegex(RuntimeError,'advanced'):e._reset_idx(torch.arange(32))
        self.assertEqual(e.reset_calls,0)
    def test_wrong_rows_fail_before_write(self):
        e=make_env();o=ResetPoseOverride(e,PLAN,'canonical_pose')
        with o.installed():
            o.arm_next()
            with self.assertRaisesRegex(RuntimeError,'rows/order'):e._reset_idx(torch.tensor([8]))
        self.assertEqual(e.reset_calls,0)
    def test_root_corruption_fails_and_wrapper_restored(self):
        e=make_env();original=e._reset_idx;o=ResetPoseOverride(e,PLAN,'canonical_pose')
        setter=e._robot.write_joint_position_to_sim_index
        def broken(**kw):setter(**kw);e._robot.data.root_link_pos_w[8,2]+=.001
        e._robot.write_joint_position_to_sim_index=broken
        with self.assertRaisesRegex(RuntimeError,'readback rejected'):
            with o.installed():o.arm_next();e._reset_idx(torch.arange(32))
        self.assertEqual(e._reset_idx,original);self.assertFalse(o.records[-1]['passed'])
    def test_original_mixed_reset_cannot_change_unselected(self):
        e=make_env();original=e._reset_idx;o=ResetPoseOverride(e,PLAN,'canonical_pose')
        with o.installed():
            for epoch in PLAN['epochs'][:2]:
                e._sim_step_counter=epoch['at_control']*8
                ids=o.arm_next();e._reset_idx(ids)
            def broken(ids):original(ids);e._robot.data.root_link_pos_w[1,2]+=.01
            o.original=broken;e._sim_step_counter=400*8
            ids=o.arm_next()
            with self.assertRaisesRegex(RuntimeError,'Original reset changed unselected'):e._reset_idx(ids)

    def test_bound_recovery_source_and_failure(self):
        bindings=json.loads((HERE/'INPUT_BINDINGS.json').read_text())
        for rel,bound in bindings['files'].items():self.assertEqual(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest(),bound)
        self.assertEqual(PLAN['recorded_failure']['requested_final8_peak_nm'],1.6276484727859497)
    def test_no_step_or_root_writer_in_override(self):
        tree=ast.parse((HERE/'reset_pose_override.py').read_text())
        attrs=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
        self.assertFalse(any('write_root' in a for a in attrs));self.assertNotIn('step',attrs)
    def test_driver_no_actor_and_fixed_schedule(self):
        tree=ast.parse((HERE/'diagnostic_driver.py').read_text())
        attrs=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
        self.assertNotIn('learn',attrs);self.assertNotIn('act',attrs)
        self.assertIn('original_reset_method_restored',(HERE/'diagnostic_driver.py').read_text())
        self.assertEqual(PLAN['max_controls_per_arm'],1000)
        self.assertEqual(set(PLAN['epochs'][2]['rows'])|set(PLAN['epochs'][3]['rows']),set(range(32)))

if __name__=='__main__':unittest.main()
