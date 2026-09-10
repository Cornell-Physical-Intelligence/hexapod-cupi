"""Execute the exact source009 reward tail with the new effective-term callback."""
import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch
from reward_adapter import install_reward
from test_moving_reward import fixture

ROOT=Path(__file__).resolve().parent.parent/'reference_physics_adapter_009/source_009/tools'


def ast_function(path,name,namespace):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


class Tests(unittest.TestCase):
    def test_mixed_effective_reward_capture_and_sums_agree(self):
        module=types.ModuleType('omni_flat_env');module.__file__=str(ROOT/'omni_flat_env.py');module.torch=torch
        for name in ('tracking_terms','slew_commands'):
            setattr(module,name,ast_function(ROOT/'omni_flat_math.py',name,{'torch':torch}))
        module.quiet_stand_terms=ast_function(ROOT/'omni_flat_env.py','quiet_stand_terms',{'torch':torch})
        ns=types.SimpleNamespace;wrap=lambda x:ns(torch=x);z=lambda *shape:torch.zeros(*shape)
        n=2;joints=z(n,18);velocity=torch.tensor([[.005,0.,0.],[0.,0.,0.]])
        data=ns(root_lin_vel_b=wrap(velocity),root_ang_vel_b=wrap(z(n,3)),root_lin_vel_w=wrap(velocity),
            projected_gravity_b=wrap(torch.tensor([[0.,0.,-1.]]*n)),root_pos_w=wrap(z(n,3)),
            root_quat_w=wrap(torch.tensor([[0.,0.,0.,1.]]*n)),
            joint_pos=wrap(joints),default_joint_pos=wrap(joints.clone()),joint_vel=wrap(torch.full((n,18),.02)),
            applied_torque=wrap(torch.full((n,18),.2)),computed_torque=wrap(torch.full((n,18),.3)),
            soft_joint_pos_limits=wrap(torch.tensor([[[-2.,2.]]*18]*n)))
        def sensor(bodies):return ns(data=ns(net_forces_w_history=wrap(z(n,1,bodies,3))))
        feet=[ns(data=ns(last_air_time=wrap(torch.full((n,1),.2))),compute_first_contact=lambda dt:wrap(torch.zeros(n,1,dtype=torch.bool))) for _ in range(6)]
        effective_keys=('linear_tracking','yaw_tracking','linear_progress','yaw_progress','vertical_velocity','roll_pitch_rate','tilt','height','torque',
            'torque_excess','worst_torque_excess','saturation','positive_power','action_rate','joint_acceleration','slip','nonfoot','joint_limits',
            'airtime','stand_posture','stand_joint_velocity','stand_target_velocity','stand_raw_action')
        weights={k:1. for k in effective_keys}
        env=ns(_robot=ns(data=data),_vector_in_command_frame=lambda x:x,
            _get_foot_contact_state=lambda:(torch.ones(n,6,dtype=torch.bool),torch.zeros(n,6,dtype=torch.bool),z(n,6)),
            _coxa_contact_sensor=sensor(6),_femur_contact_sensors=[sensor(1) for _ in range(6)],_feet_contact_sensors=feet,
            _commands=torch.tensor([[.005,0.,0.],[0.,0.,0.]]),omni_targets=torch.tensor([[.005,0.,0.],[0.,0.,0.]]),
            _actions=z(n,18),omni_previous_action=z(n,18),omni_previous_velocity=z(n,18),
            _processed_actions=torch.full((n,18),.001),omni_previous_target=z(n,18),omni_raw_policy_action=torch.ones(n,18),
            cfg=ns(nominal_height_m=0.,omni_reward_weights=weights),step_dt=.02,reset_terminated=torch.zeros(n,dtype=torch.bool),
            omni_diagnostic_enabled=True,omni_measurements_enabled=False,omni_sums={},_episode_elapsed_s=z(n),
            omni_evaluation=True,_get_rewards=lambda:None)
        env.reference_residual_target={'target_position_rad':joints.clone()}
        commands,state,contacts,valid=fixture([[.005,0.,0.],[0.,0.,0.]])
        session=ns(scorable=torch.tensor([True,False]),commands=commands,reference={'state':state},
            last={'measurement':{'distal_contact':contacts,'measurement_valid':valid}})
        captured=[];diag=types.ModuleType('omni_diagnostics')
        def capture(env,terms):
            captured.append({k:v.clone() for k,v in terms.items()});env.omni_diagnostic_sample={}
        diag.capture_step=capture
        repair=types.ModuleType('omni_repair_training')
        repair.stand_raw_action_cost=ast_function(ROOT/'omni_repair_training.py','stand_raw_action_cost',{'torch':torch})
        with patch.dict(sys.modules,{'omni_flat_env':module,'omni_diagnostics':diag,'omni_repair_training':repair}):
            with install_reward(env,session):reward=env._get_rewards()
        self.assertEqual(len(captured),1)
        total=torch.zeros(n)
        for k,value in captured[0].items():
            torch.testing.assert_close(value,session.reward_components[k],rtol=0,atol=0)
            torch.testing.assert_close(env.omni_sums[k],value*.02,rtol=0,atol=0)
            self.assertEqual(float(value[1]),0.)
            total+=value*.02
        torch.testing.assert_close(total,reward,rtol=1e-6,atol=1e-7)
        self.assertEqual(float(captured[0]['stand_joint_velocity'][0]),0.)
        self.assertEqual(float(captured[0]['linear_progress'][0]),1.)
        self.assertEqual(float(reward[1]),0.)
        self.assertEqual(env.omni_diagnostic_sample['quaternion_world_wxyz'].tolist(),[[1.,0.,0.,0.]]*2)

if __name__=='__main__':unittest.main()
