"""Physical 30-tree/18-motor mapping, closed resets and policy semantics."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import types
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
for package in ("hexapod_core", "hexapod_runtime", "hexapod_env"):
    sys.path.insert(0, str(ROOT / "packages" / package))
sys.path.insert(0, str(ROOT / "tools"))
from hexapod_core import fourbar_v1 as contract
from hexapod_runtime.fourbar_adapter_v1 import FourbarActionPipeline, FourbarJointAdapter, build_fourbar_observation
from hexapod_env.tasks.mkii_fourbar_v1.math import MotorCoordinates, observations, reward_terms
from mkii_fourbar_kinematics import closure_errors, joint_frames, load_model


class FourbarAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin = contract.load_kinematics(ROOT / contract.KINEMATICS_PATH)
        root, reference, _ = load_model()
        cls.frames = joint_frames(root, reference)

    def defaults(self):
        return [self.kin["default_joint_positions_rad"][name] for name in contract.ACTIVE_JOINT_NAMES]

    def test_named_mapping_handles_permutations_without_commanding_passives(self):
        for seed in range(6):
            names = list(contract.TREE_JOINT_NAMES)
            random.Random(seed).shuffle(names)
            adapter = FourbarJointAdapter(names, self.kin)
            data = list(range(30))
            self.assertEqual(adapter.gather_active(data), [names.index(name) for name in contract.ACTIVE_JOINT_NAMES])
            targets = adapter.active_targets([.1]*18)
            self.assertEqual({names[index] for index, _ in targets}, set(contract.ACTIVE_JOINT_NAMES))
            self.assertTrue(set(names[index] for index, _ in targets).isdisjoint(contract.PASSIVE_JOINT_NAMES))

    def test_wrong_duplicate_missing_or_serial_joint_names_fail(self):
        bad_sets = [list(contract.ACTIVE_JOINT_NAMES), list(contract.TREE_JOINT_NAMES[:-1])+[contract.TREE_JOINT_NAMES[0]]]
        bad_sets.append([name.replace("tibia_lever_pivot", "old_motor") for name in contract.TREE_JOINT_NAMES])
        for names in bad_sets:
            with self.assertRaises(ValueError):
                FourbarJointAdapter(names, self.kin)

    def test_closed_reset_randomizes_only_active_coordinates_and_velocities(self):
        names = list(reversed(contract.TREE_JOINT_NAMES))
        adapter = FourbarJointAdapter(names, self.kin)
        batched = MotorCoordinates(names, self.kin, dtype=torch.float64)
        generator = torch.Generator().manual_seed(21)
        positions = batched.default + (torch.rand(12,18,generator=generator,dtype=torch.float64)-.5)*.06
        velocities = torch.randn(12,18,generator=generator,dtype=torch.float64)*.2
        q, qd = batched.closed_reset(positions, velocities)
        self.assertTrue(torch.equal(batched.gather(q), positions))
        self.assertTrue(torch.equal(batched.gather(qd), velocities))
        self.assertEqual(batched.closure_coordinate_error(q).abs().max().item(), 0.)
        for index in range(12):
            scalar_q, scalar_qd = adapter.closed_state(positions[index].tolist(), velocities[index].tolist())
            self.assertEqual(q[index].tolist(), scalar_q)
            self.assertEqual(qd[index].tolist(), scalar_qd)
            # Verify Cartesian pin coincidence from source-derived hinge frames,
            # not merely the same algebra used by the scatter implementation.
            for dt in (-1e-5, 0., 1e-5):
                moved = dict(zip(names, (q[index]+dt*qd[index]).tolist()))
                errors = closure_errors(self.frames, moved)
                self.assertLess(max(row["point_error_m"] for row in errors.values()), 1e-12)
                self.assertLess(max(row["axis_error_rad"] for row in errors.values()), 1e-12)

    def test_independent_passive_jitter_is_detected_and_nonfinite_reset_rejected(self):
        adapter = MotorCoordinates(contract.TREE_JOINT_NAMES, self.kin)
        q, qd = adapter.closed_reset(adapter.default[None], torch.zeros(1,18))
        q[0,contract.TREE_JOINT_NAMES.index("lf_tibia_pitch")] += .01
        self.assertGreater(adapter.closure_coordinate_error(q).abs().max().item(), .009)
        for field in ("position", "velocity"):
            positions, velocities = adapter.default[None].clone(), torch.zeros(1,18)
            (positions if field == "position" else velocities)[0,0] = math.nan
            with self.assertRaises(ValueError):
                adapter.closed_reset(positions, velocities)

    def test_kinematics_contract_rejects_wrong_passive_relation_and_stale_defaults(self):
        for mutation in ("relation", "default", "active", "schema"):
            kin = copy.deepcopy(self.kin)
            if mutation == "relation":
                kin["passive_relations"]["lf_tibia_pitch"]["multiplier"] = -1.
            elif mutation == "default":
                kin["default_joint_positions_rad"]["lf_tibia_pitch"] += .1
            elif mutation == "active":
                kin["active_joint_names"][12] = "lf_tibia_pitch"
            else:
                kin["schema"] = "serial"
            with self.assertRaises(ValueError):
                contract.validate_kinematics(kin)

    def manifest(self):
        return contract.runtime_manifest(self.kin, {"model_id": "test_motor", "peak": 5.5},
            kinematics_sha256=hashlib.sha256(json.dumps(self.kin,sort_keys=True).encode()).hexdigest(), usd_sha256="a"*64)

    def test_runtime_and_torch_action_slew_parity_with_measured_reset(self):
        manifest = self.manifest()
        runtime = FourbarActionPipeline(manifest, expected_manifest=copy.deepcopy(manifest))
        runtime.reset(self.defaults(), motor_names=contract.ACTIVE_JOINT_NAMES)
        coord = MotorCoordinates(contract.TREE_JOINT_NAMES, self.kin, dtype=torch.float64)
        target = coord.default[None].clone()
        for action in ([.5]*18, [-1.2]*18, [0.]*18, [.2,-.8,.7]*6):
            _, target, _ = coord.process_action(torch.tensor([action],dtype=torch.float64), target, step_dt=.02)
            actual = runtime.step(action, [.1,0.,0.], motor_names=contract.ACTIVE_JOINT_NAMES,
                                  command_frame=contract.COMMAND_FRAME)
            torch.testing.assert_close(target[0], torch.tensor(actual,dtype=torch.float64), rtol=0, atol=1e-12)

    def test_manifest_binds_motor_asset_semantics_and_runtime_requires_reset(self):
        manifest = self.manifest()
        for key, value in (("usd_root_sha256", "b"*64), ("motor_contract_sha256", "c"*64),
                           ("observation_dim", 66), ("active_motor_names", list(reversed(contract.ACTIVE_JOINT_NAMES)))):
            changed = copy.deepcopy(manifest)
            changed[key] = value
            with self.assertRaises(ValueError):
                FourbarActionPipeline(changed, expected_manifest=manifest)
        runtime = FourbarActionPipeline(manifest, expected_manifest=manifest)
        with self.assertRaises(RuntimeError):
            runtime.step([0.]*18,[0.]*3,motor_names=contract.ACTIVE_JOINT_NAMES,command_frame=contract.COMMAND_FRAME)

    def test_observation_is84_and_navigation_motor_semantics_match_runtime(self):
        coord = MotorCoordinates(contract.TREE_JOINT_NAMES, self.kin, dtype=torch.float64)
        q = coord.default[None] + .01
        qd = torch.arange(18,dtype=torch.float64)[None]/10.
        zero3 = torch.zeros(1,3,dtype=torch.float64)
        action = torch.arange(18,dtype=torch.float64)[None]/20.
        reserve = torch.full((1,18),.5,dtype=torch.float64)
        obs = observations(torch.tensor([[0.,-1.,0.]],dtype=torch.float64),zero3,
            torch.tensor([[0.,0.,-1.]],dtype=torch.float64),zero3,q-coord.default,qd,action,reserve)
        self.assertEqual(obs.shape,(1,84))
        self.assertEqual(obs[0,:3].tolist(),[1.,0.,0.])
        fields = {}
        offset = 0
        for name,width in contract.OBSERVATION_FIELDS:
            fields[name] = obs[0,offset:offset+width].tolist()
            offset += width
        self.assertEqual(build_fourbar_observation(**fields),obs[0].tolist())
        fields["estimated_motor_burst_headroom"][3] = math.nan
        with self.assertRaises(ValueError):
            build_fourbar_observation(**fields)
        with self.assertRaises(ValueError):
            observations(zero3,zero3,zero3,zero3,torch.zeros(1,30),qd,action,reserve)

    def test_rewards_use18_motor_coordinates_and_keep_each_axis_sensitive(self):
        coord = MotorCoordinates(contract.TREE_JOINT_NAMES, self.kin)
        zero3, zero18 = torch.zeros(2,3), torch.zeros(2,18)
        values = dict(command=zero3.clone(),linear_navigation=zero3.clone(),angular_navigation=zero3.clone(),
            gravity_body=torch.tensor([[0.,0.,-1.]]).repeat(2,1),active_torque=zero18.clone(),
            active_velocity=zero18,active_acceleration=zero18,active_position=coord.default.repeat(2,1),
            soft_limits=coord.soft_limits,action=zero18,previous_action=zero18,height=torch.full((2,),.138),
            nominal_height=.138,nonfoot_contacts=torch.zeros(2),support_count=torch.full((2,),6),
            foot_slip=torch.zeros(2),clipping_nm=zero18,overload_nm=zero18)
        values["active_torque"][1,17] = 5.
        base = reward_terms(**values)
        self.assertLess(base["motor_torque"][1],base["motor_torque"][0])
        for axis in (0,1,2):
            moved = copy.deepcopy(values)
            moved["command"][0,axis] = .1
            bad = reward_terms(**moved)
            key = "track_yaw" if axis==2 else "track_linear"
            self.assertLess(bad[key][0],base[key][0])
        values["active_torque"] = torch.zeros(2,30)
        with self.assertRaises(ValueError):
            reward_terms(**values)

    def test_new_environment_has_no_legacy_environment_or_gait_clock(self):
        path = ROOT/"packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/env.py"
        tree = ast.parse(path.read_text())
        cls = next(node for node in tree.body if isinstance(node,ast.ClassDef))
        self.assertEqual(ast.unparse(cls.bases[0]),"DirectRLEnv")
        self.assertFalse(any(isinstance(node,ast.Attribute) and "gait" in node.attr for node in ast.walk(tree)))


class EnvironmentWiringTests(unittest.TestCase):
    """Execute the actual env methods with CPU data and a recording SDK boundary."""
    @classmethod
    def setUpClass(cls):
        path = ROOT/"packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/env.py"
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))]
        class Base:
            def _reset_idx(self, env_ids):
                self.base_reset_ids = env_ids.clone()
        namespace = {"DirectRLEnv":Base,"contract":contract,"torch":torch,
                     "observations":observations,"reward_terms":reward_terms,
                     "body_to_navigation_frame":lambda value: torch.stack((-value[...,1],value[...,0],value[...,2]),-1)}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),namespace)
        cls.Env = namespace["HexapodMkiiFourbarEnv"]
        # Execute the production constructor's exact SDK-index assignment.
        # The remaining constructor needs Kit; copying a dtype into the stub
        # would conceal a regression in the actual runtime conversion.
        env_class = next(node for node in nodes if isinstance(node, ast.ClassDef))
        initializer = next(node for node in env_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        assignments = [node for node in initializer.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Attribute) and target.attr == "active_joint_ids" for target in node.targets)]
        if len(assignments) != 1:
            raise AssertionError("Expected one explicit active joint SDK-index assignment")
        cls.api_index_assignment = compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec")
        cls.kin = contract.load_kinematics(ROOT/contract.KINEMATICS_PATH)

    def environment(self):
        raw = self.Env.__new__(self.Env)
        raw.device,raw.num_envs = "cpu",3
        names = list(reversed(contract.TREE_JOINT_NAMES))
        raw.coordinates = MotorCoordinates(names,self.kin)
        exec(self.api_index_assignment, {"self": raw, "torch": torch})
        raw.active_joint_names = list(contract.ACTIVE_JOINT_NAMES)
        q,qd = raw.coordinates.closed_reset(raw.coordinates.default.repeat(3,1),torch.zeros(3,18))
        data = types.SimpleNamespace(joint_pos=q.clone(),joint_vel=qd,joint_acc=torch.zeros(3,30),
            applied_torque=torch.zeros(3,30),computed_torque=torch.zeros(3,30),
            default_root_pose=torch.tensor([[0.,0.,.14297,1.,0.,0.,0.]]).repeat(3,1),
            default_root_vel=torch.zeros(3,6),root_pos_w=torch.tensor([[0.,0.,.138]]).repeat(3,1),
            root_lin_vel_b=torch.zeros(3,3),root_ang_vel_b=torch.zeros(3,3),
            projected_gravity_b=torch.tensor([[0.,0.,-1.]]).repeat(3,1))
        raw._robot = types.SimpleNamespace(data=data,reset=mock.Mock(),
            set_joint_position_target_index=mock.Mock(),write_root_pose_to_sim_index=mock.Mock(),
            write_root_velocity_to_sim_index=mock.Mock(),write_joint_position_to_sim_index=mock.Mock(),
            write_joint_velocity_to_sim_index=mock.Mock())
        def check_sdk_write(**kwargs):
            for name in ("env_ids", "joint_ids"):
                if name in kwargs:
                    self.assertEqual(kwargs[name].dtype, torch.int32, name + " must match Warp int32 kernels")
        raw._robot.reset.side_effect = lambda ids: self.assertEqual(ids.dtype, torch.int32)
        for name in ("set_joint_position_target_index", "write_root_pose_to_sim_index",
                     "write_root_velocity_to_sim_index", "write_joint_position_to_sim_index",
                     "write_joint_velocity_to_sim_index"):
            getattr(raw._robot, name).side_effect = check_sdk_write
        raw._terrain=types.SimpleNamespace(env_origins=torch.tensor([[0.,0.,0.],[2.,0.,0.],[4.,0.,0.]]))
        raw.cfg=types.SimpleNamespace(reset_joint_jitter_rad=.03,nominal_height_m=.138,command_hold_time_s=4.)
        raw._actions=torch.full((3,18),.2)
        raw._previous_actions=torch.full((3,18),.1)
        raw._processed_actions=raw.coordinates.default.repeat(3,1)
        raw._joint_target_slew_limited_fraction=torch.ones(3)
        raw._commands=torch.zeros(3,3)
        raw._command_time_left_s=torch.full((3,),4.)
        raw._standing_only=True
        raw._reward_sums={"test":torch.ones(3)}
        raw.max_episode_length_s=20.
        raw.step_dt=.02
        raw.extras={}
        raw._motor_order=torch.arange(17,-1,-1)
        raw._motor_model=types.SimpleNamespace(burst_headroom=torch.arange(18).repeat(3,1)/18.,
            clipping_nm=torch.zeros(3,18),continuous_overload_nm=torch.zeros(3,18))
        raw.reset_terminated=torch.zeros(3,dtype=torch.bool)
        raw.reset_buf=torch.zeros(3,dtype=torch.bool)
        raw._contact_state=lambda:(torch.ones(3,6,dtype=torch.bool),torch.zeros(3,6,dtype=torch.bool),
                                   torch.zeros(3,6),torch.zeros(3,dtype=torch.bool))
        raw._nonfoot_contact_counts=lambda _:(torch.zeros(3),torch.zeros(3,dtype=torch.bool))
        return raw

    def test_actual_apply_action_writes_only18_named_motors(self):
        raw=self.environment()
        raw._apply_action()
        kwargs=raw._robot.set_joint_position_target_index.call_args.kwargs
        self.assertEqual(kwargs["target"].shape,(3,18))
        self.assertEqual(kwargs["joint_ids"].tolist(),list(contract.active_indices(raw.coordinates.names)))
        self.assertEqual(kwargs["joint_ids"].dtype, torch.int32)
        self.assertEqual(raw.coordinates.active_indices.dtype, torch.int64)

    def test_actual_partial_reset_writes_closed30_positions_and_velocities(self):
        raw=self.environment()
        original_pose=raw._robot.data.default_root_pose.clone()
        other_action=raw._actions[1].clone()
        raw._reset_idx(torch.tensor([0,2]))
        q=raw._robot.write_joint_position_to_sim_index.call_args.kwargs["position"]
        qd=raw._robot.write_joint_velocity_to_sim_index.call_args.kwargs["velocity"]
        self.assertEqual(q.shape,(2,30))
        self.assertEqual(qd.shape,(2,30))
        self.assertEqual(raw.coordinates.closure_coordinate_error(q).abs().max().item(),0.)
        self.assertTrue(torch.equal(qd,torch.zeros_like(qd)))
        self.assertTrue(torch.equal(original_pose,raw._robot.data.default_root_pose))
        self.assertTrue(torch.equal(other_action,raw._actions[1]))
        self.assertTrue(torch.equal(raw._processed_actions[[0,2]],raw.coordinates.gather(q)))
        self.assertEqual(raw._robot.set_joint_position_target_index.call_args.kwargs["target"].shape,(2,18))
        self.assertEqual(raw.base_reset_ids.dtype, torch.int32)

    def test_actual_reset_normalizes_all_and_selected_sdk_indices_to_int32(self):
        for supplied, expected in ((None, [0,1,2]),
                (torch.tensor([2,0], dtype=torch.int64), [2,0]),
                (torch.tensor([1], dtype=torch.int32), [1]),
                (types.SimpleNamespace(torch=torch.tensor([0,2], dtype=torch.int64)), [0,2])):
            raw = self.environment()
            raw._reset_idx(supplied)
            ids = raw._robot.reset.call_args.args[0]
            self.assertEqual(ids.dtype, torch.int32)
            self.assertEqual(ids.tolist(), expected)
            self.assertEqual(raw.base_reset_ids.dtype, torch.int32)
            for name in ("write_root_pose_to_sim_index", "write_root_velocity_to_sim_index",
                         "write_joint_position_to_sim_index", "write_joint_velocity_to_sim_index",
                         "set_joint_position_target_index"):
                kwargs = getattr(raw._robot, name).call_args.kwargs
                self.assertEqual(kwargs["env_ids"].dtype, torch.int32, name)
                self.assertEqual(kwargs["env_ids"].tolist(), expected, name)
            self.assertEqual(raw._robot.set_joint_position_target_index.call_args.kwargs["joint_ids"].dtype, torch.int32)

    def test_actual_observation_and_reward_ignore_passive_motor_impersonation(self):
        raw=self.environment()
        observation=raw._get_observations()["policy"]
        self.assertEqual(observation.shape,(3,84))
        self.assertTrue(torch.equal(observation[:,-18:],raw._motor_model.burst_headroom[:,raw._motor_order]))
        first=raw._get_rewards().clone()
        passives=[index for index,name in enumerate(raw.coordinates.names) if name in contract.PASSIVE_JOINT_NAMES]
        raw._robot.data.applied_torque[:,passives]=1e6
        raw._robot.data.joint_acc[:,passives]=1e6
        raw._robot.data.joint_vel[:,passives]=1e6
        second=raw._get_rewards()
        self.assertTrue(torch.equal(first,second))
        self.assertTrue(torch.equal(observation,raw._get_observations()["policy"]))


if __name__ == "__main__":
    unittest.main()
