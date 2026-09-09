"""CPU complete-row comparison of an unapplied PhysicalMetrics batching patch."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

HERE = Path(__file__).resolve().parent
SETTINGS = globals().get('_COMPARISON_SETTINGS',{})
ROOT = Path(SETTINGS['source_dir'] if 'source_dir' in SETTINGS else HERE.parents[2]).resolve()
DEVICE = SETTINGS.get('device','cpu')
SCALAR_INDEX = 3 if SETTINGS.get('sdk_xyzw') else 0
sys.path[:0] = [str(ROOT/'isaaclab/tests'),str(ROOT/'tools'),str(ROOT/'isaaclab')]
if SETTINGS:
    FakeRobot=SETTINGS['fake_robot_class']
    matrix_from_quat=SETTINGS['matrix_from_quat']
else:
    from test_validate_mkii_fourbar_metrics import FakeRobot, matrix_from_quat

BASELINE_SHA = '2007a7d84d9c51888a67df8bec7292924b2e57411d9312e6e5607c713617725c'
CANDIDATE_SHA = '811726701377ec9cc55672ac5df0f417bcb22eb112258bda259feed177becdb3'


def load(name,path):
    loader=importlib.machinery.SourceFileLoader(name,str(path))
    spec=importlib.util.spec_from_loader(name,loader)
    module=importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    # Copies retain the production ROOT declaration. Only this CPU fixture
    # redirects their source/URDF lookup to the unchanged surrounding checkout.
    module.ROOT=ROOT
    return module


baseline=load('metrics_integration_baseline',Path(SETTINGS.get('baseline',HERE/'baseline_validate_mkii_fourbar.py.txt')))
candidate=load('metrics_integration_candidate',Path(SETTINGS.get('candidate',HERE/'proposed_validate_mkii_fourbar.py.txt')))
trainer=load('metrics_integration_guard',ROOT/'isaaclab/train_mkii_fourbar.py')
KIN=json.loads((ROOT/'configs/mkii_fourbar_v3_kinematics.json').read_text())


class CountingSensor:
    def __init__(self,name,force,log):
        self.name,self.force,self.log=name,force,log
        self.cfg=types.SimpleNamespace(update_period=baseline.PHYSICS_DT_S)

    @property
    def data(self):
        self.log.append(self.name)
        return types.SimpleNamespace(net_forces_w=self.force)


class Scene:
    def __init__(self,raw):
        self.raw,self.ticks=raw,0
        self.change=None

    def update(self,dt):
        self.ticks+=1
        if self.change:self.change(self.raw,self.ticks)


class SyntheticRaw(FakeRobot):
    def __init__(self,count,seed=0):
        # Reuse the existing independent CAD forward-kinematics fixture. Its
        # matrix/quaternion conversion is not a native Isaac implementation.
        super().__init__(KIN,tilt=True)
        generator=torch.Generator().manual_seed(seed)
        self.num_envs=count
        data=self._robot.data
        for name,value in list(vars(data).items()):
            setattr(data,name,value.repeat(count//2,*([1]*(value.ndim-1))).clone())
        self._terrain.env_origins=self._terrain.env_origins.repeat(count//2,1).clone()
        body_order=torch.randperm(31,generator=generator)
        old_names=list(self.names)
        self.names=[old_names[i] for i in body_order.tolist()]
        self._robot.body_names=list(self.names)
        for name,value in list(vars(data).items()):
            if name.startswith('body_'):setattr(data,name,value.index_select(1,body_order))
        data.root_pos_w=data.body_link_pos_w[:,self.names.index('body')]
        joint_order=torch.randperm(30,generator=generator)
        old_joint_names=list(self._robot.joint_names)
        self._robot.joint_names=[old_joint_names[i] for i in joint_order.tolist()]
        data.joint_pos=data.joint_pos.index_select(1,joint_order)
        data.joint_vel=data.joint_vel.index_select(1,joint_order)
        self.acquisition_log=[]
        sensor_order=torch.randperm(31,generator=generator).tolist()
        sensor_names=[self.names[i] for i in sensor_order]
        self._body_contact_sensors={name:CountingSensor(name,
            torch.tensor([0.,0.,2. if name.endswith('_tibia') else 0.]).expand(count,1,3).clone(),
            self.acquisition_log) for name in sensor_names}
        self._applied=torch.full((count,18),.8)
        self._computed=torch.full((count,18),.9)
        self._telemetry={name:torch.full((count,18),value) for name,value in
            {'instantaneous_limit_nm':1.2,'continuous_limit_nm':1.2,'burst_headroom':1.,'invalid_input':False}.items()}
        self._closure_error=torch.zeros(count,30)
        self._physics_handles_decimation=False
        self.cfg=types.SimpleNamespace(sim=types.SimpleNamespace(dt=baseline.PHYSICS_DT_S),decimation=baseline.DECIMATION)
        self.common_step_counter=0
        self.scene=Scene(self)
        self.reset_events=[]
        self.initial={name:value.clone() for name,value in vars(data).items()}
        self.device=DEVICE
        for name,value in vars(data).items():setattr(data,name,value.to(DEVICE))
        data.root_pos_w=data.body_link_pos_w[:,self.names.index('body')]
        self._terrain.env_origins=self._terrain.env_origins.to(DEVICE)
        for sensor in self._body_contact_sensors.values():sensor.force=sensor.force.to(DEVICE)
        self._applied=self._applied.to(DEVICE);self._computed=self._computed.to(DEVICE)
        self._telemetry={name:value.to(DEVICE) for name,value in self._telemetry.items()}
        self._closure_error=self._closure_error.to(DEVICE)
        self.initial={name:value.to(DEVICE) for name,value in self.initial.items()}

    def motor_state(self,name):
        return self._applied if name=='applied_torque' else self._computed

    def motor_telemetry(self,name):
        return self._telemetry[name]

    def closure_coordinate_error(self):
        return self._closure_error

    def randomize(self,seed):
        generator=torch.Generator().manual_seed(seed)
        data=self._robot.data
        for name in ('body_link_pos_w','body_link_lin_vel_w','body_link_ang_vel_w','joint_pos','joint_vel'):
            value=getattr(data,name)
            value.copy_(torch.randn(value.shape,generator=generator))
        quat=torch.randn(data.body_link_quat_w.shape,generator=generator)
        data.body_link_quat_w.copy_(quat/torch.linalg.vector_norm(quat,dim=-1,keepdim=True))
        for sensor in self._body_contact_sensors.values():
            sensor.force.copy_(torch.randn(sensor.force.shape,generator=generator))
        self._applied.copy_(torch.rand(self._applied.shape,generator=generator)*5.5)
        self._computed.copy_(torch.randn(self._computed.shape,generator=generator)*4.)
        self._closure_error.copy_(torch.randn(self._closure_error.shape,generator=generator)*.001)

    def reset_rows(self,ids):
        for name,value in vars(self._robot.data).items():
            value[ids]=self.initial[name][ids]
        self._closure_error[ids]=0
        for name,sensor in self._body_contact_sensors.items():
            sensor.force[ids]=sensor.force.new_tensor([0.,0.,2. if name.endswith('_tibia') else 0.])
        self.reset_events.append((self.scene.ticks,tuple(ids)))


class CompleteMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        if DEVICE=='cpu':assert not torch.cuda.is_initialized()
        else:assert torch.cuda.is_available()

    def setUp(self):
        sdk=types.ModuleType('isaaclab.utils.math')
        sdk.matrix_from_quat=matrix_from_quat
        key='isaaclab.utils.math'
        missing=object()
        previous=sys.modules.get(key,missing)
        sys.modules[key]=sdk
        def restore_sdk_key():
            if previous is missing:sys.modules.pop(key,None)
            else:sys.modules[key]=previous
        # Preserve unrelated modules imported lazily by torch.testing: removing
        # those Python modules cannot undo their C++ operator registrations.
        self.addCleanup(restore_sdk_key)

    def pair(self,count=32,seed=0,kinematics=None):
        raw=SyntheticRaw(count,seed)
        return raw,baseline.PhysicalMetrics(raw,kinematics or KIN),candidate.PhysicalMetrics(raw,kinematics or KIN)

    def assert_rows(self,left,right):
        self.assertEqual(left.shape,(18,))
        self.assertEqual(left.dtype,torch.float64)
        self.assertEqual(right.dtype,left.dtype)
        self.assertTrue(torch.equal(torch.isnan(left),torch.isnan(right)))
        summary=SETTINGS.get('comparison_summary')
        if summary is not None:
            a,b=left.detach().cpu(),right.detach().cpu()
            differences=(a-b).abs().nan_to_num(nan=0.)
            summary['row_comparisons']+=1
            summary['max_absolute_difference_by_field']=[max(prior,float(value)) for prior,value in
                zip(summary['max_absolute_difference_by_field'],differences)]
        torch.testing.assert_close(left,right,rtol=0,atol=0,equal_nan=True)
        self.assertTrue(torch.equal(left.contiguous().view(torch.uint8),right.contiguous().view(torch.uint8)))

    def capture_pair(self,raw,a,b):
        expected=list(raw._body_contact_sensors)
        for metrics in (a,b):
            raw.acquisition_log.clear()
            metrics.capture()
            self.assertEqual(raw.acquisition_log,expected)
        self.assert_rows(a.pending[-1],b.pending[-1])

    def test_exact_baseline_patch_scope_and_unchanged_public_helpers_draining_grading(self):
        before=Path(SETTINGS.get('baseline',HERE/'baseline_validate_mkii_fourbar.py.txt')).read_bytes()
        after=Path(SETTINGS.get('candidate',HERE/'proposed_validate_mkii_fourbar.py.txt')).read_bytes()
        self.assertEqual(hashlib.sha256(before).hexdigest(),BASELINE_SHA)
        self.assertEqual(hashlib.sha256(after).hexdigest(),CANDIDATE_SHA)
        self.assertEqual((ROOT/'isaaclab/validate_mkii_fourbar.py').read_bytes(),before)
        def excluded(text):
            tree=ast.parse(text)
            metrics=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='PhysicalMetrics')
            metrics.body=[node for node in metrics.body if not(isinstance(node,ast.FunctionDef) and node.name in ('__init__','capture'))]
            return ast.dump(tree,include_attributes=False)
        self.assertEqual(excluded(before),excluded(after))

    def test_complete_randomized_rows_at_32_and_512_with_independent_orders(self):
        for count in (32,512):
            for seed in (0,3,19):
                with self.subTest(count=count,seed=seed):
                    raw,a,b=self.pair(count,seed)
                    self.assertNotEqual(list(raw._body_contact_sensors),raw._robot.body_names)
                    for sample in range(baseline.DECIMATION):
                        raw.randomize(seed*100+sample)
                        self.capture_pair(raw,a,b)
                    a.drain();b.drain()
                    self.assertEqual(a.windows,b.windows)
                    self.assertEqual(a.velocity_telemetry_description,b.velocity_telemetry_description)

    def test_closure_and_contact_threshold_neighbours_match_in_complete_rows(self):
        kinematics=copy.deepcopy(KIN)
        for name in kinematics['closure_joint_names']:
            for side in (0,1):
                kinematics['joint_frames'][name][f'body{side}_from_hinge_matrix']=torch.eye(4).tolist()
        threshold=torch.tensor(.0001)
        force_threshold=torch.tensor(1.)
        for direction in (-1,0,1):
            value=threshold if direction==0 else torch.nextafter(threshold,torch.tensor(float('inf')*direction))
            force=force_threshold if direction==0 else torch.nextafter(force_threshold,torch.tensor(float('inf')*direction))
            raw,a,b=self.pair(32,7,kinematics)
            data=raw._robot.data
            data.body_link_quat_w.zero_();data.body_link_quat_w[...,SCALAR_INDEX]=1
            data.body_link_pos_w.zero_()
            for pair in a.frames:data.body_link_pos_w[:,pair[1][0],0]=value
            for sensor in raw._body_contact_sensors.values():
                sensor.force.zero_();sensor.force[...,0]=force
            for _ in range(baseline.DECIMATION):self.capture_pair(raw,a,b)
            a.drain();b.drain();self.assertEqual(a.windows,b.windows)
            row=a.windows['startup']
            self.assertEqual(row['max_closure_point_m']>.0001,direction==1)
            self.assertEqual(row['min_support'],6 if direction==1 else 0)

    def test_nonfinite_force_is_preserved_and_early_bad_sample_survives_buffer_reset(self):
        for count in (32,512):
            for value in (float('nan'),float('inf'),float('-inf')):
                with self.subTest(count=count,value=value):
                    raw,a,b=self.pair(count,9)
                    sensors=list(raw._body_contact_sensors.values())
                    sensors[0].force[0,0,0]=value
                    self.capture_pair(raw,a,b)
                    sensors[0].force[0,0,0]=0.
                    for _ in range(baseline.DECIMATION-1):self.capture_pair(raw,a,b)
                    a.drain();b.drain();self.assertEqual(a.windows,b.windows)
                    self.assertEqual(a.windows['startup']['invalid_samples'],1.)

    def test_repeated_drains_windows_and_partial_resets_preserve_full_history(self):
        for count in (32,512):
            raw,a,b=self.pair(count,5)
            passive=raw._robot.joint_names.index('lf_tibia_rod_pivot')
            for cycle,window in enumerate(('startup','settled','settled','driven')):
                a.window=b.window=window
                for step in range(baseline.DECIMATION):
                    raw._robot.data.joint_vel.zero_()
                    if step==0:raw._robot.data.joint_vel[1,passive]=2.+cycle
                    self.capture_pair(raw,a,b)
                raw._robot.data.joint_vel.zero_()
                a.drain();b.drain();self.assertEqual(a.windows,b.windows)
                raw.reset_rows([0,count-1])
            self.assertEqual(a.windows['settled']['substeps'],2*baseline.DECIMATION)
            self.assertEqual(a.windows['settled']['max_passive_velocity_relation_error_rad_s'],4.)
            self.assertEqual(a.windows['settled']['passive_velocity_relation_samples'],2*baseline.DECIMATION*count*12)

    def test_missing_or_extra_captures_are_rejected_by_unchanged_drain(self):
        for samples in (baseline.DECIMATION-1,baseline.DECIMATION+1):
            raw,a,b=self.pair()
            for _ in range(samples):self.capture_pair(raw,a,b)
            for metrics in (a,b):
                with self.assertRaisesRegex(ValueError,'every physical sample'):metrics.drain()

    def test_real_training_guard_retains_every_sample_across_partial_resets(self):
        results=[]
        for module in (baseline,candidate):
            raw=SyntheticRaw(32,4)
            metrics=module.PhysicalMetrics(raw,KIN)
            metrics.window='learning'
            seen=[];capture=metrics.capture
            def record():seen.append(raw.scene.ticks);capture()
            metrics.capture=record
            with trainer.PhysicalTrainingGuard(raw,metrics) as guard:
                for policy in range(3):
                    raw.common_step_counter=policy
                    for _ in range(baseline.DECIMATION):raw.scene.update(baseline.PHYSICS_DT_S)
                    raw.reset_rows([0,31])
                guard.require_coverage(3)
            self.assertEqual(seen,list(range(1,3*baseline.DECIMATION+1)))
            self.assertEqual(raw.reset_events,[(baseline.DECIMATION*i,(0,31)) for i in (1,2,3)])
            self.assertNotIn('update',vars(raw.scene))
            results.append(metrics.windows)
        self.assertEqual(results[0],results[1])

    def test_real_guard_rejects_early_closure_fault_before_automatic_reset(self):
        for module in (baseline,candidate):
            raw=SyntheticRaw(32,6)
            metrics=module.PhysicalMetrics(raw,KIN)
            endpoint=metrics.frames[0][0][0]
            def change(robot,tick):
                if tick==1:robot._robot.data.body_link_pos_w[0,endpoint,0]+=.001
                if tick==2:robot._robot.data.body_link_pos_w[0,endpoint,0]-=.001
            raw.scene.change=change
            with self.assertRaisesRegex(ValueError,'max_closure_point_m'):
                with trainer.PhysicalTrainingGuard(raw,metrics):
                    for _ in range(baseline.DECIMATION):raw.scene.update(baseline.PHYSICS_DT_S)
                    raw.reset_rows([0])
            self.assertEqual(raw.reset_events,[])
            self.assertEqual(metrics.windows['startup']['substeps'],baseline.DECIMATION)
            self.assertGreater(metrics.windows['startup']['max_closure_point_m'],.0001)
            self.assertNotIn('update',vars(raw.scene))


if __name__=='__main__':unittest.main()
