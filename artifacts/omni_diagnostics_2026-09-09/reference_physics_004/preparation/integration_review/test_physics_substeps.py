from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tmp/reference_physics_adapter_003/source_003/tools'))
from physics_substeps import PhysicsSubstepRecorder,control_integrals,displacement_check


def make():
    d=SimpleNamespace(_sim_timestamp=0.,__backend_name__='fake')
    env=SimpleNamespace(_robot=SimpleNamespace(data=d,joint_names=[f'j{i}' for i in range(18)]),
                        cfg=SimpleNamespace(decimation=8),physics_dt=.0025,step_dt=.02,
                        _sim_step_counter=0,_physics_handles_decimation=False)
    state={'p':np.zeros((1,3)),'v':np.zeros((1,3)),'torque':np.zeros((1,18)),'calls':0}
    def update(dt):
        state['calls']+=1;d._sim_timestamp+=dt
        state['v'][:]=[.001 if state['calls']%8==0 else .004,0,0]
        state['p']+=state['v']*dt
        state['torque'][:]=0.1
        if state['calls']%8==3:state['torque'][0,7]=2.
        return 'original_return'
    env.scene=SimpleNamespace(update=update)
    def reader(e):
        return {'root_link_position_world_m':state['p'].copy(),
                'root_link_quaternion_world_xyzw':np.array([[0.,0.,0.,1.]]),
                'root_link_velocity_world_mps':state['v'].copy(),
                'root_com_position_world_m':state['p'].copy()+[0,0,.01],
                'root_com_velocity_world_mps':state['v'].copy(),
                'root_angular_velocity_world_rad_s':np.zeros((1,3)),
                'computed_torque_nm':state['torque'].copy(),
                'applied_torque_nm':np.minimum(state['torque'],1.6),
                'joint_velocity_rad_s':np.zeros((1,18))}
    def tick():
        env._sim_step_counter+=1
        return env.scene.update(dt=.0025)
    def ordinary():
        row=reader(env)
        return {'position_world_m':row['root_link_position_world_m'],
                'quaternion_world_xyzw':row['root_link_quaternion_world_xyzw'],
                'velocity_world_mps':row['root_link_velocity_world_mps'],
                'computed_torque_nm':row['computed_torque_nm'],
                'applied_torque_nm':row['applied_torque_nm']}
    return env,state,reader,tick,ordinary


class SubstepTests(unittest.TestCase):
    def test_real_count_order_timing_endpoint_and_restoration(self):
        env,state,reader,tick,ordinary=make();original=env.scene.update
        with PhysicsSubstepRecorder(env,reader=reader) as r:
            for step in range(2):
                r.begin_control(step)
                for _ in range(8):self.assertEqual(tick(),'original_return')
                r.end_control(ordinary())
        self.assertEqual(env.scene.update,original);self.assertEqual(state['calls'],16)
        self.assertEqual(len(r.rows),17);self.assertEqual(r.completed_controls,2)
        np.testing.assert_allclose(r.data()['time_s'],np.arange(17)*.0025)
        state['p'][:]=999
        self.assertLess(r.rows[-1]['root_link_position_world_m'].max(),1.)

    def test_missing_extra_wrong_dt_and_endpoint_fail_closed_restore(self):
        for case in ('missing','extra','dt','endpoint'):
            env,state,reader,tick,ordinary=make();original=env.scene.update
            with self.assertRaises(RuntimeError):
                with PhysicsSubstepRecorder(env,reader=reader) as r:
                    r.begin_control(0)
                    if case=='dt':env.scene.update(dt=.02)
                    for _ in range(7 if case=='missing' else 8):tick()
                    if case=='extra':tick()
                    row=ordinary()
                    if case=='endpoint':row['position_world_m']+=.01
                    r.end_control(row)
            self.assertEqual(env.scene.update,original)
            self.assertLessEqual(state['calls'],8)

    def test_backend_handles_decimation_cannot_fake_eight_samples(self):
        env,state,reader,_,_=make();env._physics_handles_decimation=True
        with self.assertRaises(ValueError):PhysicsSubstepRecorder(env,reader=reader)
        self.assertEqual(state['calls'],0)

    def test_substep_integral_recovers_known_last_sample_alias_and_torque_spike(self):
        env,state,reader,tick,ordinary=make();state['v'][0,0]=.001
        with PhysicsSubstepRecorder(env,reader=reader) as r:
            for i in range(4):
                r.begin_control(i)
                for _ in range(8):tick()
                r.end_control(ordinary())
        data=r.data();summary=control_integrals(data,4)
        np.testing.assert_allclose(summary['link_velocity_right_integral_m'],summary['link_position_displacement_m'],atol=1e-18)
        report=displacement_check(data,0,4)
        self.assertLess(max(report['link']['integrals']['trapezoid']['position_difference_norm_m']),1e-17)
        ordinary_integral=.001*.08
        self.assertGreater(float(state['p'][0,0])-ordinary_integral,.0002)
        self.assertEqual(summary['computed_torque_abs_substep_max_nm'][:,0,7].min(),2.)
        self.assertEqual(summary['computed_torque_substep_excess_count_1p6'][:,0,7].max(),1)
        self.assertEqual(data['computed_torque_nm'][8::8,:,7].max(),.1)

    def test_exception_and_nonfinite_preserve_first_failing_raw_sample(self):
        env,state,reader,tick,ordinary=make();original=env.scene.update
        with self.assertRaises(ValueError):
            with PhysicsSubstepRecorder(env,reader=reader) as r:
                r.begin_control(0);state['p'][0,0]=float('nan');tick()
        self.assertEqual(env.scene.update,original)
        self.assertEqual(len(r.rows),2)
        self.assertTrue(np.isnan(r.rows[-1]['root_link_position_world_m']).any())

    def test_initial_capture_failure_remains_exportable_without_changing_update(self):
        env,state,reader,_,_=make();original=env.scene.update
        state['p'][0,0]=float('nan')
        r=PhysicsSubstepRecorder(env,reader=reader)
        with self.assertRaises(ValueError):
            with r:pass
        self.assertEqual(env.scene.update,original)
        self.assertEqual(len(r.rows),1);self.assertEqual(r.completed_controls,0)
        with tempfile.TemporaryDirectory() as directory:
            report=r.export(directory)
            self.assertIsNotNone(report['error']);self.assertTrue(report['method_restored'])
            saved=np.load(Path(directory)/'physics_substeps.npz')
            self.assertTrue(np.isnan(saved['root_link_position_world_m']).any())


if __name__=='__main__':unittest.main()
