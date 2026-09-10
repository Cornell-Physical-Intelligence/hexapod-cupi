from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from observation import ObservationBuilder,Limits,SOURCE_CONTRACT

HERE=Path(__file__).resolve().parent
FIXTURES=json.loads((HERE/'fixtures.json').read_text())


def sample(kind='standing',episode='episode-0',step=0):
    p=deepcopy(FIXTURES[kind]);p['episode_id']=episode;p['step_index']=step;p['time_s']=step*.02
    p['measurement']['time_s']=[p['time_s']]
    p['reference']['target_time_s']=p['time_s'];p['reference']['state']['desired_time_s']=p['time_s']
    return p


def builder(n=1,names=None,limits=Limits()):
    names=names or FIXTURES['standing']['joint_names_runtime']
    b=ObservationBuilder(names,SOURCE_CONTRACT['nominal_joint_positions'],n,limits=limits)
    b.reset(list(range(n)),['episode-0']*n,[0.]*n)
    return b


def world_transform(packet,Q,shift):
    p=deepcopy(packet);m=p['measurement'];s=p['reference']['state']
    def point(x):return (np.asarray(x)@Q.T+shift).tolist()
    def vector(x):return (np.asarray(x)@Q.T).tolist()
    m['position_world_m']=point(m['position_world_m'])
    m['rotation_world_from_body']=(Q@np.asarray(m['rotation_world_from_body'])).tolist()
    m['reference_point_world_m']=point(m['reference_point_world_m'])
    m['reference_point_velocity_world_mps']=vector(m['reference_point_velocity_world_mps'])
    cp=np.asarray(m['contact_point_world_m']);mask=np.asarray(m['contact_point_valid'])
    cp[mask]=cp[mask]@Q.T+shift;m['contact_point_world_m']=cp.tolist()
    for key in ('desired_position_world_m','reference_anchors_world_m','measured_anchors_world_m'):
        s[key]=point(s[key])
    s['reference_minus_measured_preload_world_m']=vector(s['reference_minus_measured_preload_world_m'])
    for key in ('desired_rotation_world_from_body','initial_desired_rotation_world_from_body'):
        s[key]=(Q@np.asarray(s[key])).tolist()
    if s['landing_contact_origin_world_m'] is not None:s['landing_contact_origin_world_m']=point(s['landing_contact_origin_world_m'])
    if s['landing_preload_world_m'] is not None:s['landing_preload_world_m']=vector(s['landing_preload_world_m'])
    up=Q@np.asarray(p['world_up_vector']);p['world_up_vector']=up.tolist()
    for key in ('flight_baseline_z_m','flight_peak_z_m'):
        if s[key] is not None:s[key]+=float(shift@up)
    for key in ('swing','landing'):
        t=s[key]
        if t is not None:
            t['endpoint_world_m']=point(t['endpoint_world_m'])
            coeff=np.asarray(t['coefficients'])@Q.T;coeff[0]+=shift;t['coefficients']=coeff.tolist()
    return p


class ObservationTests(unittest.TestCase):
    def test_width_is_computed_and_slices_contiguous_for_standing_and_landing(self):
        for kind in ('standing','landing'):
            out=builder().build([sample(kind)]);spec=out['schema'];end=0
            for f in spec['fields']:
                self.assertEqual(f['start'],end);end+=f['width'];self.assertEqual(f['stop'],end)
            self.assertEqual(end,out['policy'].shape[1]);self.assertEqual(out['critic'].shape[1],end+3)
            self.assertEqual(end,740);self.assertFalse(out['policy_training_allowed'])

    def test_common_translation_yaw_and_full_coordinate_rotation_invariant(self):
        for kind in ('standing','landing'):
            p=sample(kind);expected=builder().build([p])['policy']
            for angles,shift in [([0,0,143],[12,-8,2]),([31,-24,143],[12,-8,2]),([0,0,180],[0,0,0])]:
                Q=Rotation.from_euler('xyz',angles,degrees=True).as_matrix()
                changed=world_transform(p,Q,np.asarray(shift,dtype=float))
                np.testing.assert_allclose(builder().build([changed])['policy'],expected,atol=2e-6,rtol=0)

    def test_same_step_identical_and_readback_cannot_mutate_history(self):
        b=builder();p=sample();first=b.build([p]);stored=first['policy'].copy()
        first['policy'][:]=999
        np.testing.assert_array_equal(b.build([p])['policy'],stored)
        p['requested_twist'][0]=.01
        with self.assertRaisesRegex(ValueError,'Same-step'):b.build([p])
        self.assertEqual(int(b.valid.sum()),1)

    def test_history_clock_padding_and_partial_reset_do_not_leak(self):
        b=builder(2)
        for step in range(3):b.build([sample(step=step),sample(step=step)])
        self.assertEqual(b.valid.sum(axis=1).tolist(),[3,3]);other=b.history[1].copy()
        b.reset([0],['new-episode'],[0.])
        self.assertEqual(int(b.valid[0].sum()),0);np.testing.assert_array_equal(b.history[1],other)
        out=b.build([sample(episode='new-episode'),sample(step=3)])
        self.assertEqual(b.valid.sum(axis=1).tolist(),[1,4]);self.assertTrue((b.history[0,:4]==0).all())
        with self.assertRaisesRegex(ValueError,'Episode mismatch'):b.build([sample(),sample(step=3)])
        with self.assertRaisesRegex(ValueError,'fresh episode'):b.reset([0],['episode-0'],[0.])

    def test_missed_reordered_and_stale_samples_rejected_transactionally(self):
        b=builder(2);b.build([sample(),sample()]);saved=b.history.copy()
        bad=sample(step=2)
        with self.assertRaisesRegex(ValueError,'Missing/stale'):b.build([sample(step=1),bad])
        np.testing.assert_array_equal(saved,b.history);self.assertEqual(b.steps.tolist(),[0,0])
        bad=sample(step=1);bad['time_s']+=.001
        with self.assertRaisesRegex(ValueError,'timestamp'):b.build([sample(step=1),bad])
        np.testing.assert_array_equal(saved,b.history)

    def test_landing_and_next_action_state_changes_are_observable(self):
        p=sample('landing');base=builder().build([p])['policy']
        mutations=[
            lambda s:s.__setitem__('flight_count',s['flight_count']+1),
            lambda s:s.__setitem__('contact_count',2),
            lambda s:s.__setitem__('landing_contact_gap_steps',1),
            lambda s:s.__setitem__('actual_descent_seen',False),
            lambda s:s['command_filter_rate'].__setitem__(0,s['command_filter_rate'][0]+.01),
            lambda s:s.__setitem__('stop_requested_time_s',-.01),
            lambda s:s.__setitem__('next_wave_order_index',2)]
        for mutate in mutations:
            changed=deepcopy(p);mutate(changed['reference']['state'])
            self.assertFalse(np.array_equal(base,builder().build([changed])['policy']))
        for key in ('swing','landing'):
            changed=deepcopy(p);t=changed['reference']['state'][key]
            t['coefficients'][3][0]+=.0001;t['endpoint_world_m'][0]+=.0001
            self.assertFalse(np.array_equal(base,builder().build([changed])['policy']))
        changed=deepcopy(p);s=changed['reference']['state'];s['flight_peak_z_m']+=.0001;s['measured_flight_lift_m']+=.0001
        self.assertFalse(np.array_equal(base,builder().build([changed])['policy']))

    def test_executable_residual_state_is_observable_and_consistency_enforced(self):
        p=sample();base=builder().build([p])['policy'];c=p['controller']
        c['residual_position_rad'][0][0]=.001;c['target_position_rad'][0][0]+=.001
        p['measurement']['joint_target_rad'][0][0]+=.001
        c['residual_velocity_rad_s'][0][0]=.01;c['target_velocity_rad_s'][0][0]+=.01
        self.assertFalse(np.array_equal(base,builder().build([p])['policy']))
        c['target_position_rad'][0][0]+=.001
        with self.assertRaisesRegex(ValueError,'state disagrees'):builder().build([p])

    def test_privileged_critic_signal_never_changes_actor(self):
        p=sample();a=builder().build([p]);p['critic_true_navigation_velocity'][0]+=.03
        b=builder().build([p]);np.testing.assert_array_equal(a['policy'],b['policy'])
        self.assertFalse(np.array_equal(a['critic'],b['critic']))

    def test_missing_malformed_nonfinite_and_invalid_state_fail_closed(self):
        mutations=[lambda p:p.pop('controller'),lambda p:p['reference']['state'].pop('landing'),
            lambda p:p['measurement'].__setitem__('joint_position_rad',[[0.]*17]),
            lambda p:p['measurement']['gyro_body_rad_s'][0].__setitem__(1,float('nan')),
            lambda p:p['reference']['state']['landing']['coefficients'][1].__setitem__(0,float('inf')),
            lambda p:p['reference']['state'].__setitem__('mode','new_unversioned_phase'),
            lambda p:p['reference'].__setitem__('valid',[False]),
            lambda p:p['measurement'].__setitem__('terminated',[True]),
            lambda p:p.__setitem__('wave_source_sha256','wrong'),
            lambda p:p['reference']['diagnostics']['configuration'].__setitem__('lift_m',.008)]
        for mutate in mutations:
            p=sample('landing');mutate(p)
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):builder().build([p])

    def test_unknown_stale_or_nonfinite_contact_never_becomes_support(self):
        for mode in ('age','valid','point','nan'):
            p=sample()
            if mode=='age':p['contact_age_s'][0]=.041
            if mode=='valid':p['contact_valid'][0]=False
            if mode=='point':p['measurement']['contact_point_valid'][0][0]=False
            if mode=='nan':p['measurement']['contact_point_world_m'][0][0][0]=float('nan')
            with self.subTest(mode=mode),self.assertRaises(ValueError):builder().build([p])

    def test_profiles_are_separate_and_critic_overflow_cannot_commit_history(self):
        a=builder().build([sample()])['schema'];b=builder(limits=Limits('diagnostic_003',.02,.25,1.5)).build([sample()])['schema']
        self.assertNotEqual(a['limits'],b['limits'])
        with self.assertRaises(ValueError):builder(limits=Limits('formal_004',.02,.25,1.5))
        b=builder(2);p=sample();p['critic_true_navigation_velocity'][0]=1e100
        with self.assertRaisesRegex(ValueError,'overflow'):b.build([sample(),p])
        self.assertFalse(b.valid.any());self.assertEqual(b.steps.tolist(),[-1,-1])

    def test_runtime_joint_permutation_uses_names_and_records_new_order(self):
        p=sample();names=p['joint_names_runtime'];permutation=list(reversed(range(18)))
        reverse=[names[k] for k in permutation]
        for key in ('joint_position_rad','joint_velocity_rad_s','joint_target_rad'):
            p['measurement'][key]=np.asarray(p['measurement'][key])[:,permutation].tolist()
        for key in p['controller']:p['controller'][key]=np.asarray(p['controller'][key])[:,permutation].tolist()
        for key in ('q_ref','v_ref','a_ref'):p['reference'][key]=np.asarray(p['reference'][key])[:,permutation].tolist()
        p['joint_names_runtime']=reverse;p['reference']['joint_names_runtime']=reverse
        out=builder(names=reverse).build([p]);self.assertEqual(tuple(out['schema']['joint_names_runtime']),tuple(reverse))
        with self.assertRaisesRegex(ValueError,'order mismatch'):builder().build([p])

    def test_schema_identity_binds_profile_order_nominal_and_metadata_is_not_mutable(self):
        b=builder();p=sample();a=b.build([p]);original=a['schema']['schema_identity_sha256']
        a['schema']['limits']['total_velocity_rad_s']=999
        self.assertEqual(b.limits.total_velocity_rad_s,2.)
        self.assertEqual(b.build([p])['schema']['schema_identity_sha256'],original)
        d=builder(limits=Limits('diagnostic_003',.02,.25,1.5)).build([sample()])
        self.assertNotEqual(original,d['schema']['schema_identity_sha256'])
        nominal=deepcopy(SOURCE_CONTRACT['nominal_joint_positions']);nominal[next(iter(nominal))]+=.001
        with self.assertRaisesRegex(ValueError,'Nominal stance'):
            ObservationBuilder(p['joint_names_runtime'],nominal,1)

    def test_landing_preload_contact_origin_and_stop_state_are_explicit(self):
        p=sample('landing');base=builder().build([p])['policy']
        for key in ('landing_preload_world_m','landing_contact_origin_world_m'):
            changed=deepcopy(p);changed['reference']['state'][key][0]+=.0001
            self.assertFalse(np.array_equal(base,builder().build([changed])['policy']))
        changed=deepcopy(p);changed['reference']['state']['reference_quiet_time_s']=-.01
        self.assertFalse(np.array_equal(base,builder().build([changed])['policy']))


if __name__=='__main__':unittest.main()
