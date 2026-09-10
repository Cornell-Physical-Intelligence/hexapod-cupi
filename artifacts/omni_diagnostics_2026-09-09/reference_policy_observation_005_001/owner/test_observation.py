from copy import deepcopy
import unittest
from pathlib import Path
import ast
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from fixtures import Rig,fresh_encode
from observation import ObservationBuilder

torch.set_num_threads(1)
class ObservationTests(unittest.TestCase):
    def test_computed_width_reset_sample_and_idempotent_history(self):
        r=Rig(2);p=r.packet();o=r.builder.build(p)
        self.assertTrue(o['valid'].all());self.assertEqual(o['policy'].shape[1],o['schema']['actor_width'])
        self.assertEqual(o['critic'].shape[1],o['schema']['actor_width']+3)
        self.assertEqual(r.builder.history_valid.tolist(),[[False]*4+[True]]*2)
        again=r.builder.build(p);torch.testing.assert_close(o['policy'],again['policy'],atol=0,rtol=0)
        self.assertFalse(o['interval_rate_valid'].any());self.assertFalse(o['policy_training_allowed'])
        self.assertEqual(o['policy'].shape[1],846);self.assertEqual(o['critic'].shape[1],849)
        self.assertNotIn(o['policy'].shape[1],(233,740,743,525,528))
    def test_full_asynchronous_stance_swing_landing_stop_are_finite(self):
        r=Rig(3);self.assertTrue(r.builder.build(r.packet())['valid'].all());seen={0}
        for k in range(450):
            commands=[[.005,0,0] if k<180 else [0,0,0],[0,.005,0] if 40<=k<240 else [0,0,0],[0,0,0]]
            p=r.advance(commands);o=r.builder.build(p)
            self.assertTrue(o['valid'].all(),(k,p['reference']['state']['mode'],o['valid']))
            self.assertTrue(torch.isfinite(o['policy']).all());seen.update(p['reference']['state']['mode'].tolist())
        self.assertTrue({0,1,5,6}<=seen);self.assertTrue(r.builder.history_valid.all())
    def test_partial_reset_failure_latch_and_old_history_cannot_leak(self):
        r=Rig(2);r.builder.build(r.packet())
        for _ in range(5):r.builder.build(r.advance([[0,0,0]]*2))
        p=r.packet();p['measurement']['joint_velocity_rad_s'][0,0]=.3
        o=r.builder.build(p);self.assertEqual(o['valid'].tolist(),[False,True]);self.assertTrue(torch.isnan(o['policy'][0]).all())
        clean=r.packet();self.assertFalse(r.builder.build(clean)['valid'][0])
        sel=torch.tensor([True,False]);episodes=torch.tensor([1,0]);r.builder.reset(sel,episodes,clean['measurement']['time_s'])
        oldrow=r.builder.history[1].clone();clean['episode_ids']=episodes;clean['reference']['state']['episode']=episodes;clean['step_indices'][0]=0
        o=r.builder.build(clean);self.assertTrue(o['valid'].all());self.assertEqual(r.builder.history_valid[0].tolist(),[False]*4+[True])
        torch.testing.assert_close(r.builder.history[1],oldrow,atol=0,rtol=0)
        r.builder.reset(sel,episodes,clean['measurement']['time_s']);self.assertFalse(r.builder.build(clean)['valid'][0])
    def test_raw_reported_bias_is_retained_separate_from_position_interval_rate(self):
        r=Rig();p=r.packet();p['measurement']['joint_velocity_rad_s'].fill_(.02565);r.builder.build(p)
        p=r.advance([[0,0,0]]);p['measurement']['joint_velocity_rad_s'].fill_(.02565);o=r.builder.build(p)
        self.assertTrue(o['valid'].all());self.assertTrue(o['interval_rate_valid'].all())
        self.assertLess(float(o['interval_joint_rate_rad_s'].abs().max()),1e-10)
        torch.testing.assert_close(o['raw_sdk_joint_velocity_rad_s'],torch.full((1,18),.02565,dtype=torch.float64))
        self.assertFalse(o['reported_velocity_physical_consistency_verified'])
    def test_malformed_nonfinite_stale_and_changed_source_rejected(self):
        r=Rig(2)
        for mutate in [lambda p:p['measurement']['contact_age_s'].__setitem__((0,0),.041),
                       lambda p:p['reference']['state']['sw_hcoeff'].__setitem__((0,0,0),float('nan')),
                       lambda p:p['controller']['target_velocity_rad_s'].__setitem__((0,0),float('inf'))]:
            p=r.packet();mutate(p);o=fresh_encode(p);self.assertEqual(o['valid'].tolist(),[False,True])
        p=r.packet();del p['measurement']['contact_valid']
        with self.assertRaises(KeyError):fresh_encode(p)
        p=r.packet();p['reference']['source_identity']['scalar_controller_sha256']='wrong'
        with self.assertRaises(ValueError):fresh_encode(p)
        p=r.packet();p['measurement']['joint_position_rad']=p['measurement']['joint_position_rad'].float()
        with self.assertRaises(ValueError):fresh_encode(p)
    def test_full_frame_rotation_translation_covariance_during_landing(self):
        r=Rig()
        for _ in range(140):
            p=r.advance([[.005,0,0]])
            if p['reference']['state']['land_active'].all():break
        else:self.fail('No provisional landing fixture')
        a=fresh_encode(p);self.assertTrue(a['valid'].all())
        q=deepcopy(p);S=torch.tensor(Rotation.from_euler('xyz',[.31,-.22,.72]).as_matrix(),dtype=torch.float64);t=torch.tensor([1.,-2.,.6],dtype=torch.float64)
        point=lambda x:x@S.T+t
        m=q['measurement'];s=q['reference']['state']
        for key in ('position_world_m','reference_point_world_m'):m[key]=point(m[key])
        m['contact_point_world_m']=torch.where(m['contact_point_valid'][...,None],point(m['contact_point_world_m']),0.)
        m['reference_point_velocity_world_mps']=m['reference_point_velocity_world_mps']@S.T
        m['rotation_world_from_body']=S@m['rotation_world_from_body'];q['world_up']=q['world_up']@S.T
        for key in ('position','anchors','measured_anchors','landing_origin','sw_end','land_end'):s[key]=point(s[key])
        for key in ('preload_world','landing_preload'):s[key]=s[key]@S.T
        s['R0']=S@s['R0']
        for key in ('sw_coeff','sw_hcoeff','land_coeff'):
            s[key]=s[key]@S.T;s[key][:,0]+=t
        for key in ('flight_baseline_z','flight_peak_z'):s[key]+=(t*q['world_up']).sum(-1)
        b=fresh_encode(q);self.assertTrue(b['valid'].all())
        torch.testing.assert_close(a['policy'],b['policy'],atol=4e-7,rtol=0)
    def test_next_transition_state_is_observable_including_horizontal_polynomial(self):
        r=Rig()
        for _ in range(20):p=r.advance([[.005,0,0]])
        base=fresh_encode(p)
        for key in ('contact_count','landing_gap','liftoffs','raw_force_free_samples'):
            q=deepcopy(p);q['reference']['state'][key]+=1;other=fresh_encode(q)
            self.assertTrue(other['valid'].all());self.assertFalse(torch.equal(base['policy'],other['policy']),key)
        q=deepcopy(p);q['reference']['state']['sw_hcoeff'][:,1,0]+=.0001;q['reference']['state']['sw_hcoeff'][:,2,0]-=.0001
        other=fresh_encode(q);self.assertTrue(other['valid'].all());self.assertFalse(torch.equal(base['policy'],other['policy']))
    def test_reversed_named_joint_order_and_inference_created_state(self):
        initial=Rig();names=tuple(reversed(initial.names))
        with torch.inference_mode():r=Rig(2,names);o=r.builder.build(r.packet())
        self.assertTrue(o['valid'].all());o=r.builder.build(r.advance([[0,0,0]]*2));self.assertTrue(o['valid'].all())
    def test_dynamic_paths_have_no_host_tensor_transfers(self):
        for name in ('observation.py','device_pack.py'):
            tree=ast.parse((Path(__file__).parent/name).read_text())
            prohibited=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and n.attr in ('cpu','numpy','item','tolist')]
            self.assertEqual(prohibited,[],name)

    def test_acceleration_composition_and_checkpoint_incompatibility(self):
        from observation import reject_checkpoint
        r=Rig(2);p=r.packet();p['controller']['target_acceleration_rad_s2'][0,0]=.1
        self.assertEqual(fresh_encode(p)['valid'].tolist(),[False,True])
        for width in (233,495,498,525,528,740,743,849):
            with self.assertRaises(ValueError):reject_checkpoint({'actor_width':width})

    def test_explicit_noise_changes_history_once_and_bad_epoch_has_no_valid_interval(self):
        r=Rig(2);p=r.packet();p['measurement']['joint_position_rad'][0,0]+=.001
        first=r.builder.build(p);again=r.builder.build(p)
        torch.testing.assert_close(first['policy'],again['policy'],atol=0,rtol=0)
        q=r.advance([[0,0,0]]*2);q['episode_ids'][0]+=1
        out=r.builder.build(q);self.assertFalse(out['valid'][0]);self.assertFalse(out['interval_rate_valid'][0])
        self.assertTrue(torch.isnan(out['interval_joint_rate_rad_s'][0]).all())
        self.assertTrue(out['valid'][1]);self.assertEqual(out['interval_dt_s'],.02)

    def test_unloading_qualified_history_and_optional_return_are_observable(self):
        r=Rig();p=r.advance([[.005,0,0]]);base=fresh_encode(p)
        self.assertEqual(int(p['reference']['state']['mode'][0]),8);self.assertTrue(base['valid'].all())
        q=deepcopy(p);s=q['reference']['state']
        s['raw_force_free_samples']+=2;s['raw_force_free_runs']+=1;s['unqualified_contact_returns']+=1
        s['last_unqualified_run_samples']+=2;s['unqualified_return_valid'].fill_(True)
        s['unqualified_return_time']=s['time'].clone();s['unqualified_lift'].fill_(.000008)
        other=fresh_encode(q);self.assertTrue(other['valid'].all());self.assertFalse(torch.equal(base['policy'],other['policy']))
        q['reference']['state']['unqualified_return_valid'].fill_(False)
        self.assertFalse(fresh_encode(q)['valid'].all())

    def test_already_imported_wrong_reference_module_fails_without_reload(self):
        import sys
        from unittest.mock import patch
        r=Rig()
        with patch.object(sys.modules['batch_wave'],'__file__','/wrong/source/batch_wave.py'):
            with self.assertRaisesRegex(ValueError,'Previously imported'):ObservationBuilder(r.names,1)

if __name__=='__main__':unittest.main()
