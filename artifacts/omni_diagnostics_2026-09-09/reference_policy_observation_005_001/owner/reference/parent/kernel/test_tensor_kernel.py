from copy import deepcopy
import ast
import json
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from tensor_kernel import TensorGeometry, ReferenceKnots
from state_geometry import extract_geometry_state
from fixtures import oracle, rows, pack_state

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)


class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracles = {b:rows(b) for b in ('wave002_5mm','wave003_7mm')}
        cls.names = cls.oracles['wave003_7mm'][0].names

    def kernel(self, binding='wave003_7mm', names=None, profile='formal_004'):
        return TensorGeometry(names or self.names, binding=binding, profile=profile)

    def test_mixed_phase_batched_fk_ik_com_equal_scalar_for_both_bindings(self):
        for binding,(ref,examples) in self.oracles.items():
            g = self.kernel(binding)
            q = torch.tensor(np.stack([r['output']['q_ref'][0] for r in examples]),dtype=torch.float64)
            poses = torch.tensor(np.stack([r['after']['position_world_m'][0] for r in examples]),dtype=torch.float64)
            rotations = torch.tensor(np.stack([r['after']['rotation_world_from_body'][0] for r in examples]),dtype=torch.float64)
            fk = g.fk(g.leg(q)); ik = g.ik(fk['feet_body_m']); com = g.com(q,poses,rotations)
            for i in range(len(examples)):
                expected_f, expected_j, expected_T = ref.g.fk(g.leg(q)[i])
                torch.testing.assert_close(fk['feet_body_m'][i], expected_f,atol=1e-13,rtol=0)
                torch.testing.assert_close(fk['jacobian_m'][i], expected_j,atol=1e-13,rtol=0)
                torch.testing.assert_close(fk['transforms'][i],torch.stack(expected_T,1),atol=1e-13,rtol=0)
                old = ref.g.ik(expected_f)
                for key in ('q_checked','valid','reachable','in_limits','error_m','min_jacobian_singular_value_m','minimum_joint_margin_rad'):
                    torch.testing.assert_close(ik[key][i],old[key],atol=1e-12,rtol=0)
                expected_com = ref._com(g.leg(q)[i].numpy(),poses[i].numpy(),rotations[i].numpy())
                np.testing.assert_allclose(com['world_m'][i].numpy(),expected_com,atol=1e-13,rtol=0)
            self.assertTrue(com['valid'].all())

    def test_runtime_order_bijection_canonical_reverse_and_eighteen_cyclic_orders(self):
        original = self.kernel(); q = original.q0[None].repeat(2,1,1)
        q[1,:,0] += .02
        expected = original.fk(q)
        for names in [self.names,tuple(reversed(self.names))]+[self.names[i:]+self.names[:i] for i in range(18)]:
            g = self.kernel(names=names)
            runtime = torch.tensor([[float(row.reshape(-1)[original.leg_names.index(n)]) for n in names] for row in q],dtype=torch.float64)
            torch.testing.assert_close(g.leg(runtime),q,atol=0,rtol=0)
            torch.testing.assert_close(g.runtime(q),runtime,atol=0,rtol=0)
            torch.testing.assert_close(g.fk(g.leg(runtime))['feet_body_m'],expected['feet_body_m'],atol=0,rtol=0)
        with self.assertRaises(ValueError): self.kernel(names=self.names[:-1]+(self.names[0],))

    def test_unreachable_limit_singular_and_nonfinite_masks_are_per_limb(self):
        g = self.kernel(); targets = g.fk(g.q0[None].repeat(5,1,1))['feet_body_m']
        targets[1,0] += torch.tensor([10.,0.,0.])
        q = g.q0.clone(); q[0,0] = 1.1
        targets[2] = g.fk(q[None])['feet_body_m'][0]
        targets[3,2,0] = torch.nan
        targets[4,3] = torch.inf
        result = g.ik(targets)
        self.assertTrue(result['valid'][0].all())
        self.assertFalse(result['valid'][1,0]); self.assertFalse(result['reachable'][1,0])
        self.assertFalse(result['valid'][2,0]); self.assertFalse(result['in_limits'][2,0])
        self.assertFalse(result['valid'][3,2]); self.assertFalse(result['finite'][3,2])
        self.assertFalse(result['valid'][4,3]); self.assertTrue(result['valid'][4,:3].all())
        self.assertTrue(torch.isfinite(result['q_checked']).all())
        # Full extension is geometrically singular and also outside admitted
        # soft limits; it cannot pass through either diagnostic clipping path.
        q = g.q0.clone(); q[:,2] = torch.atan2(g.tip[:,2],(g.tip*g.radial).sum(-1))
        self.assertFalse(g.ik(g.fk(q[None])['feet_body_m'])['valid'].any())

    def test_polynomials_match_scalar_original_and_landing_with_mixed_active_rows(self):
        for binding,(_,examples) in self.oracles.items():
            g = self.kernel(binding); state = pack_state(examples,g)
            for i, name in enumerate(('swing','landing')):
                up = torch.tensor([0.,0.,1.],dtype=g.dtype).repeat(len(examples),1)
                out = g.polynomial(state['trajectory_coefficients'][:,i],state['trajectory_start_s'][:,i],
                    state['trajectory_duration_s'][:,i],state['trajectory_lift_m'][:,i],state['time_s'],
                    state['trajectory_active'][:,i],world_up=up)
                self.assertTrue(out['valid'].all())
                for j,row in enumerate(examples):
                    trajectory = row['output']['state'][name]
                    if trajectory is None:
                        self.assertEqual(float(out['position_world_m'][j].abs().max()),0)
                        continue
                    from wave_math import Swing
                    s=Swing(trajectory['start_s'],trajectory['duration_s'],trajectory['coefficients'][0],trajectory['endpoint_world_m'],trajectory['lift_m'])
                    s.coeff=trajectory['coefficients'].copy()
                    expected=s.sample(row['output']['target_time_s'])
                    for key,value in zip(('position_world_m','velocity_world_mps','acceleration_world_mps2'),expected):
                        np.testing.assert_allclose(out[key][j].numpy(),value,atol=1e-12,rtol=0)

    def test_reference_knots_match_scalar_mixed_rows_and_preserve_quiet_preload(self):
        for binding,(_,examples) in self.oracles.items():
            g=self.kernel(binding); n=len(examples); state=ReferenceKnots(g,n)
            q=g.runtime(torch.tensor(np.stack([r['q_previous'] for r in examples])))
            previous_v=g.runtime(torch.tensor(np.stack([r['v_previous'] for r in examples])))
            times=torch.tensor([r['before']['time_s'][0] for r in examples])
            times=times.to(g.dtype)
            limits=torch.tensor(np.stack([r['before']['soft_joint_pos_limits_rad'][0] for r in examples]))
            result=state.reset(torch.ones(n,dtype=torch.bool),torch.zeros(n,dtype=torch.int64),times,q,limits,torch.zeros_like(q))
            self.assertTrue(result['reset_accepted'].all())
            # Fixture restores a captured executable V as well as P to compare
            # an intermediate step, not a runtime reset from nonzero velocity.
            state.v.copy_(previous_v)
            targets=torch.tensor(np.stack([r['output']['diagnostics']['reference_point_world_m'] for r in examples]))
            poses=torch.tensor(np.stack([r['output']['state']['desired_position_world_m'] for r in examples]))
            rots=torch.tensor(np.stack([r['output']['state']['desired_rotation_world_from_body'] for r in examples]))
            quiet=torch.tensor([r['mode']=='reference_quiet_hold' for r in examples])
            out=state.advance(targets,poses,rots,times,quiet,torch.ones(n,dtype=torch.bool))
            self.assertTrue(out['valid'].all())
            for key, scalar in [('q_out','q_ref'),('v_out','v_ref'),('a_out','a_ref')]:
                np.testing.assert_allclose(out[key].numpy(),np.stack([r['output'][scalar][0] for r in examples]),atol=2e-10,rtol=0)
            torch.testing.assert_close(out['q_out'][0],q[0],atol=0,rtol=0)

    def test_failure_latches_no_mutation_stale_time_and_partial_fresh_reset(self):
        g=self.kernel(); n=3; s=ReferenceKnots(g,n)
        q=g.runtime(g.q0[None].repeat(n,1,1)); limits=torch.stack((g.runtime(g.lower[None].repeat(n,1,1)),g.runtime(g.upper[None].repeat(n,1,1))),-1)
        args=(torch.ones(n,dtype=torch.bool),torch.zeros(n,dtype=torch.int64),torch.zeros(n,dtype=g.dtype),q,limits,torch.zeros_like(q))
        with torch.inference_mode(): self.assertTrue(s.reset(*args)['reset_accepted'].all())
        world=g.fk(g.q0[None].repeat(n,1,1))['feet_body_m']; pose=torch.zeros(n,3,dtype=g.dtype); R=torch.eye(3,dtype=g.dtype).repeat(n,1,1)
        times=torch.tensor([0.,.1,0.],dtype=g.dtype); upstream=torch.tensor([True,True,False])
        out=s.advance(world,pose,R,times,torch.ones(n,dtype=torch.bool),upstream)
        self.assertEqual(out['valid'].tolist(),[True,False,False]); self.assertTrue(torch.isnan(out['q_out'][1:]).all())
        torch.testing.assert_close(s.q,q,atol=0,rtol=0)
        selected=torch.tensor([False,True,False]); episode=torch.tensor([0,1,0])
        accepted=s.reset(selected,episode,torch.zeros(n,dtype=g.dtype),q,limits,torch.zeros_like(q))
        self.assertEqual(accepted['reset_accepted'].tolist(),[False,True,False])
        self.assertEqual(s.failed.tolist(),[False,False,True]); self.assertAlmostEqual(float(s.time[0]),.02)
        self.assertTrue(s.reset(selected,episode,torch.zeros(n,dtype=g.dtype),q,limits,torch.zeros_like(q))['reset_rejected'][1])
        self.assertTrue(s.failed[1]); self.assertFalse(s.ready[1])

    def test_formal_and_diagnostic_rate_profiles_reject_independently(self):
        # Same feasible P/V/A transition:1.4rad/s is below1.75 but above1.25.
        for profile, expected in [('formal_004',True),('diagnostic_003',False)]:
            g=self.kernel(profile=profile); s=ReferenceKnots(g,1)
            q=g.q0[None].clone(); qnext=q.clone(); qnext[:,:,0]+=.028
            limits=torch.stack((g.runtime(g.lower[None]),g.runtime(g.upper[None])),-1)
            s.reset(torch.tensor([True]),torch.tensor([0]),torch.zeros(1,dtype=g.dtype),g.runtime(q),limits,torch.zeros((1,18),dtype=g.dtype))
            s.v[:]=g.runtime((qnext-q)/.02)
            out=s.advance(g.fk(qnext)['feet_body_m'],torch.zeros(1,3,dtype=g.dtype),torch.eye(3,dtype=g.dtype)[None],
                          torch.zeros(1,dtype=g.dtype),torch.tensor([False]),torch.tensor([True]))
            self.assertEqual(bool(out['valid'][0]),expected)

    def test_state_extraction_frame_covariance_and_required_state_observability(self):
        g=self.kernel(); data=pack_state(self.oracles[g.binding][1],g)
        original=extract_geometry_state(g,data); self.assertTrue(original['valid'].all()); self.assertEqual(original['width'],233)
        rot=torch.tensor(Rotation.from_euler('xyz',[.2,-.3,.4]).as_matrix()); shift=torch.tensor([.4,-.7,1.2])
        transformed={k:v.clone() for k,v in data.items()}
        for k in ('position_world_m','desired_position_world_m','reference_anchors_world_m','measured_anchors_world_m','trajectory_endpoints_world_m'):
            transformed[k]=data[k]@rot.T+shift
        for k in ('rotation_world_from_body','desired_rotation_world_from_body','initial_desired_rotation_world_from_body'):
            transformed[k]=rot@data[k]
        transformed['reference_minus_measured_preload_world_m']=data['reference_minus_measured_preload_world_m']@rot.T
        transformed['trajectory_coefficients']=data['trajectory_coefficients']@rot.T
        transformed['trajectory_coefficients'][:,:,0]+=shift
        active=data['trajectory_active']
        transformed['trajectory_coefficients']=torch.where(active[:,:,None,None],transformed['trajectory_coefficients'],0.)
        transformed['trajectory_endpoints_world_m']=torch.where(active[:,:,None],transformed['trajectory_endpoints_world_m'],0.)
        changed=extract_geometry_state(g,transformed)
        self.assertTrue(changed['valid'].all()); torch.testing.assert_close(original['values'],changed['values'],atol=2e-12,rtol=0)
        for key in ('initial_joint_preload_leg_major_rad','neutral_reference_toes_body_m'):
            different={k:v.clone() for k,v in data.items()}; different[key][0,0,0]+=.001
            self.assertFalse(torch.equal(extract_geometry_state(g,different)['values'],original['values']))

    def test_state_missing_nonfinite_invalid_pose_source_lift_and_preload_fail(self):
        g=self.kernel(); data=pack_state(self.oracles[g.binding][1],g)
        with self.assertRaises(ValueError): extract_geometry_state(g,{k:v for k,v in data.items() if k!='trajectory_coefficients'})
        for key in ('target_position_rad','time_s','initial_joint_preload_leg_major_rad'):
            bad={k:v.clone() for k,v in data.items()}; bad[key][0]=torch.nan
            result=extract_geometry_state(g,bad); self.assertFalse(result['valid'][0]); self.assertTrue(result['valid'][1:].all())
        bad={k:v.clone() for k,v in data.items()}; bad['rotation_world_from_body'][0]*=2
        self.assertFalse(extract_geometry_state(g,bad)['valid'][0])
        bad={k:v.clone() for k,v in data.items()}; bad['reference_minus_measured_preload_world_m'][0,0,0]+=.01
        self.assertFalse(extract_geometry_state(g,bad)['valid'][0])
        wrong=self.kernel('wave002_5mm'); result=extract_geometry_state(wrong,data)
        self.assertFalse(result['valid'][data['trajectory_active'][:,0]].any())

    def test_common_geometric_fields_equal_frozen740_encoder(self):
        spec=importlib.util.spec_from_file_location('scalar_observation_oracle',HERE/'oracle/observation002/observation.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        fixtures=json.loads((HERE/'oracle/observation002/fixtures.json').read_text())
        for name in ('standing','landing'):
            packet=fixtures[name]
            g=self.kernel('wave002_5mm',names=tuple(packet['joint_names_runtime']))
            # Offline conversion only; at runtime a vectorized state machine
            # must produce this structure of arrays directly on device.
            packed=pack_state([dict(after=packet['measurement'],output=packet['reference'])],g)
            for key in ('target_position_rad','target_velocity_rad_s','residual_position_rad',
                        'residual_velocity_rad_s','reference_position_rad','reference_velocity_rad_s'):
                packed[key]=torch.tensor(np.asarray(packet['controller'][key]).reshape(1,18),dtype=g.dtype)
            b=module.ObservationBuilder(packet['joint_names_runtime'],module.SOURCE_CONTRACT['nominal_joint_positions'],1)
            b.reset([0],[packet['episode_id']],[0.])
            # Compare unrounded float64 encoder fields before its finalfloat32 cast.
            _,current,fields,_=b._encode(packet)
            cursor=0;expected={}
            for field in fields:
                expected[field['name']]=current[cursor:cursor+field['width']];cursor+=field['width']
            out=extract_geometry_state(g,packed);self.assertTrue(out['valid'][0])
            for field in out['fields']:
                if field['name']=='initial_joint_preload_leg_major_rad':continue
                np.testing.assert_allclose(out['values'][0,field['start']:field['stop']].numpy(),
                                           expected[field['name']],atol=2e-12,rtol=0,err_msg=field['name'])

    def test_reset_rejects_moving_target_and_wrong_contract_before_mutation(self):
        g=self.kernel();s=ReferenceKnots(g,2);q=g.runtime(g.q0[None].repeat(2,1,1))
        limits=torch.stack((g.runtime(g.lower[None].repeat(2,1,1)),g.runtime(g.upper[None].repeat(2,1,1))),-1)
        moving=torch.zeros_like(q);moving[1,0]=.001
        result=s.reset(torch.ones(2,dtype=torch.bool),torch.zeros(2,dtype=torch.int64),torch.zeros(2,dtype=g.dtype),q,limits,moving)
        self.assertEqual(result['reset_accepted'].tolist(),[True,False]);self.assertTrue(s.failed[1])
        before=s.q.clone()
        with self.assertRaises(ValueError):
            s.reset(torch.ones(2,dtype=torch.bool),torch.ones(2,dtype=torch.int64),torch.zeros(2,dtype=g.dtype),q.float(),limits,moving)
        torch.testing.assert_close(before,s.q,atol=0,rtol=0)
        with self.assertRaises(ValueError): self.kernel(binding='wave004_7mm')
        with self.assertRaises(ValueError): self.kernel(profile='unlabelled')

    def test_dynamic_path_has_no_cpu_transfer_scalar_sync_or_replica_loop(self):
        # Structural complement to numeric parity; actual CUDA performance is
        # still unmeasured. Fixed3-link /2-trajectory loops are permitted.
        for path in [HERE/'tensor_kernel.py',HERE/'state_geometry.py']:
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    self.assertNotIn(node.func.attr,('cpu','numpy','item','tolist'))
                if isinstance(node,ast.For) and isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Name) and node.iter.func.id=='range':
                    self.assertTrue(all(isinstance(x,ast.Constant) for x in node.iter.args))


if __name__=='__main__': unittest.main()
