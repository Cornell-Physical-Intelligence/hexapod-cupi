"""Recorded physics inputs exercise the exact time/controller/history seam.

This CPU test does not simulate physics. Contact-clock validity is a declared
synthetic fixture here; the separate clock tests exercise the installed source
recurrence. The on-device smoke must obtain actual clock readings.
"""
from pathlib import Path
import hashlib
import json
import sys
import unittest
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
BUNDLE=HERE.parent/'reference_policy_observation_005_001'
sys.path.insert(0,str(BUNDLE))
from observation import ObservationBuilder,CONTROLLER_KEYS
from batch_wave import BatchWave005
from reference_residual_oracle import ReferenceResidualTarget,ResidualConfig


class RecordedSeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        provenance=json.loads((HERE/'inputs/PROVENANCE.json').read_text())
        source=HERE/'inputs'/provenance['derived_file']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=provenance['derived_sha256']:
            raise ValueError('Recorded standing fixture changed')
        with np.load(source) as data:cls.data={key:data[key] for key in data.files}
        torch.set_num_threads(1)

    def measurement(self,index,n):
        m={key:torch.tensor(value[index,:n],dtype=torch.bool if value.dtype==np.bool_ else torch.float64)
           for key,value in self.data.items() if value.ndim>1 and value.shape[:2]==(65,32)}
        m['contact_point_world_m_raw']=m['contact_point_world_m'].clone()
        m['contact_point_world_m']=torch.nan_to_num(m['contact_point_world_m'])
        m['position_is_float32']=torch.ones(n,dtype=torch.bool)
        m['contact_valid']=torch.ones(n,6,dtype=torch.bool)
        m['contact_age_s']=torch.zeros(n,6,dtype=torch.float64)
        m['measurement_valid']=torch.ones(n,dtype=torch.bool)
        m['root_link_velocity_body_mps']=(m['velocity_world_mps'][:,None,:]@m['rotation_world_from_body']).squeeze(1)
        return m

    def run_sequence(self,n):
        names=tuple(self.data['joint_names'])
        m=self.measurement(0,n);wave=BatchWave005(names,n)
        selected=torch.ones(n,dtype=torch.bool);episodes=torch.zeros(n,dtype=torch.int64)
        ref=wave.reset(m,selected,episodes)
        self.assertTrue(ref['valid'].all(),ref['failure_code'])
        limits=m['soft_joint_pos_limits_rad'][0]
        controller=ReferenceResidualTarget(names,dict(zip(names,limits[:,0].tolist())),dict(zip(names,limits[:,1].tolist())),n,ResidualConfig('formal_004',.02,.25,2.,8.))
        controller.reset(m['joint_target_rad'],m['joint_target_rad'])
        c={key:m[key].clone() for key in CONTROLLER_KEYS}
        encoder=ObservationBuilder(names,n);encoder.reset(selected,episodes,m['time_s'])
        command=torch.zeros(n,3,dtype=torch.float64);zero=torch.zeros(n,18,dtype=torch.float64)
        def packet(i):
            return dict(measurement_source='synthetic_fixture',joint_names_runtime=names,
                measurement=m,reference=ref,controller=c,requested_twist=command,
                world_up=torch.tensor([[0.,0.,1.]]*n,dtype=torch.float64),episode_ids=episodes,
                step_indices=torch.full((n,),i,dtype=torch.int64),critic_simulator_reported_twist=m['velocity_body_mps'])
        encoded=encoder.build(packet(0));self.assertTrue(encoded['valid'].all())
        self.assertFalse(encoded['interval_rate_valid'].any())
        for index in range(1,65):
            # This is the runtime order: old measurement -> next reference ->
            # executable controller target -> next physical sample -> observation.
            ref=wave.step(m,command);self.assertTrue(ref['valid'].all(),ref['failure_code'])
            c=controller.step(ref['q_ref'],zero,reference_valid=ref['valid'],
                analytic_reference_velocity=ref['v_ref'],analytic_reference_acceleration=ref['a_ref'])
            m=self.measurement(index,n)
            np.testing.assert_allclose(c['target_position_rad'].numpy(),m['joint_target_rad'].numpy(),atol=3e-7,rtol=0)
            encoded=encoder.build(packet(index));self.assertTrue(encoded['valid'].all(),index)
            self.assertEqual(encoded['policy'].shape,(n,846));self.assertEqual(encoded['critic'].shape,(n,849))
            torch.testing.assert_close(encoded['raw_sdk_joint_velocity_rad_s'],m['joint_velocity_rad_s'],atol=0,rtol=0)
            self.assertTrue(encoded['interval_rate_valid'].all())
            expected=(m['joint_position_rad']-self.measurement(index-1,n)['joint_position_rad'])/.02
            torch.testing.assert_close(encoded['interval_joint_rate_rad_s'],expected,atol=1e-14,rtol=0)
        return encoder,packet(64)

    def test_one_and_32_rows_have_coherent_executed_reference_time_and_history(self):
        for n in (1,32):
            with self.subTest(replicas=n):self.run_sequence(n)

    def test_post_reset_or_old_reference_cannot_be_encoded_as_current(self):
        encoder,p=self.run_sequence(1)
        p['reference']['state']['time']-=.02
        self.assertFalse(encoder.build(p)['valid'].any())

if __name__=='__main__':unittest.main()
