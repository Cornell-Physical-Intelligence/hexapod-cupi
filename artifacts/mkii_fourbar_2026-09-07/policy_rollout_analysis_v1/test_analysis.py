import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from analyze_rollout import summarize, rotation, main

class MotionAnalysis(unittest.TestCase):
    def fixture(self):
        n=4
        return {'pre_root_pos_w_m':np.tile([0,0,.14],(n,1)),
                'post_root_pos_w_m':np.tile([0,0,.14],(n,1)),
                'pre_root_quat_w_xyzw':np.tile([0,0,0,1],(n,1)),
                'post_root_quat_w_xyzw':np.tile([0,0,0,1],(n,1)),
                'pre_command_navigation':np.zeros((n,3)), 'done':np.zeros(n,dtype=bool)}
    def test_forward_left_and_rotated_body_frame(self):
        s=self.fixture();s['post_root_pos_w_m']=s['post_root_pos_w_m'].astype(float)
        s['post_root_pos_w_m'][:,:2]=[.002,-.004];s['pre_command_navigation'][:]=[.2,.1,0]
        np.testing.assert_allclose(summarize(s,.02)['tracking_rmse'],0,atol=1e-12)
        q=[0,0,np.sqrt(.5),np.sqrt(.5)]
        s['pre_root_quat_w_xyzw']=np.tile(q,(4,1));s['post_root_quat_w_xyzw']=np.tile(q,(4,1))
        s['post_root_pos_w_m'][:,:2]=[.004,.002]
        np.testing.assert_allclose(summarize(s,.02)['tracking_rmse'],0,atol=1e-12)
    def test_positive_yaw_and_quaternion_sign_invariance(self):
        s=self.fixture();theta=.006
        q=[0,0,np.sin(theta/2),np.cos(theta/2)]
        s['post_root_quat_w_xyzw']=np.tile(q,(4,1));s['pre_command_navigation'][:,2]=.3
        np.testing.assert_allclose(summarize(s,.02)['tracking_rmse'],0,atol=1e-12)
        s['post_root_quat_w_xyzw'] *= -1
        np.testing.assert_allclose(summarize(s,.02)['tracking_rmse'],0,atol=1e-12)
    def test_reset_teleport_excluded_and_stationary_policy_revealed(self):
        s=self.fixture();s['post_root_pos_w_m']=s['post_root_pos_w_m'].astype(float)
        s['post_root_pos_w_m'][1]=[100,100,0];s['done'][1]=True
        s['pre_command_navigation'][:,0]=.1
        r=summarize(s,.02)
        self.assertEqual(r['transitions_used'],3);self.assertEqual(r['actual_planar_path_length_m'],0)
        self.assertEqual(r['near_stationary_fraction_during_moving_commands'],1)
        self.assertEqual(r['excluded_reset_transition_indices'],[1])
        self.assertIsNone(r['acceptance_pass'])
    def test_tilted_body_translation_and_local_yaw_are_not_world_heading(self):
        s=self.fixture();roll=.4;theta=.006
        sr,cr=np.sin(roll/2),np.cos(roll/2)
        pre=np.array([sr,0,0,cr])
        post=np.array([sr*np.cos(theta/2),-sr*np.sin(theta/2),cr*np.sin(theta/2),cr*np.cos(theta/2)])
        s['pre_root_quat_w_xyzw']=np.tile(pre,(4,1));s['post_root_quat_w_xyzw']=np.tile(post,(4,1))
        displacement=rotation(pre[None])[0] @ np.array([.002,-.004,.001])
        s['post_root_pos_w_m']=s['pre_root_pos_w_m']+displacement
        s['pre_command_navigation'][:]=[.2,.1,.3]
        r=summarize(s,.02)
        np.testing.assert_allclose(r['tracking_rmse'],0,atol=1e-12)
        self.assertIn('not the exact instantaneous',r['training_reward_comparison'])
        self.assertEqual(r['sdk_alias_provenance']['root_lin_vel_b_alias_line'],1186)

    def test_cli_verifies_hashes_and_preserves_capture_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            np.savez_compressed(base/'states.npz',**self.fixture())
            metadata={'root_quaternion_order':'XYZW','fps':50.,'source_sha256':'a'*64,
                      'input_sha256':{'checkpoint.pt':'b'*64}}
            (base/'metadata.json').write_text(json.dumps(metadata))
            hashes={name:{'sha256':hashlib.sha256((base/name).read_bytes()).hexdigest(),
                          'bytes':(base/name).stat().st_size} for name in ('states.npz','metadata.json')}
            (base/'report.json').write_text(json.dumps({'pass':True,'artifacts':hashes}))
            output=base/'analysis.json'
            with patch('sys.argv',['analyze_rollout',str(base),'--output',str(output)]):main()
            result=json.loads(output.read_text())
            self.assertEqual(result['capture_report_sha256'],hashlib.sha256((base/'report.json').read_bytes()).hexdigest())
            self.assertEqual(result['capture_input_sha256'],metadata['input_sha256'])
            self.assertIsNone(result['acceptance_pass'])
            with (base/'states.npz').open('ab') as stream:stream.write(b'changed')
            with patch('sys.argv',['analyze_rollout',str(base),'--output',str(output)]):
                with self.assertRaisesRegex(ValueError,'Capture artifact bytes changed'):main()

    def test_invalid_archive_data_fails(self):
        s=self.fixture();s['pre_root_quat_w_xyzw'][0]=[0,0,0,0]
        with self.assertRaises(ValueError):summarize(s,.02)
        s=self.fixture();s['pre_command_navigation'][0,0]=np.nan
        with self.assertRaises(ValueError):summarize(s,.02)
        with self.assertRaises(ValueError):summarize(self.fixture(),0)

if __name__=='__main__':unittest.main()
