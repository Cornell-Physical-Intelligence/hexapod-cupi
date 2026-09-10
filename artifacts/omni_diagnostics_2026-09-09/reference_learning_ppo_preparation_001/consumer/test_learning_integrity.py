import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_recovery import Reader,Recorder
from test_inference_reset import ExactResetBufferEnv
import moving_session
from recovery import defer_training_resets
from masked_runner import MaskedRunner
from moving_runner import initialize_scratch,runner_config
from test_masked_rsl import Episodes

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_actual_session_unoptimized_event_recovery_and_compact_export(self):
        env=ExactResetBufferEnv();stance={'joint_positions_rad':dict(zip(env.rig.names,env._robot.data.default_joint_pos[0].tolist()))}
        with tempfile.TemporaryDirectory() as d,patch.object(moving_session,'DeviceTelemetry',Reader),patch.object(moving_session,'PhysicsSubstepRecorder',Recorder):
            s=moving_session.MovingSession(env,{'joint_names_runtime':env.rig.names},None,stance,
                warp_to_torch=lambda x:x,output=Path(d)/'raw');env.session=s;s.quiet_only.fill_(True)
            with defer_training_resets(env,s._capture):
                for _ in range(200):s._advance(s.zero,initializing=True)
                runner=MaskedRunner(s,runner_config(),log_dir=str(Path(d)/'logs'),device='cpu');initialize_scratch(runner)
                runner.alg.evidence_callback=s.integrity.bootstrap
                for _ in range(512):
                    obs,reward,done,extra=s.step(s.zero)
                    runner.alg.audit_unoptimized_transition(obs,extra)
                summary=s.integrity.export(Path(d)/'ledger')
                self.assertEqual(summary['completed_finite_recovery_rows'],1)
                self.assertEqual(summary['bootstrap_event_records'],1)
                self.assertTrue(summary['all_records_verified'])
                self.assertEqual(runner.current_learning_iteration,0)
                self.assertEqual(runner.alg.normalizer_rows,0)
                self.assertEqual(runner.alg.storage.step,0)
                self.assertFalse(runner.alg.optimizer.state)
                self.assertEqual(env.reset_ids[0].dtype,torch.int32)
                reset=next(r for r in s.integrity.records if str(r['kind'])=='reset')
                bootstrap=next(r for r in s.integrity.records if str(r['kind'])=='bootstrap')
                self.assertEqual(float(bootstrap['used_next_value'][0]),0.)
                np.testing.assert_array_equal(reset['after__history'][1:],reset['before__history'][1:])
                with self.assertRaises(FileExistsError):s.integrity.export(Path(d)/'ledger')

    def test_modified_final_packet_is_rejected_after_reset(self):
        env=ExactResetBufferEnv();stance={'joint_positions_rad':dict(zip(env.rig.names,env._robot.data.default_joint_pos[0].tolist()))}
        with tempfile.TemporaryDirectory() as d,patch.object(moving_session,'DeviceTelemetry',Reader),patch.object(moving_session,'PhysicsSubstepRecorder',Recorder):
            s=moving_session.MovingSession(env,{'joint_names_runtime':env.rig.names},None,stance,
                warp_to_torch=lambda x:x,output=Path(d)/'raw');env.session=s;s.quiet_only.fill_(True)
            with defer_training_resets(env,s._capture):
                for _ in range(200):s._advance(s.zero,initializing=True)
                for _ in range(15):obs,reward,done,extra=s.step(s.zero)
                data=extra['masked_transition'];data['final_observation']['critic'][0,0]=123.
                with self.assertRaisesRegex(RuntimeError,'final packet differs'):
                    s.integrity.bootstrap(data,torch.zeros(32),obs,source='synthetic_fault',normalization_rows=0)

    def test_actual_RSL_callback_records_final_timeout_values_and_normalization_counts(self):
        env=Episodes();recorded=[]
        with tempfile.TemporaryDirectory() as d:
            runner=MaskedRunner(env,runner_config(),log_dir=d,device='cpu');initialize_scratch(runner)
            def witness(data,value,next_obs,**meta):
                ended=data['terminated']|data['truncated']
                if not ended.any():return
                self.assertEqual(meta['source'],'actual_masked_PPO_storage')
                self.assertFalse(value[data['terminated']].any())
                timeout=data['truncated']
                if timeout.any():
                    # Preserve the actual selected batch: different GEMM batch
                    # sizes can differ by float32 rounding without any state error.
                    need=data['learnable']&~data['terminated']
                    expected=runner.alg.critic(data['final_observation'][need]).detach().squeeze(-1)
                    torch.testing.assert_close(value[need],expected,rtol=0,atol=0)
                    self.assertTrue((next_obs['episode_id'].squeeze(-1)[timeout]>data['final_episode_id'][timeout]).all())
                recorded.append(copy.deepcopy(meta))
            runner.alg.evidence_callback=witness
            reports=runner.collect_updates(2,maximum_updates=2)
            self.assertEqual(len(recorded),3)
            self.assertEqual(sum(r['normalization_rows_this_update'] for r in reports),4*512-60)
            self.assertEqual(reports[-1]['normalization_rows_total'],4*512-60)

if __name__=='__main__':unittest.main()
