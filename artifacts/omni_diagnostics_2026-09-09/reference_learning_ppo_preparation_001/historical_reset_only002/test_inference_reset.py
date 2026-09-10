"""Actual SDK reset-buffer allocation semantics, executed through the session."""
import ast
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
from test_recovery import Env,Reader,Recorder
import moving_session
from recovery import defer_training_resets


class ExactResetBufferEnv(Env):
    def __init__(self):
        super().__init__();self.reset_modes=[]

    def step(self,actions):
        result=super().step(actions)
        # Byte-bound installed DirectRLEnv.step performs this assignment while
        # the caller's existing inference context is active. A copied normal
        # constructor buffer does not reproduce the first actual failure.
        self.reset_buf=self.reset_terminated | self.reset_time_outs
        assert torch.is_inference(self.reset_buf)
        return result

    def _reset_idx(self,ids):
        self.reset_modes.append(torch.is_inference_mode_enabled())
        return super()._reset_idx(ids)


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def run_reset(self,old=False):
        env=ExactResetBufferEnv();stance={'joint_positions_rad':dict(zip(env.rig.names,env._robot.data.default_joint_pos[0].tolist()))}
        with tempfile.TemporaryDirectory() as d,patch.object(moving_session,'DeviceTelemetry',Reader),patch.object(moving_session,'PhysicsSubstepRecorder',Recorder):
            s=moving_session.MovingSession(env,{'joint_names_runtime':env.rig.names},None,stance,
                warp_to_torch=lambda x:x,output=Path(d)/'raw');env.session=s;s.quiet_only.fill_(True)
            if old:
                source=Path(__file__).resolve().parent/'inputs/moving_session001.py.txt'
                self.assertEqual(__import__('hashlib').sha256(source.read_bytes()).hexdigest(),'2ce5791eaa83c05de7f33fbf0eaa2fd16959f5100214f10e92f3feadaba176e9')
                tree=ast.parse(source.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='MovingSession')
                method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_reset_rows')
                namespace={'torch':torch};exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),namespace)
                s._reset_rows=types.MethodType(namespace['_reset_rows'],s)
            with defer_training_resets(env,s._capture):
                for _ in range(200):s._advance(s.zero,initializing=True)
                for _ in range(14):s.step(s.zero)
                if old:
                    with self.assertRaisesRegex(RuntimeError,'Inplace update to inference tensor outside InferenceMode'):
                        s.step(s.zero)
                    self.assertEqual(env.reset_modes,[])
                    self.assertFalse(env.reset_ids)
                    return
                out,reward,done,extra=s.step(s.zero)
                self.assertEqual(env.reset_modes,[True])
                self.assertEqual(env.reset_ids[0].tolist(),[0])
                self.assertEqual(float(reward[0]),-3.)
                self.assertEqual(int(done.sum()),1)
                self.assertEqual(int(out['episode_id'][0]),1)
                self.assertFalse(out['learning_valid'][0])
                self.assertFalse(torch.is_inference_mode_enabled())
                for _ in range(200):s.step(s.zero)
                self.assertTrue(s.active.all())
                self.assertFalse(s.encoder.last_fd_valid[0])
                self.assertTrue(s.encoder.last_fd_valid[1:].all())
                self.assertEqual(int(s.encoder.history_valid[0].sum()),1)
                self.assertTrue((s.encoder.history_valid[1:].sum(-1)==5).all())
                self.assertTrue(s.freshness.ready.all())
                self.assertEqual(env.control,415)
                self.assertFalse(torch.is_inference_mode_enabled())

    def test_old_consumer_reproduces_actual_inference_buffer_failure(self):self.run_reset(old=True)
    def test_successor_restores_only_original_reset_context_and_recovery(self):self.run_reset()

if __name__=='__main__':unittest.main()
