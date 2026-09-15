"""Synthetic orchestration fixtures; never native BC or qualification evidence."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import json
import sys
import unittest
import numpy as np
import torch
from experiments.paper_walk import evaluate as run
from experiments.paper_walk import train
from experiments.paper_walk.tests.test_evaluate import FakeNative, FakeCapture


class Native(FakeNative):
    def __init__(self,terminate_at=None):
        super().__init__(terminate_at)
        self.total_controls=0;self.neutral=torch.zeros(18);self.held=torch.zeros(1,18)
        self.history=torch.zeros(1,5,42)
        self.native_readback={'limits':np.tile([[-2.,2.]],(1,18,1)).tolist(),
                             'native_max_velocity':np.full((1,18),20.).tolist()}
        self.events=[];self.missing_contact=False;self.on_step=None
        self.cfg=run.EvaluationEnvConfig(num_envs=1,render=True)

    def _observations(self,state):
        obs=torch.cat((self.history.flatten(1),self.commands,self.held/.35),-1)
        return {'obs':obs,'critic':torch.cat((obs,torch.zeros(1,3)),-1),'amp':torch.zeros(1,61)}

    def step(self,action):
        self.history=torch.roll(self.history,-1,1);self.history[:,-1]=self.steps+1
        self.held=self.held+(action.clamp(-1,1)*.35-self.held).clamp(-.04,.04)
        result=super().step(action);self.total_controls+=1
        if self.on_step:self.on_step()
        return result

    def render(self,index=0):return np.ones((2,2,3),np.uint8)


class Actor(torch.nn.Module):
    def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.full((18,),.1))
    def actor(self,obs):return self.bias.expand(len(obs),-1),torch.zeros(len(obs),3)


class AMP(torch.nn.Module):
    def __init__(self):super().__init__();self.discriminator=torch.nn.Linear(122,1)
    def style_reward(self,a,b):return torch.zeros(len(a))
    def features(self,a,b):return torch.cat((a,b),-1)


class Policy:
    def __init__(self):self.model=Actor();self.amp=AMP();self.mismatch=False
    def act(self,obs,deterministic=True):return self.model.actor(obs)[0]+(1 if self.mismatch else 0)


class Capture(FakeCapture):
    def __init__(self,env,output,geometry):
        super().__init__(env,output,geometry);self.output=Path(output);self.output.mkdir(parents=True)
        self.start=env.counter

    def control_record(self,result,command,control):
        row=super().control_record(result,command,control)
        row['command']=self.env.commands.numpy().copy()
        row['joint_target_rad']=self.env.held.numpy().copy()
        row['velocity_navigation_mps']=np.asarray(command)
        return row

    def close(self):
        receipt=super().close();n=self.count;seq=np.arange(n);root=np.zeros((n,1,7));root[:,:,2]=.1;root[:,:,6]=1
        zero=np.zeros((n,1,18));nonfoot=np.zeros((n,1),bool)
        for i in self.env.events:
            if i<n:nonfoot[i]=True
        raw={'sequence':seq,'control_index':seq//8,'substep_index':seq%8,'explicit_counter':self.start+seq+1,
             'joint_position_rad':zero,'joint_velocity_rad_s':zero,'applied_torque_nm':zero,
             'computed_torque_nm':zero,'native_input_pre_nm':zero,'root_pose_xyzw':root,
             'nonfoot_contact':nonfoot,'minimum_non_toe_floor_m':np.full((n,1),.01),
             'distal_contact':np.ones((n,1,6),bool)}
        file=self.output/'substeps_000.npz';np.savez_compressed(file,**raw)
        contacts=self.output/'contacts.jsonl'
        contacts.write_text(''.join(json.dumps({'sequence':int(i),'explicit_counter':int(self.start+i+1),'patches':[]})+'\n'
                                    for i in seq[:n-1 if self.env.missing_contact else n]))
        receipt.update(initial_counter=self.start,final_counter=self.start+n,
            substep_files=[file.name],files={file.name:run._sha(file),contacts.name:run._sha(contacts)},
            nonfoot_contact_steps_400hz=[int(nonfoot.sum())])
        return receipt


class StartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def invoke(self,root,arm='neutral4',native=None,policy=None,**kwargs):
        native=native or Native();policy=policy or Policy()
        geometry=Path(root)/'geometry';geometry.write_bytes(b'synthetic geometry')
        checkpoint=Path(root)/'checkpoint';checkpoint.write_bytes(b'synthetic checkpoint identity')
        with patch.object(run,'ExactEvaluationCapture',Capture):
            result=run.run_startup_diagnostic(native,policy,Path(root)/'output',geometry,checkpoint,
                arm=arm,record_video=kwargs.pop('record_video',False),**kwargs)
        return result,native,policy

    def test_both_exact_windows_handoff_native_history_and_unchanged_manifests(self):
        full=run.case_manifest();probes=run.learning_probe_cases()
        for arm,prefix in [('cold',0),('neutral4',200)]:
            with self.subTest(arm=arm),TemporaryDirectory() as root,patch.object(run.scoring,'score_recording',wraps=run.scoring.score_recording) as score:
                result,native,_=self.invoke(root,arm)
                self.assertEqual((native.reset_count,native.steps),(1,prefix+1000))
                self.assertEqual(result['recorded_physics_steps'],(prefix+1000)*8)
                scored=score.call_args.args[0]
                self.assertEqual(len(scored['time_s']),1000)
                self.assertEqual(result['results'][0]['window_start_control'],100)
                self.assertEqual(float(scored['time_s'][0]),(prefix+1)*.02)
                self.assertTrue(result['startup_physical_windows']['pass'])
                self.assertFalse(result['stage2_complete'])
                self.assertEqual(len(result['required_cases_not_evaluated']),96)
                self.assertEqual(len(result['learning_probes_not_evaluated']),13)
                handoff=json.loads((Path(root)/'output/policy_handoff.json').read_text())
                self.assertEqual(handoff['global_control_index'],prefix)
                self.assertEqual(handoff['physics_counter'],prefix*8)
                self.assertFalse(handoff['extra_reset_or_physics_step'])
                self.assertEqual(np.abs(handoff['actual_held_target_rad']).max(),0)
                with np.load(Path(root)/'output/control_trace.npz') as d:
                    np.testing.assert_array_equal(d['global_control_index'],np.arange(prefix+1000))
                    np.testing.assert_array_equal(d['policy_control_index'][prefix:],np.arange(1000))
                    np.testing.assert_array_equal(d['policy_observation'][:,:,210:213],d['command'])
                    self.assertEqual(d['action_source'][:prefix].tolist(),['scripted_neutral']*prefix)
                    self.assertEqual(d['action_source'][prefix:].tolist(),['bc']*1000)
                    np.testing.assert_array_equal(d['issued_action'][prefix:],d['actor_mean_action'][prefix:])
                    if prefix:
                        self.assertEqual(abs(d['issued_action'][:prefix]).max(),0)
                        self.assertGreater(abs(d['actor_mean_action'][:prefix]).max(),0)
                        self.assertEqual(abs(d['joint_target_rad'][:prefix]).max(),0)
                        np.testing.assert_array_equal(d['policy_observation'][200,0,:210].reshape(5,42)[:,0],np.arange(196,201))
                        self.assertEqual(abs(d['policy_observation'][200,0,213:]).max(),0)
                        self.assertEqual(abs(d['command'][:200]).max(),0)
                        self.assertAlmostEqual(float(d['command'][200,0,0]),.05)
        self.assertEqual(full,run.case_manifest());self.assertEqual(probes,run.learning_probe_cases())

    def test_prefix_and_policy_mid_hold_contacts_cannot_hide_in_full_average(self):
        for events,failed_window in [([3],'scripted_prefix'),(list(range(1601,1610)),'policy')]:
            with self.subTest(window=failed_window),TemporaryDirectory() as root:
                native=Native();native.events=events;result,_,_=self.invoke(root,native=native)
                windows=result['startup_physical_windows']['windows']
                self.assertTrue(windows['full']['pass'])
                self.assertFalse(windows[failed_window]['pass'])
                self.assertFalse(result['diagnostic_policy_screen_passed'])

    def test_terminal_prefix_and_missing_contact_packets_fail_closed(self):
        for mode in ('terminal','contacts'):
            with self.subTest(mode=mode),TemporaryDirectory() as root:
                native=Native(terminate_at=4 if mode=='terminal' else None);native.missing_contact=mode=='contacts'
                result,_,_=self.invoke(root,native=native)
                self.assertFalse(result['diagnostic_policy_screen_passed'])
                self.assertFalse(result['startup_physical_windows']['pass'])
                if mode=='terminal':
                    self.assertEqual((result['controls'],result['recorded_physics_steps']),(4,32))
                    self.assertEqual(result['failure_kind'],'native_terminal_prefix')
                    self.assertEqual(result['results'][0]['failed_bounds'][0],'incomplete_native_prefix')

    def test_actor_issued_mismatch_checked_on_first_bc_row_only_explicit_prefix_skips(self):
        with TemporaryDirectory() as root:
            policy=Policy();policy.mismatch=True
            result,native,_=self.invoke(root,policy=policy)
            self.assertEqual(native.steps,201)
            self.assertEqual(result['controls'],200)
            self.assertEqual(result['failure_kind'],'acquisition_error')
            self.assertFalse(result['diagnostic_policy_screen_passed'])

    def test_changed_policy_or_checkpoint_blocks_success_and_reuse_rejected(self):
        for mode in ('model','checkpoint'):
            with self.subTest(mode=mode),TemporaryDirectory() as root:
                native=Native();policy=Policy()
                def change():
                    if native.steps==10:
                        if mode=='model':policy.model.bias.data.add_(.1)
                        else:(Path(root)/'checkpoint').unlink()
                native.on_step=change
                result,_,_=self.invoke(root,native=native,policy=policy)
                self.assertIsNotNone(result['integrity_failure']);self.assertFalse(result['diagnostic_policy_screen_passed'])
                with self.assertRaisesRegex(ValueError,'fresh one-robot'):
                    run.run_startup_diagnostic(native,policy,Path(root)/'again',Path(root)/'geometry',Path(root)/'checkpoint',arm='cold',record_video=False)

    def test_deadline_in_prefix_and_nonstartup_cases_cannot_gain_prefix(self):
        with TemporaryDirectory() as root:
            native=Native()
            with patch.object(run.time,'monotonic',side_effect=lambda:float(native.steps)):
                result,_,_=self.invoke(root,native=native,max_wall_seconds=50.)
            self.assertEqual(result['controls'],100);self.assertTrue(result['allocation_limit_reached'])
            self.assertFalse(result['diagnostic_policy_screen_passed'])
        with TemporaryDirectory() as root:
            native=Native()
            with patch.object(run.time,'monotonic',side_effect=lambda:float(native.steps)):
                result,_,_=self.invoke(root,native=native,max_wall_seconds=1200.)
            self.assertEqual(result['controls'],1200);self.assertTrue(result['allocation_limit_reached'])
            self.assertFalse(result['diagnostic_policy_screen_passed'])
        with self.assertRaisesRegex(ValueError,'exact additional'):
            run.run_batch(Native(),Policy(),[run.learning_probe_cases()[0]],'unused','unused',
                checkpoint_sha256='synthetic',source_sha256='synthetic',_startup_prefix_controls=0)

    def test_video_covers_prefix_and_policy_and_finalization_failure_is_not_success(self):
        for fail_close in (False,True):
            with self.subTest(fail_close=fail_close),TemporaryDirectory() as root:
                frames=[]
                def close():
                    if fail_close:raise RuntimeError('synthetic finalization failure')
                def writer(path,**kwargs):
                    Path(path).write_bytes(b'synthetic temporary video fixture')
                    return SimpleNamespace(append_data=lambda frame:frames.append(frame.copy()),close=close)
                imageio=SimpleNamespace(get_writer=writer,imwrite=lambda path,frame:Path(path).write_bytes(b'synthetic frame'))
                with patch.dict(sys.modules,{'imageio':SimpleNamespace(v2=imageio),'imageio.v2':imageio}):
                    result,_,_=self.invoke(root,record_video=True)
                self.assertEqual(result['video_frames'],600);self.assertEqual(len(frames),600)
                self.assertEqual(result['controls'],1200)
                self.assertEqual(result['acquisition_complete'],not fail_close)
                if fail_close:
                    self.assertEqual(result['failure_kind'],'acquisition_error')
                    self.assertFalse(result['diagnostic_policy_screen_passed'])

    def test_cli_scope_requires_exact_arm_and_no_command_override(self):
        base=['--mode','evaluate','--preflight-only']+[x for name in ('asset','model','geometry','geometry-extrema','prior','prior-metadata','output') for x in ('--'+name,'/unused/'+name)]
        for flags in (['--eval-scope','startup'],['--startup-arm','cold'],
                      ['--eval-scope','startup','--startup-arm','cold','--command','.1','0','0']):
            with self.subTest(flags=flags),self.assertRaises(ValueError):train.main(base+flags)
        for arm in ('cold','neutral4'):
            with patch.object(train,'verify_assets',side_effect=RuntimeError('Reached assets')):
                with self.assertRaisesRegex(RuntimeError,'Reached assets'):
                    train.main(base+['--eval-scope','startup','--startup-arm',arm])


if __name__=='__main__':unittest.main()
