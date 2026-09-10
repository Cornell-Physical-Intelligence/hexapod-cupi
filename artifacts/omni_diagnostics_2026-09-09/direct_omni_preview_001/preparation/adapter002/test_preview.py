import ast, copy, hashlib, json, subprocess, sys, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from tensordict import TensorDict
import entry_adapter as entry
import preview_contract as contract
import preview_recorder as recorder
import record_direct_preview as cli

HERE=Path(__file__).resolve().parent

class StructuralTests(unittest.TestCase):
    def test_exact_native_ast_round_trip(self):
        original=(HERE/'inputs/native_entry.py').read_text()
        tree,counts=entry.instrument(original)
        self.assertEqual(counts,dict(app=1,save=1,config=1,render=1,dispatch=1))
        class Undo(ast.NodeTransformer):
            def visit_Expr(self,node):
                self.generic_visit(node)
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name):
                    if node.value.func.id in ('_preview_app_ready','_preview_render_config'):return None
                    if node.value.func.id=='_preview_record':node.value.func.id='function'
                return node
            def visit_Assign(self,node):
                self.generic_visit(node)
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='_preview_bind_save':return None
                return node
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Name) and node.func.id=='OmniFlatEnv':
                    next(k for k in node.keywords if k.arg=='render_mode').value=ast.parse('"rgb_array" if args.mode=="video" else None',mode='eval').body
                return node
        self.assertEqual(ast.dump(Undo().visit(tree)),ast.dump(ast.parse(original)))
        with self.assertRaisesRegex(ValueError,'Unpinned'):entry.instrument(original+'\n')

    def test_app_ready_is_after_actual_constructor(self):
        tree,_=entry.instrument((HERE/'inputs/native_entry.py').read_text())
        index=next(i for i,n in enumerate(tree.body) if entry.same(n,'app = AppLauncher(args).app'))
        calls=[]
        class App:
            def __init__(self,args):calls.append('construct');self.app='actual'
        block=ast.Module(body=tree.body[index:index+2],type_ignores=[])
        exec(compile(block,'ordering','exec'),{'AppLauncher':App,'args':None,'_preview_app_ready':lambda:calls.append('ready')})
        self.assertEqual(calls,['construct','ready'])

    def test_entry_imports_on_stdlib_only_python(self):
        code="import record_direct_preview,sys; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        subprocess.run([sys.executable,'-S','-B','-c',code],cwd=HERE,check=True,capture_output=True)

    def test_native_cli_keeps_evaluate_and_enables_render(self):
        a=types.SimpleNamespace(source_root=Path('/source'),output=Path('/output'),admission=Path('/admission/admission.json'),checkpoint=Path('/checkpoint/evaluated.pt'),device='cuda:0',kit_args='x')
        argv=cli.native_arguments(a)
        self.assertEqual(argv[argv.index('--mode')+1],'evaluate')
        self.assertEqual(argv[argv.index('--direct-evaluation')+1],'constant')
        self.assertIn('--enable_cameras',argv);self.assertNotIn('--num_envs',argv)
        self.assertNotIn('--direct-branch',argv)

    def test_only_late_render_config_changes(self):
        class Config:
            def __init__(self):
                self.scene=types.SimpleNamespace(num_envs=48)
                self.video_recorder=types.SimpleNamespace(window_width=900,window_height=600)
                self.physics={'dt':.0025,'decimation':8,'solver':[16,4],'pd':[30,.6]}
            def to_dict(self):return copy.deepcopy({'scene':vars(self.scene),'video_recorder':vars(self.video_recorder),'physics':self.physics})
        dump=[];mod=types.ModuleType('isaaclab.utils.io');mod.dump_yaml=lambda path,cfg:dump.append(cfg.to_dict())
        with tempfile.TemporaryDirectory() as t,patch.dict(sys.modules,{'isaaclab.utils.io':mod}):
            cfg=Config();entry.configure_rendering(cfg,Path(t));report=contract.read(Path(t)/'rendering_config_delta.json')
            self.assertEqual(dump[0]['scene']['num_envs'],48)
            self.assertEqual(cfg.scene.num_envs,1)
            self.assertEqual(set(report['changed']),{'scene.num_envs','video_recorder.window_width','video_recorder.window_height'})
            with self.assertRaises(ValueError):entry.configure_rendering(cfg,Path(t))

    def test_schedule_exact_real_time(self):
        rows=contract.schedule();self.assertEqual(len(rows),1900)
        self.assertEqual(sum(i%2==1 for i in range(len(rows))),950)
        self.assertEqual(950/25,1900*.02)
        self.assertEqual([rows[i]['segment'] for i in (0,200,500,700,1000,1200,1500)],
            ['QUIET','FORWARD','STOP FORWARD','STRAFE LEFT','STOP STRAFE','FORWARD LEFT ARC','STOP ARC'])

    def test_failure_after_app_close_overrides_completed_state(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);contract.save(out/'state.json',{'status':'completed','checkpoint_sha256':'a'})
            contract.save(out/'video.json',{'complete':True,'frames':950})
            cli.preserve_failure(out,RuntimeError('app close'))
            self.assertEqual(contract.read(out/'state.json')['status'],'failed')
            self.assertFalse(contract.read(out/'video.json')['complete'])
            self.assertEqual(contract.read(out/'video.json')['frames'],950)

    def test_unlisted_source_and_symlink_fail(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'x').write_text('x');contract.save(root/'map.json',contract.tree(root))
            bound=contract.sha(root/'map.json');contract.verify_tree(root,'map.json',bound)
            (root/'extra').write_text('bad')
            with self.assertRaises(ValueError):contract.verify_tree(root,'map.json',bound)
            (root/'extra').unlink();(root/'alias').symlink_to(root/'x')
            with self.assertRaisesRegex(ValueError,'Symlink'):contract.verify_tree(root,'map.json',bound)

    def test_actual_xyzw_pose_and_command_packet(self):
        np.testing.assert_allclose(recorder.actual_pose([1,2,3],[0,0,0,1]),[1,2,-np.pi/2])
        np.testing.assert_allclose(recorder.actual_pose([1,2,3],[0,0,np.sqrt(.5),np.sqrt(.5)]),[1,2,0],atol=1e-15)
        command=torch.tensor([[.1,.2,.3]]);obs={'policy':torch.zeros(1,315),'critic':torch.zeros(1,318)}
        obs['policy'][:,-57:-54]=command*torch.tensor([5.,5.,2.5]);recorder.check_packet(obs,command)
        with self.assertRaises(ValueError):recorder.check_packet(obs,torch.zeros_like(command))
        obs['policy']=torch.zeros(1,495)
        with self.assertRaises(ValueError):recorder.check_packet(obs,command)

    def test_exact_checkpoint_tensors_and_normalizers(self):
        class Module(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(3));self.register_buffer('normalizer',torch.arange(3))
        runner=types.SimpleNamespace(alg=types.SimpleNamespace(actor=Module(),critic=Module()))
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'checkpoint.pt'
            torch.save({'actor_state_dict':runner.alg.actor.state_dict(),'critic_state_dict':runner.alg.critic.state_dict()},path)
            self.assertTrue(recorder.checkpoint_readback(runner,path)['passed'])
            runner.alg.actor.normalizer[0]=5
            with self.assertRaisesRegex(ValueError,'normalizer'):recorder.checkpoint_readback(runner,path)

    def test_native_rgb_size_is_checked(self):
        a=np.zeros((720,1280,3),np.uint8);a[:,640:]=255
        self.assertIs(recorder.validate_rgb(a),a)
        for bad in (None,a[:600],np.zeros_like(a)):
            with self.assertRaises(ValueError):recorder.validate_rgb(bad)

class FakeEnv:
    num_envs=1;step_dt=.02;device='cpu'
    def __init__(self,terminal_at=None):
        self.terminal_at=terminal_at;self.steps=0;self.renders=[];self._commands=torch.zeros(1,3);self.targets=torch.zeros(1,3)
        self.episode_length_buf=torch.zeros(1);self._feet_contact_sensors=[types.SimpleNamespace(body_names=['foot']) for _ in range(6)]
        self._robot=types.SimpleNamespace(num_joints=18,num_bodies=19,joint_names=['j'+str(i) for i in range(18)],
            data=types.SimpleNamespace(root_pos_w=types.SimpleNamespace(torch=torch.tensor([[0.,0.,.13]])),root_quat_w=types.SimpleNamespace(torch=torch.tensor([[0.,0.,0.,1.]]))))
    def reset(self,seed):self._commands.zero_()
    def set_evaluation_targets(self,x):self.targets.copy_(x)
    def _get_observations(self):
        obs={'policy':torch.zeros(1,315),'critic':torch.zeros(1,318)}
        obs['policy'][:,-57:-54]=self._commands*torch.tensor([5.,5.,2.5]);return obs
    def step(self,action):
        self.steps+=1;terminated=torch.tensor([self.steps==self.terminal_at]);truncated=torch.tensor([False])
        self.omni_diagnostic_sample={'terminated':terminated.numpy().copy(),'truncated':truncated.numpy().copy(),
            'quaternion_world_wxyz':np.array([[0.,0.,0.,1.]]),'position_world_m':np.array([[0.,-.001*self.steps,.13]]),
            'velocity_navigation_mps':np.array([[.05,0.,0.]]),'gyro_navigation_rad_s':np.zeros((1,3)),
            'computed_torque_nm':np.zeros((1,18)),'applied_torque_nm':np.zeros((1,18)),
            'command':self._commands.numpy().copy()}
        self._commands.add_((self.targets-self._commands).clamp(-.005,.005))
        return None,None,terminated,truncated,{}
    def render(self):self.renders.append(self.steps);return np.arange(72,dtype=np.uint8).reshape(4,6,3)

class ReceiptTests(unittest.TestCase):
    def fixture(self,root):
        pilot=root/'pilot';pilot.mkdir();checkpoint=root/'selected.pt';checkpoint.write_bytes(b'exact trained final')
        bound=contract.sha(checkpoint);selected={'updates':50}
        phases=['standing','initial_constant','initial_stop','train','final_constant','final_stop']
        receipts={}
        for phase in phases:
            d=pilot/phase;d.mkdir();receipt={'phase':phase,'checkpoint_sha256':bound}
            contract.save(d/'state.json',receipt);receipts[phase]=receipt
            if phase=='standing':contract.save(d/'admission.json',{'gate':{'passed':True}})
            if phase=='train':contract.save(d/'training_receipt.json',{'final_checkpoint_sha256':bound})
            contract.save(pilot/(phase+'_immutable.sha256.json'),contract.tree(d))
        identity={'source_manifest_sha256':contract.SOURCE,'plan_sha256':contract.PLAN,'checkpoint_sha256':contract.ORIGINAL,
                  'actor_width':315,'critic_width':318,'selection':selected}
        campaign={'status':'completed','terminal_inputs_unchanged':True,'host_freeze_sha256':contract.HOST,
            'identity':identity,'allocation':'pilot','branch':'caps','planned_phases':phases,'accepted_phases':receipts}
        contract.save(pilot/'campaign.json',campaign)
        def validate(directory,phase,identity,expected_checkpoint_sha256=None):
            if phase.startswith('final'):self.assertEqual(expected_checkpoint_sha256,bound)
            return contract.read(directory/'state.json')
        native=types.SimpleNamespace(selection=lambda *x:selected,validate_result=validate)
        args=types.SimpleNamespace(pilot=pilot,checkpoint=checkpoint,checkpoint_sha256=bound,admission=pilot/'standing/admission.json')
        return args,native,campaign

    def test_all_six_phases_and_exact_selected_checkpoint_required(self):
        with tempfile.TemporaryDirectory() as t:
            args,native,campaign=self.fixture(Path(t));receipt=contract.verify_pilot(args,native)
            self.assertEqual(receipt['completed_updates'],50)
            args.checkpoint.write_bytes(b'different')
            with self.assertRaisesRegex(ValueError,'checkpoint'):contract.verify_pilot(args,native)

    def test_completed_phase_and_admission_changes_reject(self):
        with tempfile.TemporaryDirectory() as t:
            args,native,campaign=self.fixture(Path(t))
            (args.pilot/'initial_stop/unlisted').write_text('x')
            with self.assertRaisesRegex(ValueError,'tree'):contract.verify_pilot(args,native)
            (args.pilot/'initial_stop/unlisted').unlink()
            alias=Path(t)/'admission.json';contract.save(alias,{'gate':{'passed':False}});args.admission=alias
            with self.assertRaisesRegex(ValueError,'admission'):contract.verify_pilot(args,native)

    def test_incomplete_or_wrong_actor_campaign_rejects(self):
        with tempfile.TemporaryDirectory() as t:
            args,native,campaign=self.fixture(Path(t))
            campaign['status']='failed';contract.save(args.pilot/'campaign.json',campaign)
            with self.assertRaises(ValueError):contract.verify_pilot(args,native)
            campaign['status']='completed';campaign['identity']['actor_width']=495;contract.save(args.pilot/'campaign.json',campaign)
            with self.assertRaises(ValueError):contract.verify_pilot(args,native)

    def test_contract_import_does_not_poison_actual_source_module(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();(root/'direct_config.py').write_text('VALUE=123\n')
            (root/'direct_contract.py').write_text('from direct_config import VALUE\n')
            before=sys.modules.get('direct_config');paths=list(sys.path)
            self.assertEqual(contract.load_native(root).VALUE,123)
            self.assertIs(sys.modules.get('direct_config'),before);self.assertEqual(sys.path,paths)

class LoopTests(unittest.TestCase):
    def run_loop(self,terminal=None):
        env=FakeEnv(terminal);writer=types.SimpleNamespace(frames=[],closed=False)
        writer.append_data=lambda frame:writer.frames.append(frame.copy())
        writer.close=lambda:setattr(writer,'closed',True)
        imageio=types.ModuleType('imageio.v2');imageio.get_writer=lambda *a,**kw:writer;imageio.imwrite=lambda *a,**kw:None
        imageio_package=types.ModuleType('imageio');imageio_package.v2=imageio
        camera=types.ModuleType('isaaclab_physx.renderers.kit_viewport_utils');camera.set_kit_renderer_camera_view=lambda **kw:None
        drawing=types.ModuleType('omni_path_demo');drawing.GroundDrawing=lambda:types.SimpleNamespace(reference=lambda *a:None,line=lambda *a,**kw:None)
        runner=types.SimpleNamespace(get_inference_policy=lambda **kw:lambda obs:torch.zeros(1,18))
        p={'checkpoint_sha256':'x','pilot':{'branch':'caps','completed_updates':50}}
        modules={'imageio':imageio_package,'imageio.v2':imageio,'omni_path_demo':drawing,'isaaclab_physx.renderers.kit_viewport_utils':camera}
        with tempfile.TemporaryDirectory() as t,patch.dict(sys.modules,modules),\
             patch.object(recorder,'checkpoint_readback',return_value={'passed':True}),\
             patch.object(recorder,'command_reference',return_value=(np.zeros((1900,3)),np.zeros((1900,3)))),\
             patch.object(recorder,'initial_rgb_frame',side_effect=lambda env:(env.render(),{'physics_steps':0})),\
             patch.object(recorder,'draw_live_command'),patch.object(recorder,'annotate',side_effect=lambda raw,*a,**kw:raw.copy()),\
             patch.object(recorder,'validate_rgb',side_effect=lambda x:x),\
             patch.object(recorder,'native_inference_origins',return_value={}),\
             patch.object(recorder,'verify_inputs',return_value=p):
            args=types.SimpleNamespace(checkpoint_sha256='x',checkpoint=Path(t)/'input.pt',seed=7057,source_root=Path(t))
            if terminal:
                with self.assertRaisesRegex(RuntimeError,'Incomplete'):recorder.record(env,runner,{},Path(t),'x',args,p)
            else:recorder.record(env,runner,{},Path(t),'x',args,p)
            report=contract.read(Path(t)/'video.json')
            with np.load(Path(t)/'trace.npz') as z:trace={k:z[k].copy() for k in z.files}
        return env,writer,report,trace

    def test_complete_actual_control_loop_timing(self):
        env,writer,result,trace=self.run_loop()
        self.assertTrue(result['complete']);self.assertEqual(env.steps,1900);self.assertEqual(len(writer.frames),950)
        self.assertEqual(env.renders,[0]+list(range(2,1901,2)));self.assertTrue(writer.closed)
        self.assertEqual(trace['deterministic_actor_action'].shape,(1900,1,18))
        self.assertEqual(trace['time_s'][-1],38.)
        np.testing.assert_array_equal(trace['quaternion_world_xyzw'],trace['quaternion_world_wxyz'])
        np.testing.assert_array_equal(trace['quaternion_world_wxyz_converted'][0,0],[1,0,0,0])
        self.assertEqual(result['telemetry'][200]['actor_observed_command'],[0,0,0])
        self.assertGreater(result['telemetry'][201]['actor_observed_command'][0],0)

    def test_first_terminal_preserves_raw_without_reset_render(self):
        env,writer,result,trace=self.run_loop(7)
        self.assertFalse(result['complete']);self.assertEqual(env.steps,7)
        self.assertEqual(env.renders,[0,2,4,6]);self.assertEqual(len(writer.frames),4)
        self.assertTrue(trace['terminated'][-1,0]);self.assertEqual(result['terminal_event']['control_step'],7)
        self.assertEqual(result['recorded_control_steps'],7)

    def test_warmup_advances_no_physics(self):
        from render_helpers import initial_rgb_frame
        calls=[]
        env=types.SimpleNamespace(sim=types.SimpleNamespace(render=lambda:calls.append('render_only')))
        sequence=iter([None,None,np.arange(72).reshape(4,6,3)])
        env.render=lambda:next(sequence)
        _,report=initial_rgb_frame(env)
        self.assertEqual(calls,['render_only','render_only']);self.assertEqual(report['physics_steps'],0)

if __name__=='__main__':unittest.main()
