import ast, copy, hashlib, importlib.util, json, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
import yaml
from unittest.mock import patch
from direct_config import selection, configure, protocol
from direct_contract import runtime_arguments, validate_result, sha
from build_source import patch_entry
from direct_stop_evaluation import target_at
from direct_quiet_metrics import quiet_metrics, QUIET_GATES
from direct_training import audited_environment, equal_tree

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'baseline/inputs/old_entry.py'

class NativeTests(unittest.TestCase):
    def test_shared_allocation_and_branch(self):
        cfg=yaml.safe_load((HERE.parent/'training/inputs/agent.yaml').read_text())
        for allocation,n,updates in [('smoke',32,2),('pilot',1024,50)]:
            for branch in ['curriculum','caps','quiet_priority']:
                selected=selection('train',allocation,branch,None,updates)
                self.assertEqual((selected['replicas'],selected['updates']),(n,updates))
                result=configure(cfg,selected)
                self.assertEqual((result['num_steps_per_env'],result['max_iterations']),(24,updates))
                self.assertEqual(result['algorithm']['caps_options']['temporal_weight'],.1 if branch!='curriculum' else 0.)
        for mode,allocation,branch,evaluation,iterations in [('train','pilot','caps',None,10),('train','smoke','caps',None,50),('train','smoke','caps',None,True),('evaluate',None,None,None,None),('validate','pilot',None,None,None)]:
            with self.assertRaises(ValueError): selection(mode,allocation,branch,evaluation,iterations)

    def test_scope_command_checkpoint_mounts(self):
        for phase in ['standing','train','final_constant','final_stop']:
            argv=runtime_arguments(phase,'smoke','quiet_priority')
            if phase.startswith('final_'): self.assertEqual(argv[argv.index('--checkpoint')+1],'/checkpoint/evaluated.pt')
            if phase=='train': self.assertEqual(argv[argv.index('--iterations')+1],'2')
        for phase in ['initial_constant','initial_stop']:
            with self.assertRaises(ValueError): runtime_arguments(phase,'smoke','quiet_priority')
        self.assertEqual(runtime_arguments('train','pilot','caps')[-1],'50')

    def test_entry_only_declared_physics_allocation_delta(self):
        original=OLD.read_text().replace('from repair_length_study_inertias import repair_and_verify','from candidate_asset_audit import audit_candidate_usd as repair_and_verify')
        result=patch_entry(original)
        def functions(text): return {n.name:ast.dump(n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        before=functions(original);after=functions(result)
        self.assertEqual({k:v for k,v in before.items() if k!='main'},{k:v for k,v in after.items() if k!='main'})
        def physics(text):
            main=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='main')
            return [ast.dump(n) for n in ast.walk(main) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ('cfg','env_module') for x in ast.walk(n.targets[0]))]
        expected=physics(original)
        added=ast.dump(ast.parse('cfg.scene.num_envs = direct_selection["replicas"]').body[0])
        actual=physics(result);self.assertEqual(actual.count(added),1);actual.remove(added);self.assertEqual(actual,expected)
        self.assertIn('allow_abbrev=False',result)
        # Newly declared selectors are parsed before AppLauncher, not an ignored flag.
        self.assertLess(result.index('direct_selection = selection'),result.index('app = AppLauncher'))

    def test_frozen_behavior_bytes_reused(self):
        old=json.loads((HERE.parent/'training/FREEZE_SHA256.json').read_text())
        for name in ['curriculum.py']:
            self.assertEqual(sha(HERE/name),old[name])

    def test_actual_late_app_flags_parse_before_launch(self):
        text=patch_entry(OLD.read_text());tree=ast.parse(text)
        start=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='parser' for x in n.targets))
        end=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='direct_selection' for x in n.targets))
        block=ast.Module(body=tree.body[start:end+1],type_ignores=[])
        import argparse
        class Launcher:
            @staticmethod
            def add_app_launcher_args(p):
                p.add_argument('--device');p.add_argument('--headless',action='store_true');p.add_argument('--info',action='store_true');p.add_argument('--kit_args')
        old=sys.argv
        try:
            sys.argv=['entry',*runtime_arguments('train','smoke','quiet_priority')[1:],'--info']
            ns={'argparse':argparse,'Path':Path,'__doc__':'test','AppLauncher':Launcher}
            exec(compile(block,'entry','exec'),ns)
            self.assertEqual(ns['direct_selection']['updates'],2);self.assertEqual(ns['args'].device,'cuda:0');self.assertTrue(ns['args'].headless)
        finally:sys.argv=old

    def test_real_stop_schedule_has_motion_then_ten_second_quiet(self):
        target=torch.tensor([[.1,0.,.2]])
        actual=torch.stack([target_at(i,target) for i in range(1600)])
        self.assertTrue(torch.all(actual[:200]==0));self.assertTrue(torch.all(actual[200:800]==target));self.assertTrue(torch.all(actual[800:]==0))
        self.assertEqual(int((actual[:,:,0]!=0).sum()),600)
        for bad in [-1,1600,True]:
            with self.assertRaises(ValueError):target_at(bad,target)

    def test_quiet_yaw_and_failure_not_hidden_by_reset(self):
        n=1600;q=np.zeros((n,1,18));quat=np.zeros((n,1,4));quat[...,0]=1
        data={'joint_position_rad':q.copy(),'joint_target_rad':q.copy(),'joint_velocity_rad_s':q.copy(),'computed_torque_nm':q.copy(),'applied_torque_nm':q.copy(),'position_world_m':np.zeros((n,1,3)), 'quaternion_world_wxyz':quat,'terminated':np.zeros((n,1),bool),'truncated':np.zeros((n,1),bool)}
        names=[str(i) for i in range(18)]
        row=quiet_metrics(data,0,1100,names,.02);self.assertTrue(row['pass'])
        yaw=np.linspace(0,np.radians(10),500);data['quaternion_world_wxyz'][1100:,0,0]=np.cos(yaw/2);data['quaternion_world_wxyz'][1100:,0,3]=np.sin(yaw/2)
        row=quiet_metrics(data,0,1100,names,.02);self.assertAlmostEqual(row['max_heading_excursion_deg'],10);self.assertFalse(row['pass'])
        data['quaternion_world_wxyz']=quat.copy();data['terminated'][10,0]=True
        self.assertFalse(quiet_metrics(data,0,1100,names,.02)['pass'])

    def test_stdio_host_import_has_no_torch_or_numpy(self):
        command='import sys;sys.path.insert(0,'+repr(str(HERE))+');import direct_contract;assert "torch" not in sys.modules and "numpy" not in sys.modules'
        subprocess.run([sys.executable,'-S','-c',command],check=True)

    def test_completed_smoke_campaign_boundary_not_just_phase_states(self):
        from direct_contract import verify_smoke_campaign
        identity={'source':'bound','selection':selection('train','smoke','quiet_priority',None,None)}
        phases=['standing','train','final_constant','final_stop']
        receipts={p:{'phase':p,'checkpoint_sha256':'c'*64} for p in phases}
        campaign={'status':'completed','terminal_inputs_unchanged':True,'allocation':'smoke','branch':'quiet_priority','identity':identity,'planned_phases':phases,'accepted_phases':receipts}
        with tempfile.TemporaryDirectory() as d,patch('direct_contract.validate_result',side_effect=lambda directory,phase,*a,**kw:receipts[phase]):
            p=Path(d)/'campaign.json'
            p.write_text(json.dumps(campaign));self.assertEqual(verify_smoke_campaign(Path(d),identity),sha(p))
            for change in [{'status':'failed'},{'terminal_inputs_unchanged':False},{'identity':{}},{'branch':'curriculum'},{'planned_phases':['standing','train']}]:
                p.write_text(json.dumps({**campaign,**change}))
                with self.assertRaises(ValueError):verify_smoke_campaign(Path(d),identity)

    def test_real_new_reload_helper_from_previously_trained_cpu_checkpoint(self):
        # Read/reload only; do not repeat the already completed two-update regression.
        old=HERE.parent/'training';spec=importlib.util.spec_from_file_location('_prior_regression',old/'real_rsl_regression.py');module=importlib.util.module_from_spec(spec)
        sys.path.insert(0,str(old))
        try:spec.loader.exec_module(module)
        finally:sys.path.pop(0)
        from rsl_rl.runners import OnPolicyRunner
        from caps import CapsPairWrapper
        from direct_training import verify_reload
        env=CapsPairWrapper(module.Synthetic());runner=OnPolicyRunner(env,module.cfg('caps'),device='cpu')
        checkpoint=old/'cpu_smoke_002/caps/final.pt';runner.load(str(checkpoint),map_location='cpu')
        result=verify_reload(runner,env,checkpoint)
        self.assertTrue(result['passed']);self.assertEqual(result['optimizer_entries'],17)

if __name__=='__main__':unittest.main()
