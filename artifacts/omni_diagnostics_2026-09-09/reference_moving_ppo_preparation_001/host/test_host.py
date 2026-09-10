import ast, importlib.util, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('moving_host',Path(__file__).with_name('launch_moving_ppo_spark.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

class HostTest(unittest.TestCase):
    def args(self,base):
        return SimpleNamespace(**{k:base/k for k in ['source','run','device_run','bridge','consumer','observation','output']})

    def test_phase_modes_and_readonly_completed_dependencies(self):
        args=self.args(Path('/private/test'))
        for phase in h.PHASES[1:]:
            c=h.command(args,'owned',phase);mounts=[c[i+1] for i,v in enumerate(c) if v=='-v']
            self.assertEqual(sum(v.endswith(':rw') for v in mounts),1)
            self.assertIn('/private/test/consumer:/consumer:ro',mounts)
            self.assertIn('/private/test/output/inputs/study:/study:ro',mounts)
            self.assertIn('/private/test/output/inputs:/outputs/inputs:ro',mounts)
            self.assertLess(mounts.index('/private/test/output:/outputs:rw'),mounts.index('/private/test/output/inputs:/outputs/inputs:ro'))
            for prior in h.PHASES[:h.PHASES.index(phase)]:
                self.assertIn('/private/test/output/'+prior+':/outputs/'+prior+':ro',mounts)
            self.assertEqual(c[c.index('--mode')+1],h.MODE.get(phase,phase))
            self.assertEqual(c[c.index('--output')+1],'/outputs/'+phase)
            self.assertIn('/consumer/run_moving_ppo.py',c)
            self.assertNotIn('--decision-receipt',c)
        for mode in ('train_25','profile_128','train_forever'):
            with self.assertRaises(ValueError):h.command(args,'owned',mode)
        self.assertEqual(h.PHASE_SECONDS['train_10'],1800)
        self.assertEqual(sum(h.PHASE_SECONDS.values()),5400)

    def test_stdlib_only_import(self):
        source="import runpy,sys;runpy.run_path(sys.argv[1]);assert not {'numpy','torch'}&set(sys.modules)"
        subprocess.run([sys.executable,'-S','-c',source,h.__file__],check=True)

    def test_parent_owns_unchanged_standing(self):
        args=self.args(Path('/test'))
        with patch.object(h,'PARENT',SimpleNamespace(command=lambda *a:list(a))):
            self.assertEqual(h.command(args,'name','standing'),[args.source,args.output,'name','standing'])

    def exercise(self,fail=None,mutation=None,wrong_checkpoint=False,wrong_mode=False,wrong_raw=False,identity_change=False):
        with tempfile.TemporaryDirectory() as d:
            args=self.args(Path(d));args.output.mkdir();calls=[];identity={'bound':True}
            for phase in h.PHASES:(args.output/phase).mkdir()
            (args.output/'standing/admission.json').write_text('{}')
            (args.output/'calibration/initial.pt').write_bytes(b'exact initial checkpoint')
            (args.output/'train_10/decision_010.pt').write_bytes(b'exact decision checkpoint')
            def run(a,phase):
                calls.append(phase)
                if mutation and phase==mutation[0]:(args.output/mutation[1]).write_bytes(b'changed completed evidence')
                prior=('train_10','decision_010.pt') if phase in ('evaluate_010','quiet_010') else ('calibration','initial.pt')
                return {'input_checkpoint_sha256':'wrong' if wrong_checkpoint else h.sha(args.output/prior[0]/prior[1])}
            def validate(p,i):
                if p.name==fail:raise ValueError('rejected result')
                return {'passed':True,'mode':'wrong' if wrong_mode else h.MODE.get(p.name,p.name),'phase':p.name,
                        'source_identity':i,'files_sha256':{} if wrong_raw else h.tree_hashes(p)}
            def inputs(a,phase='calibration',require_standing=False):
                return {'bound':False} if identity_change and len(calls)>=3 else identity
            consumer=SimpleNamespace(verify_standing=lambda *a:None,validate_result=validate)
            with patch.object(h,'validate_inputs',side_effect=inputs),patch.object(h,'run_owned',side_effect=run),patch.object(h,'PARENT',SimpleNamespace(check_source=lambda p:None)),patch.object(h,'CONSUMER',consumer):
                if fail or mutation or wrong_checkpoint or wrong_mode or wrong_raw or identity_change:
                    with self.assertRaises(ValueError):h.run_phases(args,identity,{})
                else:
                    report={};h.run_phases(args,identity,report)
                    self.assertEqual(report['PPO_updates_completed'],10)
            return calls

    def test_seven_serial_phases(self):self.assertEqual(self.exercise(),list(h.PHASES))
    def test_failed_calibration_prevents_learning(self):self.assertEqual(self.exercise(fail='calibration'),list(h.PHASES[:2]))
    def test_failed_profile_prevents_learning(self):self.assertEqual(self.exercise(fail='profile_32'),list(h.PHASES[:3]))
    def test_failed_training_prevents_screen(self):self.assertEqual(self.exercise(fail='train_10'),list(h.PHASES[:4]))
    def test_rejected_initial_prevents_next_screen(self):self.assertEqual(self.exercise(fail='evaluate_initial'),list(h.PHASES[:5]))
    def test_completed_checkpoint_mutation_caught_immediately(self):
        self.assertEqual(self.exercise(mutation=('profile_32','calibration/initial.pt')),list(h.PHASES[:3]))
    def test_standing_mutation_caught_immediately(self):
        self.assertEqual(self.exercise(mutation=('calibration','standing/admission.json')),list(h.PHASES[:2]))
    def test_final_screen_cannot_mutate_prior_result(self):
        self.assertEqual(self.exercise(mutation=('quiet_010','evaluate_010/added.bin')),list(h.PHASES))
    def test_checkpoint_phase_binding(self):self.assertEqual(self.exercise(wrong_checkpoint=True),list(h.PHASES[:3]))
    def test_cli_mode_mapping_binding(self):self.assertEqual(self.exercise(wrong_mode=True),list(h.PHASES[:2]))
    def test_raw_receipt_binding(self):self.assertEqual(self.exercise(wrong_raw=True),list(h.PHASES[:2]))
    def test_changed_source_stops_before_learning(self):self.assertEqual(self.exercise(identity_change=True),list(h.PHASES[:3]))

    def test_owned_supervision_parity_except_explicit_deadlines(self):
        p=Path(os.environ['MOVING_HOST_PARENT_SOURCE']) if 'MOVING_HOST_PARENT_SOURCE' in os.environ else next((root/'tmp/reference_residual_ppo_launch_002/launch_residual_ppo_spark.py' for root in Path(__file__).resolve().parents if (root/'tmp/reference_residual_ppo_launch_002/launch_residual_ppo_spark.py').is_file()), Path(__file__).resolve().parents[1]/'reference_residual_ppo_launch_002/launch_residual_ppo_spark.py')
        old=ast.parse(p.read_text());new=ast.parse(Path(h.__file__).read_text())
        find=lambda tree,name:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        self.assertEqual(ast.dump(find(old,'owned_container')),ast.dump(find(new,'owned_container')))
        source=ast.get_source_segment(p.read_text(),find(old,'run_owned'))
        source=source.replace('deadline_seconds=600,','deadline_seconds=PHASE_SECONDS[phase],').replace('deadline = time.monotonic() + 600','deadline = time.monotonic() + PHASE_SECONDS[phase]').replace('raise TimeoutError("Standing phase exceeded ten-minute bound")','raise TimeoutError(f"{phase} exceeded declared {PHASE_SECONDS[phase]} second bound")')
        self.assertEqual(ast.dump(ast.parse(source).body[0]),ast.dump(find(new,'run_owned')))

if __name__=='__main__':unittest.main()
