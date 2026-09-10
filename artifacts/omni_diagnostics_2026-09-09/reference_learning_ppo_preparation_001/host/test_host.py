import ast, importlib.util, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('learning_host',Path(__file__).with_name('launch_learning_ppo_spark.py'))
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

class HostTest(unittest.TestCase):
    def args(self,base):
        return SimpleNamespace(**{k:base/k for k in ['source','run','device_run','bridge','consumer','observation','output']},phase_group='admission',decision_receipt=base/'decision.json')

    def test_phase_modes_and_readonly_completed_dependencies(self):
        args=self.args(Path('/private/test'))
        args.prior_file_hashes={name:'exact' for name in ('campaign.json','standing_accepted.json','standing_immutable.sha256.json','jobs/standing.json','logs/standing.log')}
        for phase in h.PHASES[1:]:
            args.phase_group='admission' if phase in h.ADMISSION_PHASES else 'learn'
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
            self.assertEqual('--decision-receipt' in c,phase=='train_10')
            if phase=='train_10':self.assertIn('/private/test/decision.json:/review/decision.json:ro',mounts)
            if args.phase_group=='learn':
                for name in args.prior_file_hashes:
                    self.assertIn('/private/test/output/'+name+':/outputs/'+name+':ro',mounts)
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

    def exercise(self,fail=None,mutation=None,wrong_checkpoint=False,wrong_mode=False,wrong_raw=False,identity_change=False,group='admission'):
        with tempfile.TemporaryDirectory() as d:
            args=self.args(Path(d));args.phase_group=group;args.output.mkdir();calls=[];identity={'bound':True}
            for phase in h.PHASES:(args.output/phase).mkdir()
            (args.output/'standing/admission.json').write_text('{}')
            (args.output/'calibration/initial.pt').write_bytes(b'exact initial checkpoint')
            (args.output/'train_10/decision_010.pt').write_bytes(b'exact decision checkpoint')
            args.completed_prior_phases={p:h.tree_hashes(args.output/p) for p in h.ADMISSION_PHASES} if group=='learn' else {}
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
                return {'bound':False} if identity_change and len(calls)>=2 else identity
            consumer=SimpleNamespace(verify_standing=lambda *a:None,validate_result=validate)
            with patch.object(h,'validate_inputs',side_effect=inputs),patch.object(h,'run_owned',side_effect=run),patch.object(h,'PARENT',SimpleNamespace(check_source=lambda p:None)),patch.object(h,'CONSUMER',consumer):
                if fail or mutation or wrong_checkpoint or wrong_mode or wrong_raw or identity_change:
                    with self.assertRaises(ValueError):h.run_phases(args,identity,{})
                else:
                    report={};h.run_phases(args,identity,report)
                    self.assertEqual(report.get('PPO_updates_completed',0),10 if group=='learn' else 0)
            return calls

    def test_admission_stops_after_three_phases(self):self.assertEqual(self.exercise(),list(h.ADMISSION_PHASES))
    def test_learning_runs_only_four_reviewed_phases(self):self.assertEqual(self.exercise(group='learn'),list(h.LEARNING_PHASES))
    def test_failed_calibration_prevents_recovery(self):self.assertEqual(self.exercise(fail='calibration'),list(h.PHASES[:2]))
    def test_failed_recovery_remains_rejected(self):self.assertEqual(self.exercise(fail='learning_recovery_32'),list(h.ADMISSION_PHASES))
    def test_failed_training_prevents_screen(self):self.assertEqual(self.exercise(fail='train_10',group='learn'),['train_10'])
    def test_rejected_initial_prevents_next_screen(self):self.assertEqual(self.exercise(fail='evaluate_initial',group='learn'),list(h.LEARNING_PHASES[:2]))
    def test_completed_checkpoint_mutation_caught_immediately(self):
        self.assertEqual(self.exercise(mutation=('learning_recovery_32','calibration/initial.pt')),list(h.ADMISSION_PHASES))
    def test_standing_mutation_caught_immediately(self):
        self.assertEqual(self.exercise(mutation=('calibration','standing/admission.json')),list(h.PHASES[:2]))
    def test_final_screen_cannot_mutate_prior_result(self):
        self.assertEqual(self.exercise(mutation=('quiet_010','evaluate_010/added.bin'),group='learn'),list(h.LEARNING_PHASES))
    def test_checkpoint_phase_binding(self):self.assertEqual(self.exercise(wrong_checkpoint=True),list(h.ADMISSION_PHASES))
    def test_cli_mode_mapping_binding(self):self.assertEqual(self.exercise(wrong_mode=True),list(h.PHASES[:2]))
    def test_raw_receipt_binding(self):self.assertEqual(self.exercise(wrong_raw=True),list(h.PHASES[:2]))
    def test_changed_source_stops_before_recovery(self):self.assertEqual(self.exercise(identity_change=True),list(h.PHASES[:2]))
    def test_cross_group_commands_rejected(self):
        args=self.args(Path('/test'))
        with self.assertRaises(ValueError):h.command(args,'name','train_10')
        args.phase_group='learn'
        with self.assertRaises(ValueError):h.command(args,'name','standing')
    def test_report_paths_separate(self):
        args=self.args(Path('/test'));self.assertEqual(h.report_path(args),args.output/'campaign.json')
        args.phase_group='learn';self.assertEqual(h.report_path(args),args.output/'learning_campaign.json')

    def continuation(self,corrupt=None):
        with tempfile.TemporaryDirectory() as directory:
            args=self.args(Path(directory));args.phase_group='learn';args.output.mkdir()
            args.host_freeze_sha256='exact_host';baseline={'bound':'source'};identity={'bound':'standing'}
            accepted={}
            for phase in h.ADMISSION_PHASES:
                folder=args.output/phase;folder.mkdir();h.save(folder/'state.json',{'phase':phase})
                if phase=='standing':h.save(folder/'admission.json',{'passed':True})
                files=h.tree_hashes(folder)
                accepted[phase]={'passed':True,'phase':phase,'files_sha256':files}
                h.save(args.output/(phase+'_accepted.json'),accepted[phase])
                h.save(args.output/(phase+'_immutable.sha256.json'),files)
            (args.output/'inputs/study').mkdir(parents=True)
            (args.output/'inputs/study/asset.bin').write_bytes(b'exact study')
            assets=h.tree_hashes(args.output/'inputs/study');h.save(args.output/'inputs/study_before.sha256.json',assets)
            prior={'status':'completed','learning_admission_only_complete':True,'phase_group':'admission',
                   'PPO_updates_completed':0,'consumer_freeze_sha256':h.CONSUMER_FREEZE,
                   'host_freeze_sha256':args.host_freeze_sha256,'baseline':baseline,
                   'source_and_inputs_unchanged':True,'accepted_phases':accepted}
            h.save(args.output/'campaign.json',prior);h.save(args.decision_receipt,{'accepted':True})
            if corrupt=='source':prior['consumer_freeze_sha256']='different';h.save(args.output/'campaign.json',prior)
            if corrupt=='failed':prior['status']='failed';h.save(args.output/'campaign.json',prior)
            if corrupt=='phase':(args.output/'calibration/state.json').write_text('changed')
            if corrupt=='assets':(args.output/'inputs/study/asset.bin').write_bytes(b'changed')
            if corrupt=='existing':(args.output/'train_10').mkdir()
            if corrupt=='receipt':h.save(args.decision_receipt,{'accepted':False})
            if corrupt=='internal_receipt':args.decision_receipt=args.output/'decision.json';h.save(args.decision_receipt,{'accepted':True})
            if corrupt=='accepted':h.save(args.output/'calibration_accepted.json',{'passed':False})
            def decision(a,i):
                self.assertEqual(i,identity)
                if h.read(a.decision_receipt).get('accepted') is not True:raise ValueError('No reviewed allocation')
                return {'receipt_sha256':h.sha(a.decision_receipt),'approved_updates':10}
            consumer=SimpleNamespace(verify_standing=lambda *a:None,validate_result=lambda path,i:accepted[path.name],validate_learning_decision=decision)
            before=h.tree_hashes(args.output)
            with patch.object(h,'validate_inputs',return_value=identity),patch.object(h,'CONSUMER',consumer):
                if corrupt:
                    with self.assertRaises(ValueError):h.admit_existing_campaign(args,baseline)
                else:
                    returned,review=h.admit_existing_campaign(args,baseline)
                    self.assertEqual(returned,assets);self.assertEqual(review['approved_updates'],10)
                    self.assertEqual(args.prior_file_hashes,before)
                    self.assertEqual(set(args.completed_prior_phases),set(h.ADMISSION_PHASES))
                    self.assertEqual(h.tree_hashes(args.output),before)
                    h.verify_prior_files(args)
                    prior_receipt=args.decision_receipt.read_bytes()
                    args.decision_receipt.write_bytes(b'changed decision')
                    with self.assertRaises(ValueError):h.verify_prior_files(args)
                    args.decision_receipt.write_bytes(prior_receipt)
                    (args.output/'campaign.json').write_text('changed previous report')
                    with self.assertRaises(ValueError):h.verify_prior_files(args)

    def test_exact_reviewed_continuation_preserves_admission(self):self.continuation()
    def test_continuation_rejects_changed_source(self):self.continuation('source')
    def test_continuation_rejects_failed_admission(self):self.continuation('failed')
    def test_continuation_rejects_changed_phase(self):self.continuation('phase')
    def test_continuation_rejects_changed_assets(self):self.continuation('assets')
    def test_continuation_never_overwrites_attempt(self):self.continuation('existing')
    def test_continuation_requires_accepted_external_review(self):self.continuation('receipt')
    def test_continuation_rejects_writable_receipt_alias(self):self.continuation('internal_receipt')
    def test_continuation_rejects_rewritten_acceptance(self):self.continuation('accepted')

    def test_owned_supervision_parity_except_explicit_deadlines(self):
        p=Path(os.environ['MOVING_HOST_PARENT_SOURCE']) if 'MOVING_HOST_PARENT_SOURCE' in os.environ else next(root/'tmp/reference_moving_ppo_launch_001/launch_moving_ppo_spark.py' for root in Path(__file__).resolve().parents if (root/'tmp/reference_moving_ppo_launch_001/launch_moving_ppo_spark.py').is_file())
        old=ast.parse(p.read_text());new=ast.parse(Path(h.__file__).read_text())
        find=lambda tree,name:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        self.assertEqual(ast.dump(find(old,'owned_container')),ast.dump(find(new,'owned_container')))
        source=ast.get_source_segment(p.read_text(),find(old,'run_owned'))
        source=source.replace('deadline_seconds=600,','deadline_seconds=PHASE_SECONDS[phase],').replace('deadline = time.monotonic() + 600','deadline = time.monotonic() + PHASE_SECONDS[phase]').replace('raise TimeoutError("Standing phase exceeded ten-minute bound")','raise TimeoutError(f"{phase} exceeded declared {PHASE_SECONDS[phase]} second bound")')
        self.assertEqual(ast.dump(ast.parse(source).body[0]),ast.dump(find(new,'run_owned')))

if __name__=='__main__':unittest.main()
