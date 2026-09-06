"""CPU tests: exact full-phase selection, PID ownership and at-most-once dispatch."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import follow_campaign as follower


class FollowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()/'campaign'
        self.run = self.root/'full'/'hexapod-fourbar-train-full'
        self.run.mkdir(parents=True)
        self.report = self.run/'report.json'
        self.checkpoint = self.run/'checkpoint.pt'
        self.checkpoint.write_bytes(b'full checkpoint fixture')
        self.admission = self.root/'admission.json'
        self.admission.write_text('{"pass":true}\n')
        self.training = {'num_envs':512,'iterations_requested':1000,'iterations_completed':1000,'paused':False}
        self.report.write_text(json.dumps(self.training))
        self.contract = {'sha256':'a'*64,'files':{}}
        self.supervisor = {'supervisor_exit_code':0,'execution':'finished','validator_report_status':'passed',
            'cleanup':'removed_exact_id','source_identity_unchanged_at_finish':True,'contract':self.contract}
        (self.run/'supervisor.json').write_text(json.dumps(self.supervisor))
        self.common = SimpleNamespace(digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(),
                                     read_json=lambda p:json.loads(Path(p).read_text()))
        self.campaign = {'state':'complete','pass':True,'errors':[],'separate_process_resume_verified':True,
            'contract':self.contract,'final_checkpoint':str(self.checkpoint),
            'admission':{'path':str(self.admission),'sha256':self.common.digest(self.admission),'pass':True},
            'phases':[{'name':'scratch','state':'passed'}, {'name':'full','state':'passed','launcher_exit_code':0,
                'requested':{'name':'full','mode':'train','num_envs':512,'iterations':1000},
                'report':str(self.report),'report_sha256':self.common.digest(self.report)}]}

    def select(self):
        return follower.select_full_capture(self.campaign,self.root/'campaign.json',self.common)

    def test_full_checkpoint_selected_with_own_admission(self):
        selected = self.select()
        self.assertEqual(Path(selected['checkpoint']),self.checkpoint)
        self.assertEqual(Path(selected['training_report']),self.report)
        self.assertEqual(Path(selected['admission']),self.admission)

    def test_failed_paused_incomplete_campaign_never_selects(self):
        for state in ('failed','paused','running'):
            with self.subTest(state=state):
                self.campaign['state']=state
                with self.assertRaises(ValueError):self.select()

    def test_scratch_and_short_full_training_never_selected(self):
        baseline=copy.deepcopy(self.campaign)
        for change in ('scratch_only','requested_three','completed_three','paused','duplicate_full'):
            self.campaign=copy.deepcopy(baseline)
            training=dict(self.training)
            if change=='scratch_only':self.campaign['phases']=self.campaign['phases'][:1]
            elif change=='requested_three':self.campaign['phases'][1]['requested']['iterations']=3
            elif change=='completed_three':training['iterations_completed']=3
            elif change=='paused':training['paused']=True
            else:self.campaign['phases'].append(copy.deepcopy(self.campaign['phases'][1]))
            self.report.write_text(json.dumps(training))
            if len(self.campaign['phases'])>1:self.campaign['phases'][1]['report_sha256']=self.common.digest(self.report)
            with self.subTest(change=change),self.assertRaises(ValueError):self.select()

    def test_checkpoint_escape_and_symlink_escape_rejected(self):
        foreign=self.root.parent/'checkpoint.pt'
        foreign.write_bytes(b'foreign')
        self.campaign['final_checkpoint']=str(foreign)
        with self.assertRaisesRegex(ValueError,'escapes'):self.select()
        self.checkpoint.unlink()
        self.checkpoint.symlink_to(foreign)
        self.campaign['final_checkpoint']=str(self.checkpoint)
        with self.assertRaisesRegex(ValueError,'escapes'):self.select()

    def test_report_hash_and_failed_supervisor_rejected(self):
        self.report.write_text(json.dumps({**self.training,'tampered':True}))
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.select()
        self.campaign['phases'][1]['report_sha256']=self.common.digest(self.report)
        (self.run/'supervisor.json').write_text(json.dumps({**self.supervisor,'supervisor_exit_code':1}))
        with self.assertRaisesRegex(ValueError,'supervisor'):self.select()

    def test_fsynced_reservation_blocks_restart_duplicate(self):
        path=self.root/'follower_state.json'
        state={'state':'waiting'}
        events=[]
        def save():
            follower.durable_json(path,state)
            events.append(json.loads(path.read_text()))
        argv=['/usr/bin/python3','capture_policy.py','--checkpoint',str(self.checkpoint)]
        follower.reserve_launch(state,save,argv)
        self.assertEqual(events[0]['state'],'launch_reserved')
        self.assertEqual(events[0]['capture_argv'],argv)
        # Simulates restart in the narrow Popen/PID-write crash window.
        restarted=json.loads(path.read_text())
        with self.assertRaisesRegex(ValueError,'duplicate'):
            follower.reserve_launch(restarted,save,argv)
        for phase in ('capture_running','video_complete','video_failed'):
            restarted['state']=phase
            with self.assertRaises(ValueError):follower.reserve_launch(restarted,save,argv)
        self.assertEqual(len(events),1)

    def test_new_dispatch_rejects_preexisting_and_dangling_output(self):
        output=self.root/'capture_outputs'
        follower.require_unused_capture_output(output)
        output.mkdir()
        with self.assertRaisesRegex(ValueError,'earlier evidence'):
            follower.require_unused_capture_output(output)
        output.rmdir()
        output.symlink_to(self.root/'missing')
        with self.assertRaisesRegex(ValueError,'earlier evidence'):
            follower.require_unused_capture_output(output)

    def capture_fixture(self):
        folder=self.root/'follower'
        output=folder/'capture_outputs'/'hexapod-policy-capture-fixture'
        output.mkdir(parents=True)
        tools={'capture_common.py':'b'*64,'record_admitted_policy.py':'c'*64,'capture_policy.py':'d'*64}
        inputs={'checkpoint':'e'*64}
        state={'binding':{'capture_tools':str(self.root/'tools'),'capture_tools_sha256':tools,
                         'source':str(self.root/'source'),'source_sha256':self.contract['sha256']},
               'input_sha256':inputs,'selected_inputs':{'checkpoint':str(self.checkpoint),
                         'admission':str(self.admission),'training_report':str(self.report)}}
        verified={'contract':self.contract,'input_sha256':inputs}
        supervisor={'pass':True,'supervisor_exit_code':0,'execution':'finished','cleanup':'removed_exact_id',
                    'decoded_video_verified':True,'source_and_inputs_unchanged':True,
                    'contract':self.contract,'input_sha256':inputs,'capture_tools_sha256':tools}
        (output/'supervisor.json').write_text(json.dumps(supervisor))
        result={'capture_tools_sha256':tools}
        calls=[]
        common=SimpleNamespace(read_json=self.common.read_json,
            tool_identity=lambda path:tools,verify_inputs=lambda *args:verified,
            validate_capture_report=lambda *args:calls.append(args) or result)
        return folder,output,state,common,verified,supervisor,result,calls

    def test_completed_video_requires_exact_tool_source_and_input_lineage(self):
        folder,output,state,common,verified,supervisor,result,calls=self.capture_fixture()
        self.assertEqual(follower.verify_completed_capture(folder,state,common),output/'supervisor.json')
        self.assertEqual(calls,[(output/'report.json',verified,750)])
        baseline=copy.deepcopy(supervisor)
        changes=({'capture_tools_sha256':{'wrong':'tool'}}, {'source_and_inputs_unchanged':False},
                 {'decoded_video_verified':False}, {'contract':{'sha256':'f'*64}},
                 {'input_sha256':{'checkpoint':'0'*64}}, {'supervisor_exit_code':1}, {'cleanup':'FAILED'})
        for change in changes:
            (output/'supervisor.json').write_text(json.dumps({**baseline,**change}))
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'exact bound'):
                follower.verify_completed_capture(folder,state,common)

    def test_changed_tools_and_final_report_cannot_reuse_successful_supervisor(self):
        folder,output,state,common,verified,supervisor,result,calls=self.capture_fixture()
        with patch.object(common,'tool_identity',return_value={'changed':'tools'}):
            with self.assertRaisesRegex(ValueError,'tools changed'):
                follower.verify_completed_capture(folder,state,common)
        result['capture_tools_sha256']={'changed':'recorder'}
        with self.assertRaisesRegex(ValueError,'different capture tools'):
            follower.verify_completed_capture(folder,state,common)
        result['capture_tools_sha256']=state['binding']['capture_tools_sha256']
        verified['input_sha256']={'changed':'checkpoint'}
        with self.assertRaisesRegex(ValueError,'exact bound'):
            follower.verify_completed_capture(folder,state,common)

    def test_campaign_pid_flags_and_start_ticks(self):
        source=self.root.parent/'source'
        binding={'campaign_pid':42,'campaign_start_ticks':555,'source':str(source),
                 'campaign_file':str(self.root/'campaign.json'),'source_commit':'c'*40}
        identity={'pid':42,'start_ticks':555,'argv':['python3',str(source/'isaaclab/deploy/run-mkii-fourbar-campaign'),
            '--source-dir',str(source),'--source-commit','c'*40,'--output-root',str(self.root.parent)]}
        follower.verify_campaign_process(identity,binding)
        with self.assertRaisesRegex(ValueError,'PID was reused'):
            follower.verify_campaign_process({**identity,'start_ticks':556},binding)
        bad={**identity,'argv':identity['argv']+['--source-dir','/foreign']}
        with self.assertRaises(ValueError):follower.verify_campaign_process(bad,binding)

    def test_signal_uses_pidfd_and_never_reused_pid(self):
        expected={'pid':42,'start_ticks':555,'argv':['capture'],'boot_id':'same-boot'}
        with patch.object(follower,'process_identity',return_value={**expected,'start_ticks':556}), \
                patch.object(follower.os,'pidfd_open',create=True) as opener:
            self.assertFalse(follower.forward_owned_signal(expected,15))
            opener.assert_not_called()
        with patch.object(follower,'process_identity',return_value=expected), \
                patch.object(follower.os,'pidfd_open',return_value=91,create=True), \
                patch.object(follower.signal,'pidfd_send_signal',create=True) as send, \
                patch.object(follower.os,'close') as close:
            self.assertTrue(follower.forward_owned_signal(expected,15))
            send.assert_called_once_with(91,15)
            close.assert_called_once_with(91)

    def test_signal_does_not_fail_when_owned_process_exits_during_pidfd_open(self):
        expected={'pid':42,'start_ticks':555,'argv':['capture'],'boot_id':'same-boot'}
        with patch.object(follower,'process_identity',return_value=expected), \
                patch.object(follower.os,'pidfd_open',side_effect=ProcessLookupError,create=True), \
                patch.object(follower.signal,'pidfd_send_signal',create=True) as send:
            self.assertFalse(follower.forward_owned_signal(expected,15))
            send.assert_not_called()


if __name__=='__main__':unittest.main()
