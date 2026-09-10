"""Stdlib-only allocation contract; numerical proof remains a separate audit."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from campaign_contract import validate_learning_decision,validate_learning_integrity,sha,save

class Tests(unittest.TestCase):
    def test_no_automatic_allocation_and_exact_three_phase_binding(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);campaign=base/'campaign';identity={'exact_source':'synthetic fixture'}
            for name in ('standing','calibration','learning_recovery_32'):
                folder=campaign/name;folder.mkdir(parents=True)
                save(folder/'state.json',{'status':'completed','identity':identity,'phase':name})
            save(campaign/'learning_recovery_32/learning_integrity.json',{'synthetic_contract_fixture':True})
            record=dict(schema='moving_PPO_learning_admission_review_v1',identity=identity,accepted=True,approved_updates=10,
                phase_state_sha256={name:sha(campaign/name/'state.json') for name in ('standing','calibration','learning_recovery_32')},
                learning_integrity_sha256=sha(campaign/'learning_recovery_32/learning_integrity.json'),
                event_and_recovery_review=True,per_environment_progress_review=True,timing_and_memory_review=True,
                raw_evidence_integrity_review=True,no_gate_relaxation=True)
            args=SimpleNamespace(output=campaign/'train_10',decision_receipt=base/'review.json')
            save(args.decision_receipt,record)
            # Prior full phase validators are independently tested. Here isolate
            # the new review's routing/immutability and allocation requirements.
            with patch('campaign_contract.verify_standing') as standing,patch('campaign_contract.validate_result') as prior:
                result=validate_learning_decision(args,identity)
                self.assertEqual(result['approved_updates'],10);self.assertEqual(prior.call_count,2)
                standing.assert_called_once_with((campaign/'standing/admission.json').resolve(),campaign.resolve())
                for key,value in [('approved_updates',25),('accepted',False),('event_and_recovery_review',False),
                    ('timing_and_memory_review',False),('raw_evidence_integrity_review',False),('no_gate_relaxation',False),
                    ('learning_integrity_sha256','f'*64),('phase_state_sha256',{})]:
                    changed=copy.deepcopy(record);changed[key]=value;save(args.decision_receipt,changed)
                    with self.subTest(key=key),self.assertRaises(ValueError):validate_learning_decision(args,identity)
                args.decision_receipt=None
                with self.assertRaises(ValueError):validate_learning_decision(args,identity)
                args.decision_receipt=campaign/'self_approval.json';save(args.decision_receipt,record)
                with self.assertRaises(ValueError):validate_learning_decision(args,identity)

    def test_actual_evidence_counts_hashes_and_no_recovery_reject(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d);folder=output/'raw/learning_ledger';folder.mkdir(parents=True)
            (folder/'00000_reset.npz').write_bytes(b'synthetic contract fixture; numeric test uses actual exported NPZ')
            episodes=[1]+[0]*31
            ledger=dict(controls=712,replicas=32,completed_finite_recovery_rows=1,finite_reset_events=1,
                bootstrap_event_records=1,all_records_verified=True,session_failure=None,error=None,
                per_environment_episodes=episodes,files_sha256={'00000_reset.npz':sha(folder/'00000_reset.npz')})
            save(folder/'summary.json',ledger)
            identity={'synthetic':True}
            record=dict(schema='moving_learning_recovery_admission_v1',identity=identity,infrastructure_passed=True,
                physical_admission=False,automatic_training_allowed=False,ledger=ledger,ledger_sha256=sha(folder/'summary.json'))
            save(output/'learning_integrity.json',record)
            per=[dict(row=i,active_controls=312 if i==0 else 512,excluded_recovery_controls=200 if i==0 else 0,
                finite_terminal_transitions=episodes[i],time_limit_transitions=0,moving_intervals_without_reset_crossing=0,
                measured_forward_increment_sum_m=0.,raw_SDK_mean_forward_mps=0.) for i in range(32)]
            state=dict(identity=identity,learning_recovery_passed=True,learning_integrity_sha256=sha(output/'learning_integrity.json'),
                recovery_controls=512,recovery_control_seconds=[.01]*512,recovery_wall_s=5.3,
                recovery_per_environment=per,active_transitions=32*512-200)
            self.assertEqual(validate_learning_integrity(output,state),record)
            for key,value in [('learning_recovery_passed',False),('active_transitions',32*512),('recovery_controls',511),('learning_integrity_sha256','x')]:
                bad=copy.deepcopy(state);bad[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):validate_learning_integrity(output,bad)
            changed=copy.deepcopy(ledger);changed['completed_finite_recovery_rows']=0;save(folder/'summary.json',changed)
            with self.assertRaises(ValueError):validate_learning_integrity(output,state)

if __name__=='__main__':unittest.main()
