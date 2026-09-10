import copy, importlib.util, json, sys, tempfile, unittest
from pathlib import Path
import numpy as np
from analyze import Inputs, constant, compare_constant, stop, compare_stop, training, parse_timings, sha, human

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
COLD=ROOT/'tmp/direct_omni_cold_results_001/run/baseline'
NATIVE=ROOT/'tmp/direct_omni_recovery_001/native_003'

class AnalysisTests(unittest.TestCase):
    def test_actual_cold_self_pair_and_one_direction_regression(self):
        inputs=Inputs();before=constant(inputs,COLD);after=copy.deepcopy(before)
        rows=compare_constant(before,after)
        self.assertEqual(len(rows),12);self.assertTrue(all(all(x==0 for x in r['delta_after_minus_before'].values()) for r in rows))
        self.assertEqual(before['cases'][0]['aggregate_replicas'],4)
        self.assertEqual(before['cases'][0]['sampled_replica_rates']['traced_env_id'],0)
        after['cases'][3]['metrics']['planar_error_mps']+=.03
        result=compare_constant(before,after)
        self.assertFalse(result[3]['directional_observations']['planar_error_lower'])
        self.assertGreater(result[3]['delta_after_minus_before']['planar_error_mps'],0)
        after['overrides']['target_slew_rad_per_20ms']=.03
        with self.assertRaisesRegex(ValueError,'confound'):compare_constant(before,after)
        inputs.unchanged()

    def test_full_native_stop_replay_all_replicas_and_corrupt_verdict(self):
        sys.path.insert(0,str(NATIVE))
        try:
            spec=importlib.util.spec_from_file_location('_native_wiring_fixture',NATIVE/'test_runtime_wiring.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            import torch
            from types import SimpleNamespace
            from direct_stop_evaluation import evaluate_stop
            from direct_contract import EXPECTED_OVERRIDES
            with tempfile.TemporaryDirectory() as d:
                p=Path(d);env=module.FakeStopEnv();runner=SimpleNamespace(get_inference_policy=lambda device:lambda obs:torch.zeros(48,18))
                evaluate_stop(env,runner,{'omni':{'overrides':EXPECTED_OVERRIDES}},p,'a'*64)
                actual=stop(Inputs(),p)
                self.assertEqual(len(actual['cases']),48);self.assertEqual(actual['quiet_passed_replicas'],47)
                self.assertFalse(actual['cases'][0]['quiet']['pass']);self.assertEqual(actual['cases'][0]['quiet']['terminations'],1)
                comparison=compare_stop(actual,copy.deepcopy(actual));self.assertTrue(all(all(v==0 for v in r['delta_after_minus_before'].values()) for r in comparison))
                report=json.loads((p/'stop_diagnostics.json').read_text());report['scenarios'][0]['replicas'][0]['quiet']['pass']=True
                (p/'stop_diagnostics.json').write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError,'verdict'):stop(Inputs(),p)
        finally:sys.path.pop(0)

    def test_training_raw_events_update_counts_and_hidden_noisy_replica(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);n=32;t=48
            raw={k:np.zeros((t,n)) for k in ['requested_torque_abs_max_nm','applied_torque_abs_max_nm','requested_saturation_fraction','reported_joint_velocity_rms_rad_s','target_delta_rms_rad']}
            for key in ['terminated','truncated','nonfoot']:raw[key]=np.zeros((t,n),bool)
            raw['command']=np.zeros((t,n,3));raw['command'][20:,:,0]=.1;raw['time_s']=(np.arange(t)+1)*.02
            raw['terminated'][11,31]=True;raw['nonfoot'][11,31]=True;raw['requested_torque_abs_max_nm'][11,31]=7.
            raw['applied_torque_abs_max_nm'][:]=1.6
            np.savez_compressed(p/'training_trace.npz',**raw);np.savez_compressed(p/'training_joint_trace.npz',joint_position_rad=np.zeros((t,8,18)))
            event={'control':12,'ids':[31],'fields':{key:[raw[key][11,31].tolist()] for key in ['command','terminated','truncated','requested_torque_abs_max_nm','nonfoot']}}
            (p/'training_events.json').write_text(json.dumps([event]))
            audit={'replicas':n,'controls':t,'terminations_per_row':raw['terminated'].sum(0).tolist(),'truncations_per_row':raw['truncated'].sum(0).tolist(),
                   'requested_torque_max_per_row_nm':raw['requested_torque_abs_max_nm'].max(0).tolist(),'applied_torque_max_per_row_nm':raw['applied_torque_abs_max_nm'].max(0).tolist(),
                   'requested_saturation_fraction_per_row':raw['requested_saturation_fraction'].mean(0).tolist(),
                   'trace_sha256':sha(p/'training_trace.npz'),'joint_trace_sha256':sha(p/'training_joint_trace.npz'),'event_ledger_sha256':sha(p/'training_events.json')}
            receipt={'audit':audit,'complete':False,'updates_completed':2,'optimizer_updates':[{'completed_update':i} for i in [1,2]],'wall_seconds':2.,'selection':{'allocation':'smoke'}}
            (p/'training_receipt.json').write_text(json.dumps(receipt));(p/'state.json').write_text('{}');(p/'repair_initialization.json').write_text('{}')
            result=training(Inputs(),p);self.assertEqual(result['per_row'][31]['terminations'],1);self.assertEqual(result['per_row'][31]['requested_peak_nm'],7.)
            self.assertEqual(result['updates_completed'],2);self.assertEqual(result['transitions'],1536)
            event['ids']=[30];(p/'training_events.json').write_text(json.dumps([event]));audit['event_ledger_sha256']=sha(p/'training_events.json');(p/'training_receipt.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'event'):training(Inputs(),p)

    def test_immutable_inputs_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json';p.write_text('{}');inputs=Inputs();inputs.json(p);p.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'changed'):inputs.unchanged()

    def test_actual_rsl_timing_format_without_loss_promotion(self):
        text='\x1b[1mIteration\x1b[0m\nCollection time: 1.235s\nLearning time: 0.72s\nSteps per second: 34567\nMean value loss: 0.001'
        value=parse_timings(text)
        self.assertEqual(value,{'collection_seconds':[1.235],'learning_seconds':[.72],'reported_steps_per_second':[34567.]})

    def test_actual_failed_smoke_receipt_formats_missing_reload_without_admission(self):
        # Exact failed native002 run is a formatter regression fixture only.
        # The successor's top-level campaign admission still requires source002.
        source=ROOT/'tmp/direct_omni_train_smoke_results_001/run/train'
        inputs=Inputs();result=training(inputs,source)
        self.assertEqual(result['updates_completed'],2)
        self.assertEqual(result['transitions'],1536)
        self.assertFalse(result['complete']);self.assertIsNone(result['strict_reload'])
        report={'conclusion':'Incomplete campaign; no continuation admission.',
                'training':result,'errors':['Strict final reload failed; no final evaluations.'],
                'next_decision':'Correct finalizer in a separate source; never admit this failed run.'}
        text=human(report)
        self.assertIn('Strict reload reported False',text)
        self.assertIn('2 updates',text)
        self.assertIn('no continuation admission',text)
        self.assertIn('Strict final reload failed',text)
        inputs.unchanged()

if __name__=='__main__':unittest.main()
