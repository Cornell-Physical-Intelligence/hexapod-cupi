import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import analyze,fetch,remote_inventory

ROOT=Path(__file__).resolve().parents[2]
HIST=ROOT/'tmp/direct_omni_recovery_001/baseline/inputs/old_diagnostics.json'
RAW=ROOT/'tmp/direct_omni_cold_results_001/run'

class AuditTests(unittest.TestCase):
    def test_real_complete_comparison_preserves_profiles(self):
        r=analyze.analyze(RAW,HIST)
        self.assertEqual(r['cold_formal004']['terminations_all48'],0)
        self.assertEqual(len(r['cold_formal004']['scenarios']),12)
        self.assertEqual(r['historical_diagnostic003']['target_slew_rad_per_20ms'],.03)
        self.assertEqual(r['cold_formal004']['target_slew_rad_per_20ms'],.04)
        self.assertFalse(r['training_admission']);self.assertFalse(r['stop_transition_measured'])
        self.assertEqual(r['trace_subset']['saved_replicas'],12)
        self.assertGreater(r['cold_formal004']['scenarios'][0]['windows']['post_settle']['torque_saturation_fraction'],.1)

    def test_wrong_historical_profile_rejected(self):
        with self.assertRaisesRegex(ValueError,'Profile label'):
            analyze.diagnostic_summary(HIST,.04)

    def test_missing_scenario_rejected(self):
        d=analyze.read(HIST);d['scenarios'].pop()
        with patch.object(analyze,'read',return_value=d):
            with self.assertRaisesRegex(ValueError,'Scenario'):
                analyze.diagnostic_summary(HIST,.03)

    def test_nonfinite_metric_rejected(self):
        d=analyze.read(HIST);d['scenarios'][0]['windows']['all']['planar_error_mps']=float('nan')
        with patch.object(analyze,'read',return_value=d):
            with self.assertRaisesRegex(ValueError,'Nonfinite'):
                analyze.diagnostic_summary(HIST,.03)

    def test_path_substitution_rejected(self):
        for s in ('../trace.npz','/trace.npz','x/../../q','x//q','x;rm','x q'):
            with self.assertRaises(ValueError):fetch.safe_relative(s)
        self.assertEqual(str(fetch.safe_relative('baseline/diagnostic_trace.npz')),'baseline/diagnostic_trace.npz')

    def test_terminal_and_identity_guard(self):
        with tempfile.TemporaryDirectory() as t:
            run=Path(t);(run/'jobs').mkdir()
            campaign=analyze.read(RAW/'campaign.json')
            (run/'campaign.json').write_text(json.dumps(campaign))
            (run/'jobs/standing.json').write_bytes((RAW/'jobs/standing.json').read_bytes())
            with patch.object(remote_inventory,'RUN',run):
                self.assertEqual(remote_inventory.inventory(run)['campaign_status'],'completed')
                campaign['status']='running';(run/'campaign.json').write_text(json.dumps(campaign))
                with self.assertRaisesRegex(ValueError,'terminal'):remote_inventory.inventory(run)
                campaign['status']='completed';campaign['identity']['checkpoint_sha256']='bad'
                (run/'campaign.json').write_text(json.dumps(campaign))
                with self.assertRaisesRegex(ValueError,'checkpoint'):remote_inventory.inventory(run)

    def test_trace_time_shift_rejected(self):
        import numpy as np
        original=np.load
        class Changed:
            def __init__(self,p):
                with original(p,allow_pickle=False) as f:self.data={k:f[k] for k in f.files}
                self.data['time_s']+=.02;self.files=list(self.data)
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def __getitem__(self,k):return self.data[k]
        d=analyze.diagnostic_summary(RAW/'baseline/diagnostics.json',.04)
        with patch.object(np,'load',side_effect=lambda p,**kw:Changed(p)):
            with self.assertRaisesRegex(ValueError,'time trace'):
                analyze.trace_summary(RAW/'baseline/diagnostic_trace.npz',d)

if __name__=='__main__':unittest.main()
