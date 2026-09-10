import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from campaign_contract import sha,save
from run_moving_ppo import accepted_calibration,require_profile,require_continuation


class Tests(unittest.TestCase):
    def test_failed_or_wrong_profile_cannot_allocate_updates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'profile_32';folder.mkdir()
            args=SimpleNamespace(output=root/'train_10');identity={'test':'source'}
            state={'status':'completed','identity':identity,'profile_passed':True,'profile_controls':512,'replicas':32}
            save(folder/'state.json',state);self.assertEqual(require_profile(args,identity,32),sha(folder/'state.json'))
            for key,value in [('profile_passed',False),('profile_controls',511),('replicas',128),('status','rejected')]:
                bad=dict(state);bad[key]=value;save(folder/'state.json',bad)
                with self.assertRaises(ValueError):require_profile(args,identity,32)

    def test_no_automatic25_or_regression_waiver(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args=SimpleNamespace(output=root/'train_25',decision_receipt=root/'review.json');identity={'test':'source'}
            states={}
            for name in ('train_10','evaluate_initial','evaluate_010','quiet_010'):
                (root/name).mkdir();state={'status':'completed','identity':identity,'retention_passed':True,'quiet_passed':True}
                save(root/name/'state.json',state);states[name]=sha(root/name/'state.json')
            receipt={'schema':'moving_PPO_10_to25_review_v1','identity':identity,'accepted':True,
                'phase_state_sha256':states,'measured_improvement_review':True,'no_direction_or_quiet_regression_review':True}
            save(args.decision_receipt,receipt);self.assertEqual(require_continuation(args,identity),sha(args.decision_receipt))
            for key,value in [('accepted',False),('measured_improvement_review',False),('no_direction_or_quiet_regression_review',False)]:
                bad=copy.deepcopy(receipt);bad[key]=value;save(args.decision_receipt,bad)
                with self.assertRaises(ValueError):require_continuation(args,identity)
            save(args.decision_receipt,receipt)
            state=json.loads((root/'quiet_010/state.json').read_text());state['quiet_passed']=False
            save(root/'quiet_010/state.json',state);receipt['phase_state_sha256']['quiet_010']=sha(root/'quiet_010/state.json')
            save(args.decision_receipt,receipt)
            with self.assertRaises(ValueError):require_continuation(args,identity)

if __name__=='__main__':unittest.main()
