from pathlib import Path
import copy,json,sys,unittest
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'reference_physics_adapter_006/source_006/tools')]
from solver_comparison import validate_iteration_readback
ACTUAL=json.loads((HERE/'inputs/actual006_failed_readback.json').read_text())['actual_iteration_settings']['rows']

class LinkIterationTests(unittest.TestCase):
    def test_actual006_snapshot_has_valid32roots_and608ownedlinks(self):
        rows=copy.deepcopy(ACTUAL);validate_iteration_readback(rows,32)
        roots=[r for r in rows if r['kind']=='articulation'];bodies=[r for r in rows if r['kind']=='body']
        self.assertEqual(len(roots),32);self.assertEqual(len(bodies),608)
        self.assertEqual(sum(bool(r['absent_unauthored_link_attributes']) for r in bodies),576)
        self.assertTrue(all(r['position']==16 and r['velocity']==1 for r in roots))
        self.assertTrue(all(r['articulation_owner_path'] for r in bodies))

    def test_wrong_or_missing_actual_articulation_root_still_rejects(self):
        for wrong in ['position','velocity','missing','position_authored','velocity_authored']:
            rows=copy.deepcopy(ACTUAL)
            if wrong=='missing':rows.pop(0)
            elif wrong.endswith('_authored'):rows[0][wrong]=False
            else:rows[0][wrong]=4
            with self.assertRaises(RuntimeError):validate_iteration_readback(rows,32)

    def test_absent_link_metadata_never_masks_authored_conflict(self):
        for wrong in ['authored_none','velocity4','unknownstring']:
            rows=copy.deepcopy(ACTUAL);body=next(r for r in rows if r['kind']=='body' and r['velocity'] is None)
            if wrong=='authored_none':body['velocity_authored']=True
            elif wrong=='velocity4':body['velocity']=4;body['velocity_authored']=True
            else:body['velocity']='unknown'
            with self.assertRaises(RuntimeError):validate_iteration_readback(rows,32)

    def test_unowned_body_or_missing_body_still_rejects(self):
        for wrong in ['outside','missing']:
            rows=copy.deepcopy(ACTUAL)
            if wrong=='missing':rows.pop()
            else:rows[-1]['path']='/World/envs/env_31/Robot/NotInArticulation'
            with self.assertRaises(RuntimeError):validate_iteration_readback(rows,32)

if __name__=='__main__':unittest.main()
