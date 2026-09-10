import importlib.util,json,math,struct,sys,tempfile,unittest,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import analyze_remote as a
from numeric_evidence import numeric
class Checks(unittest.TestCase):
 def test_actual_report_partition(self):
  audit=json.loads((HERE.parent/'canonical_native_standing32_terminal_002/audit.json').read_text());r=audit['standing_report']['replicas']
  self.assertEqual([x['env']for x in r if x['pass']],[0,1,3,5,14,17,22,24,26,28,29]);self.assertEqual(sum(bool(x['quiet']['failed_bounds'])for x in r),7);self.assertEqual(sum(bool(x['failed_physical_bounds'])for x in r),21)
  self.assertEqual(a.sha(HERE.parent/'canonical_native_standing32_terminal_002/audit.json'),a.AUDIT_SHA)
 def test_names_to_grid_does_not_assume_array_order(self):
  report={'replicas':[{'env':0,'pass':True,'physical':{'post_settle_missing_six_toe_substeps':0},'failed_physical_bounds':[],'quiet':{'failed_bounds':[],'joints':{},'max_joint_position_range_rad':0,'max_planar_excursion_m':0}}]}
  x=a.summarize_report(report,['/Robot_019'])[0];self.assertEqual(x['grid_xy_m'],[6.,4.]);self.assertEqual(x['authored_index'],19)
 def test_source_rotation_projection_against_actual_single_patch(self):
  repo=HERE.parents[1];source=repo/'tmp/updated_native_standing_003';sys.path.insert(0,str(source))
  import numpy as np
  from standing_math import Geometry
  meta=json.loads((source/'geometry/geometry.json').read_text());single=repo/'tmp/canonical_native_standing_terminal_004/run/standing';names=json.loads((single/'session.json').read_text())['body_names'];g=Geometry(meta,{},names)
  # Original source002 contact bytes are identical to current native4 by boundSHA.
  raw=repo/'tmp/canonical_native_standing_terminal_002/run/standing/contacts.jsonl'
  with raw.open()as f:
   row=next(r for line in f for r in [json.loads(line)] if any(p['shape_point_m']is not None for p in r['patches']))
  self.assertLess(row['sequence'],800)
  with np.load(single/'substeps_000.npz')as z:pose=z['link_pose_xyzw'][row['sequence'],0]
  tests=0
  for p in row['patches']:
   if p['shape_point_m']is None:continue
   po=pose[names.index(p['body'])];T=g.shapes[p['body']]['shape_to_link'];actual=a.shape_point(p['point_world_m'],po.tolist(),T);expected=g.cap(p['body'],p['point_world_m'],po)[1]
   np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12);tests+=1
  self.assertGreater(tests,0)
 def test_numeric_reader_strict_shape_and_nonfinite(self):
  def write(p,values):
   header=repr({'descr':'<f4','fortran_order':False,'shape':(2,)}).encode();header+=b' '*((16-(10+len(header)+1)%16)%16)+b'\n'
   with zipfile.ZipFile(p,'w')as z:z.writestr('x.npy',b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+struct.pack('<ff',*values))
  with tempfile.TemporaryDirectory()as td:
   p=Path(td)/'x.npz';write(p,[1,2]);self.assertEqual(list(numeric(p,'x',(2,))),[1,2])
   with self.assertRaises(ValueError):numeric(p,'x',(1,2))
   write(p,[float('nan'),0])
   with self.assertRaises(ValueError):numeric(p,'x',(2,))
 def test_no_remote_write_or_execution_imports(self):
  s=(HERE/'analyze_remote.py').read_text();self.assertNotIn('subprocess',s);self.assertNotIn('.write_',s);self.assertNotIn("open('w",s);self.assertNotIn('import numpy',s)
if __name__=='__main__':unittest.main()
