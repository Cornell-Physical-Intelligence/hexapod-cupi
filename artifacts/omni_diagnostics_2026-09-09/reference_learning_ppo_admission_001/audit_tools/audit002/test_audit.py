from pathlib import Path
from types import SimpleNamespace
import hashlib,json,tempfile,unittest
import audit_remote as a
from fetch_verified import remote
class Tests(unittest.TestCase):
 def test_only_exact_terminal_invocation(self):
  v=dict(ActiveState='inactive',SubState='dead',InvocationID=a.INVOCATION,Result='success',ExecMainStatus='0');a.terminal(v)
  for k,x in [('ActiveState','active'),('SubState','running'),('InvocationID',''),('InvocationID','reused')]:
   with self.subTest(k=k,x=x),self.assertRaises(ValueError):a.terminal({**v,k:x})
 def test_collected_invocation_requires_exact_start_and_terminal_journal(self):
  v=dict(ActiveState='inactive',SubState='dead',InvocationID='',Result='success',ExecMainStatus='0')
  rows=[dict(USER_UNIT=a.UNIT,USER_INVOCATION_ID=a.INVOCATION,MESSAGE='Started '+a.UNIT),dict(USER_UNIT=a.UNIT,USER_INVOCATION_ID=a.INVOCATION,MESSAGE=a.UNIT+': Consumed 10s CPU time')]
  a.terminal(v,rows)
  for corrupt in [rows[:1],[dict(rows[0],USER_INVOCATION_ID='different'),rows[1]],rows+[dict(rows[0],USER_INVOCATION_ID='newer')]]:
   with self.assertRaises(ValueError):a.terminal(v,corrupt)
 def test_known_absence_only(self):
  a.absence(SimpleNamespace(returncode=1,stderr='Error: No such object: x'),'x')
  for code,msg in [(0,''),(1,'permission denied'),(2,'No such object')]:
   with self.assertRaises(ValueError):a.absence(SimpleNamespace(returncode=code,stderr=msg),'x')
 def test_inventory_changed_and_unlisted_reject(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);(p/'one').write_bytes(b'exact');m={'one':a.sha(p/'one')};(p/'map.json').write_text(json.dumps(m));bound=a.sha(p/'map.json')
   a.tree(p,'map.json',bound,1)
   (p/'extra').write_bytes(b'not listed')
   with self.assertRaises(ValueError):a.tree(p,'map.json',bound,1)
   (p/'extra').unlink();(p/'one').write_bytes(b'changed')
   with self.assertRaises(ValueError):a.tree(p,'map.json',bound,1)
 def test_symlink_reject_even_matching_hash(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);r=p/'root';r.mkdir();(p/'elsewhere').write_bytes(b'same');(r/'one').symlink_to(p/'elsewhere')
   (r/'map.json').write_text(json.dumps({'one':a.sha(p/'elsewhere')}))
   with self.assertRaises(ValueError):a.tree(r,'map.json',a.sha(r/'map.json'),1)
 def test_paths_never_escape_expected_root(self):
  self.assertTrue(remote('run/learning_recovery_32/raw/learning_ledger/00001_reset.npz').endswith('reference_learning_ppo_001/learning_recovery_32/raw/learning_ledger/00001_reset.npz'))
  for f in ['../source','run/../../etc','/absolute','other/file']:
   with self.subTest(f=f),self.assertRaises(ValueError):remote(f)
 def test_restoration_exact_prior_timer_set(self):
  p={'unit':a.UNIT,'output':str(a.RUN),'created_unix':1.,'units':{'a.timer':'ActiveState=active','b.timer':'ActiveState=inactive','c.service':'ActiveState=active'}}
  r={'restored_unix':2.,'timers':['a.timer'],'owned_cleanup_checked':[]};a.restoration(p,r)
  for key,value in [('restored_unix',float('nan')),('restored_unix',0.),('timers',['a.timer','b.timer']),('owned_cleanup_checked',None)]:
   with self.subTest(key=key),self.assertRaises(ValueError):a.restoration(p,{**r,key:value})
if __name__=='__main__':unittest.main()
