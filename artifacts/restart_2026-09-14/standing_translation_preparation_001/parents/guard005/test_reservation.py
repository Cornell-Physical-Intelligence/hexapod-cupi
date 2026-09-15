"""CPU-only exact mask, queue-lock and immutable-file failure cases."""
import contextlib,copy,hashlib,importlib.util,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('mask_guard',HERE/'launch_guarded_remote.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Reservation(unittest.TestCase):
 def fixture(self):
  t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);base=Path(t.name);(base/'inputs').mkdir()
  reservation=base/'reservation';reservation.mkdir();marker=reservation/'ACTIVE';marker.write_text('{"exclusive":true}')
  original=base/'original_condition.conf';original.write_text('[Unit]\nConditionPathExists=!ACTIVE\n')
  entry=base/'worker.py';entry.mkdir();worker=entry/'__main__.py';worker.write_text('raise SystemExit("deferred")\n')
  names=json.loads((HERE/'inputs/reservation_policy.json').read_text())['mask_paths'];masks=[];states={}
  for source in names:
   p=base/'units'/Path(source).name;p.parent.mkdir(exist_ok=True);p.symlink_to('/dev/null');masks.append(str(p))
   states[p.name]={'LoadState':'masked','UnitFileState':'masked','ActiveState':'inactive','FragmentPath':str(p),'NeedDaemonReload':'yes'}
   if p.suffix=='.service':states[p.name]['MainPID']='0'
  backup=base/'backup';backup.mkdir();(backup/'old.service').write_text('original definition')
  before=base/'before.json';before.write_text(json.dumps({'files':{'old.service':{'kind':'file','sha256':digest(backup/'old.service')}}}))
  helper=base/'hold_queue_lock.py';helper.write_text('reviewed lock holder');unit=base/'queue.service';unit.write_text('reviewed unit')
  lock=base/'queue-worker.lock';lock.touch();record=base/'lock_acquired.json';record.write_text(json.dumps({'pid':1234,'lock_path':str(lock),'reservation_path':str(marker),'reservation_sha256':digest(marker),'queue_mutated':False}))
  proc=base/'proc';(proc/'1234').mkdir(parents=True);(proc/'1234/cmdline').write_bytes(('/usr/bin/python3\0-B\0'+str(helper)+'\0').encode())
  st=lock.stat();inode=f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}';(proc/'locks').write_text('8: FLOCK ADVISORY WRITE 1234 '+inode+' 0 EOF\n')
  system=['wx-forecast.service','wx-forecast.timer','wx-forecast-pm.service','wx-forecast-pm.timer']
  for name in system:
   states[name]={'LoadState':'masked','ActiveState':'inactive','UnitFileState':'masked'}
   if name.endswith('.service'):states[name]['MainPID']='0'
  states['queue.service']={'ActiveState':'active','SubState':'running','MainPID':'1234','UnitFileState':'enabled','NeedDaemonReload':'no','FragmentPath':str(unit),'DropInPaths':''}
  policy={'schema':'canonical_exclusive_mask_reservation_v2','blocked_entry_directories':[str(entry)],'release_only_on_user_instruction':True,'files':{str(p):digest(p)for p in (worker,helper,unit,before)},'mask_paths':masks,'system_masked_units':system,'original_units_before':str(before),'original_units_backup':str(backup),'queue_unit':'queue.service','queue_unit_path':str(unit),'queue_helper':str(helper),'queue_lock_path':str(lock),'queue_lock_record':str(record),'marker_sha256':digest(marker)}
  file=base/'inputs/reservation_policy.json';file.write_text(json.dumps(policy))
  return dict(base=base,reservation=reservation,original=original,worker=worker,entry=entry,backup=backup,proc=proc,states=states,policy=policy,file=file,record=record,masks=masks)
 @contextlib.contextmanager
 def run_fixture(self,f):
  paths=lambda p: f['proc'] if str(p)=='/proc' else f['proc']/'locks' if str(p)=='/proc/locks' else Path(p)
  def call(cmd):
   self.assertEqual(cmd[0],'systemctl');self.assertIn('show',cmd);name=cmd[cmd.index('show')+1]
   return '\n'.join(k+'='+v for k,v in f['states'][name].items())
  pins={str(f['reservation']/'ACTIVE'):digest(f['reservation']/'ACTIVE'),str(f['original']):digest(f['original'])}
  with patch.multiple(g,__file__=str(f['base']/'launch_guarded_remote.py'),RESERVATION=f['reservation'],RESERVATION_PINS=pins,MASK_POLICY_SHA256=digest(f['file'])),patch.object(g,'Path',side_effect=paths),patch.object(g,'call',side_effect=call),patch.object(g.subprocess,'run')as mutation:
   yield
   mutation.assert_not_called()
 def test_all31_masks_system4_missing_timer_pid_and_reload_yes_are_truthful(self):
  f=self.fixture()
  with self.run_fixture(f):r=g.verify_reservation()
  self.assertEqual(r['masked_user_units'],31);self.assertTrue(r['queue_lock_held']);self.assertFalse(r['release_attempted']);self.assertEqual(set(r['mask_reload_flags'].values()),{'yes'})
 def test_mask_removed_changed_target_or_loaded_state_rejects(self):
  for mode in ('missing','wrong_target','loaded','running','service_pid'):
   f=self.fixture();p=Path(f['masks'][0]);name=p.name
   if mode in ('missing','wrong_target'):
    p.unlink()
    if mode=='wrong_target':p.symlink_to('/tmp/not-the-mask')
   elif mode=='loaded':f['states'][name]['LoadState']='loaded'
   elif mode=='running':f['states'][name]['ActiveState']='active'
   else:f['states'][Path(f['masks'][1]).name]['MainPID']='9'
   with self.subTest(mode=mode),self.run_fixture(f),self.assertRaises(RuntimeError):g.verify_reservation()
 def test_entry_directory_file_symlink_or_missing_main_rejects(self):
  for mode in ('file','symlink','missing_main'):
   f=self.fixture();f['worker'].unlink()
   if mode in ('file','symlink'):
    f['entry'].rmdir()
    if mode=='file':f['entry'].write_text('restored GPU worker')
    else:f['entry'].symlink_to(f['base']/'backup',target_is_directory=True)
   with self.subTest(mode=mode),self.run_fixture(f),self.assertRaises(RuntimeError):g.verify_reservation()
 def test_all_system_forecast_masks_required(self):
  for name in ('wx-forecast.service','wx-forecast.timer','wx-forecast-pm.service','wx-forecast-pm.timer'):
   f=self.fixture();f['states'][name]['LoadState']='loaded'
   with self.subTest(name=name),self.run_fixture(f),self.assertRaisesRegex(RuntimeError,'System forecast'):g.verify_reservation()
 def test_worker_original_backup_and_policy_mutation_rejects(self):
  for mode in ('worker','backup','policy'):
   f=self.fixture()
   with self.run_fixture(f):
    if mode=='worker':f['worker'].write_text('changed')
    elif mode=='backup':(f['backup']/'old.service').write_text('changed')
    else:f['file'].write_text('{}')
    with self.subTest(mode=mode),self.assertRaises(RuntimeError):g.verify_reservation()
 def test_inactive_overridden_or_pid_mismatched_queue_holder_rejects(self):
  for key,value in [('ActiveState','inactive'),('UnitFileState','disabled'),('DropInPaths','/tmp/unknown.conf'),('MainPID','0'),('MainPID','5678')]:
   f=self.fixture();f['states']['queue.service'][key]=value
   with self.subTest(key=key),self.run_fixture(f),self.assertRaises(RuntimeError):g.verify_reservation()
 def test_lock_record_pid_command_and_actual_inode_owner_are_required(self):
  for mode in ('record','command','inode','pid','no_lock'):
   f=self.fixture()
   if mode=='record':d=json.loads(f['record'].read_text());d['pid']=999;f['record'].write_text(json.dumps(d))
   elif mode=='command':(f['proc']/'1234/cmdline').write_bytes(b'/usr/bin/sleep\0infinity\0')
   elif mode=='inode':(f['proc']/'locks').write_text('8: FLOCK ADVISORY WRITE 1234 00:00:1 0 EOF\n')
   elif mode=='pid':p=f['proc']/'locks';p.write_text(p.read_text().replace('WRITE 1234','WRITE 9'))
   else:(f['proc']/'locks').write_text('')
   with self.subTest(mode=mode),self.run_fixture(f),self.assertRaises(RuntimeError):g.verify_reservation()
 def test_actual_queue_record_marker_path_is_file_not_directory(self):
  actual=json.loads((HERE/'inputs/reconstruction_block_receipt.json').read_text())['queue_lock']
  self.assertEqual(actual['reservation_path'],str(g.RESERVATION/'ACTIVE'))
  f=self.fixture();record=json.loads(f['record'].read_text());record['reservation_path']=str(f['reservation']);f['record'].write_text(json.dumps(record))
  with self.run_fixture(f),self.assertRaisesRegex(RuntimeError,'acquisition record'):g.verify_reservation()
 def test_exact_policy_matches_root_receipts_and_original_reservation_files(self):
  policy=json.loads((HERE/'inputs/reservation_policy.json').read_text());mask=json.loads((HERE/'inputs/automation_block_verified.json').read_text());worker=json.loads((HERE/'inputs/reconstruction_block_receipt.json').read_text())
  self.assertEqual(policy['mask_paths'],mask['mask_paths']);self.assertEqual(len(policy['mask_paths']),31);self.assertEqual(set(policy['system_masked_units']),set(mask['system_units_already_masked']))
  entry=json.loads((HERE/'inputs/reconstruction_entry_block_receipt.json').read_text())
  self.assertEqual(policy['blocked_entry_directories'],[entry['blocked_entry_directory']])
  self.assertNotIn(entry['blocked_entry_directory'],policy['files'])
  for p,h in worker['files'].items():
   if p!=entry['blocked_entry_directory']:self.assertEqual(policy['files'][p],h)
  for p,meta in entry['files'].items():self.assertEqual(policy['files'][p],meta['sha256'])
  self.assertEqual(len(g.RESERVATION_PINS),13)
if __name__=='__main__':unittest.main()
