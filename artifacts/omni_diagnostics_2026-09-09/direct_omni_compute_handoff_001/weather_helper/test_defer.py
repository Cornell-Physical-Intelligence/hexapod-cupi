"""Actual defer/archive/restore logic with fake services and real temporary files."""
import copy,importlib.util,json,os,shutil,subprocess,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).parent
s=importlib.util.spec_from_file_location('halo_defer_test',HERE/'defer_halo.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)


class DeferTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
  self.output=self.base/'defer';self.work=self.base/'weather';self.work.mkdir();self.weather=self.work/'halo104-1010';self.weather.mkdir()
  (self.weather/'forecast.npz').write_bytes(b'complete frame');(self.weather/'frame.tmp').write_bytes(b'interrupted bytes')
  (self.weather/'inputs').mkdir();(self.weather/'inputs/data').write_bytes(b'original input')
  self.log=self.work/'halo104-1010.log';self.log.write_text('three completed frames\n')
  self.script=self.work/'halo_replay.py';self.script.write_text('original weather program\n')
  self.fragment=self.work/'unit.service';self.fragment.write_text('exact reviewed transient definition\n')
  for name,value in [('OUTPUT',self.output),('BASE',self.base),('WORK',self.work),('WEATHER_OUTPUT',self.weather),('WEATHER_LOG',self.log),('SCRIPT',self.script),('SCRIPT_SHA',g.sha(self.script)),('FRAGMENT_SHA',g.sha(self.fragment))]:
   p=patch.object(g,name,value);p.start();self.addCleanup(p.stop)
  self.active={'LoadState':'loaded','ActiveState':'active','InvocationID':g.INVOCATION,'MainPID':'3544965','ControlGroup':g.CGROUP,'FragmentPath':str(self.fragment)}
  self.stopped={'LoadState':'not-found','ActiveState':'inactive','InvocationID':'','MainPID':'0','FragmentPath':''}
  self.calls=[];self.is_stopped=False
 def fake_run(self,cmd,**kwargs):
  self.calls.append(cmd)
  if cmd[:3]==['systemctl','--user','stop']:self.assertEqual(cmd[3],g.UNIT);self.is_stopped=True
  return subprocess.CompletedProcess(cmd,0,'','')
 def inspect(self):return dict(self.stopped if self.is_stopped else self.active)
 def patch_effects(self):
  for name,value in [('bundle',lambda:'a'*64),('inspect_weather',self.inspect),('run',self.fake_run),('matching_active',lambda state:state['ActiveState']=='active')]:
   p=patch.object(g,name,side_effect=value);p.start();self.addCleanup(p.stop)
 def defer(self):
  self.patch_effects();return g.defer()
 def restore_with(self,**kwargs):
  with patch.object(g.os,'open',side_effect=[71,72]),patch.object(g.os,'close') as close,patch.object(g.fcntl,'flock'),patch.object(g,'prove_hexapod_absent',**kwargs),patch.object(g,'await_restart',return_value={'own_weather_lock_verified':True}):
   result=g.restore();self.assertEqual(close.call_count,2);return result
 def test_fallback_precedes_exact_stop_full_archive_retains_every_file(self):
  before=g.tree(self.weather);log=g.sha(self.log);r=self.defer()
  self.assertEqual(self.calls[0][0],'systemd-run');self.assertIn('--on-active=90m',self.calls[0])
  self.assertEqual(self.calls[1],['systemctl','--user','stop',g.UNIT])
  self.assertTrue(r['archive_complete']);self.assertEqual(g.tree(self.weather),before)
  self.assertEqual(g.tree(self.output/r['archive']/'outputs'),before)
  self.assertEqual(g.sha(self.output/r['archive']/'halo104-1010.log'),log)
  self.assertFalse(any('stormscope-dispatch' in ' '.join(c) or 'stormscope-scout' in ' '.join(c) for c in self.calls))
 def test_fallback_failure_never_stops_weather(self):
  self.patch_effects()
  def fail(cmd,**kw):self.calls.append(cmd);raise subprocess.CalledProcessError(1,cmd)
  with patch.object(g,'run',side_effect=fail),self.assertRaises(subprocess.CalledProcessError):g.defer()
  self.assertFalse(self.is_stopped);self.assertEqual(len(self.calls),1)
 def test_completed_before_defer_skips_all_mutations(self):
  self.patch_effects();self.is_stopped=True
  self.assertEqual(g.defer()['status'],'skipped_completed_naturally');self.assertEqual(self.calls,[]);self.assertFalse(self.output.exists())
 def test_completed_between_fallback_and_stop_is_not_restarted(self):
  self.patch_effects()
  with patch.object(g,'inspect_weather',side_effect=[self.active,self.stopped]):r=g.defer()
  self.assertEqual(r['status'],'skipped_completed_naturally');self.assertEqual(len(self.calls),1)
  self.assertEqual(g.restore()['status'],'no_deferred_weather_to_restart')
 def test_replacement_invocation_rejected_by_actual_identity_guard(self):
  state={**self.active,'InvocationID':'b'*32}
  with self.assertRaisesRegex(RuntimeError,'replacement'):g.matching_active(state)
  with self.assertRaisesRegex(RuntimeError,'changed after stop'):g.stopped_original({**self.stopped,'InvocationID':'b'*32})
 def test_archive_failure_keeps_weather_stopped_without_restart(self):
  self.patch_effects()
  with patch.object(g.shutil,'copytree',side_effect=OSError('disk error')),self.assertRaises(OSError):g.defer()
  self.assertTrue(self.is_stopped);self.assertFalse(any(c[:3]==['systemctl','--user','start'] for c in self.calls))
  self.assertFalse(g.read(self.output/'state.json')['archive_complete'])
 def test_corrupt_archive_prevents_restart_and_preserves_it(self):
  r=self.defer();bad=self.output/r['archive']/'outputs/forecast.npz';bad.write_bytes(b'corrupted')
  count=len(self.calls)
  with self.assertRaisesRegex(RuntimeError,'archive changed'):g.restore()
  self.assertEqual(len(self.calls),count);self.assertEqual(bad.read_bytes(),b'corrupted')
 def test_collected_transient_recreated_only_from_exact_command(self):
  self.defer();r=self.restore_with(return_value={'all_absent':True})
  cmd=self.calls[-1];self.assertEqual(cmd[:2],['systemd-run','--user']);self.assertIn('--unit='+g.UNIT,cmd)
  self.assertEqual(cmd[-3:],['/bin/bash','-c',g.SHELL_COMMAND]);self.assertIn('--property=RuntimeMaxSec=2h',cmd)
  self.assertTrue(r['recomputes_original_replay']);self.assertFalse(r['midframe_resume'])
  before=len(self.calls);self.assertEqual(g.restore(),r);self.assertEqual(len(self.calls),before)
 def test_still_loaded_exact_unit_restarted_without_recreating(self):
  self.defer();self.stopped.update(LoadState='loaded',FragmentPath=str(self.fragment))
  with patch.object(g,'same_definition') as definition:r=self.restore_with(return_value={'all_absent':True})
  definition.assert_called_once();self.assertEqual(self.calls[-1],['systemctl','--user','start',g.UNIT]);self.assertEqual(r['method'],'start_existing_exact_unit')
 def test_active_hexapod_blocks_restart_and_releases_both_locks(self):
  self.defer();n=len(self.calls)
  with patch.object(g.os,'open',side_effect=[71,72]),patch.object(g.os,'close') as close,patch.object(g.fcntl,'flock'),patch.object(g,'prove_hexapod_absent',side_effect=RuntimeError('HEXAPOD active')):
   with self.assertRaisesRegex(RuntimeError,'HEXAPOD'):g.restore()
   self.assertEqual(close.call_count,2)
  self.assertEqual(len(self.calls),n);self.assertFalse((self.output/'restored.json').exists())
 def test_gpu_lock_contention_prevents_restart_without_reservation(self):
  self.defer();n=len(self.calls)
  with patch.object(g.os,'open',return_value=71),patch.object(g.os,'close') as close,patch.object(g.fcntl,'flock',side_effect=BlockingIOError):
   with self.assertRaises(BlockingIOError):g.restore()
   close.assert_called_once_with(71)
  self.assertEqual(len(self.calls),n)
 def test_actual_absence_check_blocks_services_cuda_and_unknown_container(self):
  for outputs in [('hexapod-owner.service loaded active running','', ''),('', '123,python',''),('', '', 'abc hexapod-owned')]:
   replies=iter(outputs)
   with patch.object(g,'run',side_effect=lambda *a,**kw:subprocess.CompletedProcess(a,0,next(replies),'')):
    with self.assertRaises(RuntimeError):g.prove_hexapod_absent()
  jobs=self.base/'direct_omni_case/jobs';jobs.mkdir(parents=True);(jobs/'phase.json').write_text(json.dumps({'container_name':'exact-owned','container_id':'c'*64}))
  def answer(cmd,**kw):return subprocess.CompletedProcess(cmd,1,'','daemon unavailable') if cmd[:2]==['docker','inspect'] else subprocess.CompletedProcess(cmd,0,'','')
  with patch.object(g,'run',side_effect=answer),self.assertRaisesRegex(RuntimeError,'absence unknown'):g.prove_hexapod_absent()

 def test_verification_locks_release_before_restarting_nonblocking_producer(self):
  self.defer()
  with patch.object(g.os,'open',side_effect=[71,72]),patch.object(g.os,'close') as close,patch.object(g.fcntl,'flock'),patch.object(g,'prove_hexapod_absent',return_value={}),patch.object(g,'await_restart',return_value={'own_weather_lock_verified':True}):
   def restart(cmd,**kw):
    self.assertEqual(close.call_count,2)
    return subprocess.CompletedProcess(cmd,0,'','')
   with patch.object(g,'run',side_effect=restart):g.restore()
   self.assertEqual(close.call_count,2)

 def test_restarted_unit_lock_loss_is_explicit_failure_without_retry(self):
  self.defer();count=len(self.calls)
  with patch.object(g.os,'open',side_effect=[71,72]),patch.object(g.os,'close'),patch.object(g.fcntl,'flock'),patch.object(g,'prove_hexapod_absent',return_value={}),patch.object(g,'await_restart',side_effect=RuntimeError('exited before lock')):
   with self.assertRaisesRegex(RuntimeError,'before lock'):g.restore()
  self.assertEqual(len(self.calls),count+1);self.assertFalse((self.output/'restored.json').exists())

 def test_actual_kernel_lock_proof_matches_pid_device_and_inode(self):
  state={'ActiveState':'active','InvocationID':'e'*32,'MainPID':'99','ControlGroup':g.CGROUP,
         'WorkingDirectory':str(g.WORK),'ExecStart':'{ argv[]=/bin/bash -c '+g.SHELL_COMMAND+' ; }'}
  fake=SimpleNamespace(stat=lambda:SimpleNamespace(st_dev=os.makedev(8,1),st_ino=222),
       read_text=lambda:'12: FLOCK ADVISORY WRITE 98 08:01:222 0 EOF\n13: FLOCK ADVISORY WRITE 99 08:01:222 0 EOF\n')
  with patch.object(g,'Path',return_value=fake),patch.object(g,'inspect_weather',return_value=state),patch.object(g.time,'monotonic',side_effect=[0.,1.]):
   result=g.await_restart()
  self.assertEqual(result['main_pid'],99);self.assertTrue(result['own_weather_lock_verified'])

 def test_archive_manifest_replacement_rejects_before_recompute(self):
  r=self.defer();archive=self.output/r['archive'];data=archive/'outputs/forecast.npz';data.write_bytes(b'changed')
  hashes=g.read(archive/'SHA256.json');hashes['outputs/forecast.npz']=g.sha(data);g.save(archive/'SHA256.json',hashes)
  count=len(self.calls)
  with self.assertRaisesRegex(RuntimeError,'manifest changed'):g.restore()
  self.assertEqual(len(self.calls),count)


if __name__=='__main__':unittest.main()
