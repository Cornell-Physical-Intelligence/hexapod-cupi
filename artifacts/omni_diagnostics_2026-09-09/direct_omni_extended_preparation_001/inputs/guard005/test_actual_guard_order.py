"""Execute actual guard main with all external effects replaced by fakes."""
import contextlib,io,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch

HERE=Path(__file__).parent
s=importlib.util.spec_from_file_location('actualnativeguard',HERE/'launch_guarded_remote.py')
g=importlib.util.module_from_spec(s);s.loader.exec_module(g)


class ActualMainOrderTests(unittest.TestCase):
 def execute(self,fail_at=None):
  temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);base=Path(temp.name)
  pause,out=base/'pause',base/'out';coord=base/'coord';coord.write_text('fake coordination bytes')
  foreign_status=base/'foreign_status';foreign_status.write_text('Name:\tpython\nPPid:\t1\n')
  events=[];validations=[]
  def call(cmd):
   events.append(('call',cmd))
   if cmd[0]=='nvidia-smi':return '987654, python' if fail_at=='foreign_cuda' else ''
   if cmd[:2]==['docker','ps']:return ''
   if cmd[:3]==['systemctl','--user','show']:
    name=cmd[3];state='active' if name=='stormscope-dispatch.timer' else 'inactive'
    return 'ActiveState='+state+'\nSubState=dead\nMainPID=0'
   raise AssertionError(cmd)
  def run(cmd,**kwargs):
   events.append(('run',cmd))
   if fail_at=='dispatch' and cmd[:2]==['systemd-run','--user'] and '--unit='+g.UNIT in cmd:
    raise subprocess.CalledProcessError(1,cmd)
   return subprocess.CompletedProcess(cmd,0,'','')
  def validate(host):
   validations.append(pause.exists());events.append(('validate',pause.exists()))
   if fail_at=='second_validation' and len(validations)==2:raise RuntimeError('changed after pause')
  parent=SimpleNamespace(check_source=lambda source:events.append(('supervisor_checked',source)))
  old=SimpleNamespace(preflight=lambda:events.append(('post_pause_preflight',None)) or {'no_competitors':True})
  host=SimpleNamespace();spec=SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module:None))
  with contextlib.ExitStack() as stack:
   for name,value in [('PAUSE',pause),('OUTPUT',out),('SUPERVISOR_SOURCE',base/'source'),('HOST',base/'host.py')]:stack.enter_context(patch.object(g,name,value))
   stack.enter_context(patch.object(g,'require_final_bindings',side_effect=lambda:events.append(('bindings',None))))
   stack.enter_context(patch.object(g,'validate_train_inputs',side_effect=validate))
   stack.enter_context(patch.object(g,'verify_previous_owner',side_effect=lambda:events.append(('prior_owner',None))))
   stack.enter_context(patch.object(g,'call',side_effect=call));stack.enter_context(patch.object(g,'sha',return_value=g.HOST_SHA256))
   stack.enter_context(patch.object(g,'Path',side_effect=lambda p:coord if str(p)=='/home/orionh/SPARK_COMPUTE_COORDINATION.md' else (foreign_status if str(p)=='/proc/987654/status' else Path(p))))
   stack.enter_context(patch.object(g.hashlib,'sha256',return_value=SimpleNamespace(hexdigest=lambda:'35af5100e39980f55a3f5e19654e4550b3c3fb6da114b64fbe0d143817242cf3')))
   stack.enter_context(patch.dict(sys.modules,{'launch_reference_physics_spark':parent,'launch_length_study_spark':old}))
   stack.enter_context(patch.object(sys,'path',list(sys.path)))
   stack.enter_context(patch.object(g.importlib.util,'spec_from_file_location',return_value=spec))
   stack.enter_context(patch.object(g.importlib.util,'module_from_spec',return_value=host))
   stack.enter_context(patch.object(g.subprocess,'run',side_effect=run))
   stack.enter_context(patch.object(g.os,'open',side_effect=[101,102]))
   close=stack.enter_context(patch.object(g.os,'close'))
   stack.enter_context(patch.object(g.fcntl,'flock'))
   with contextlib.redirect_stdout(io.StringIO()):
    if fail_at:
     with self.assertRaises((RuntimeError,subprocess.CalledProcessError)):g.main()
    else:g.main()
   self.assertEqual(close.call_count,0 if fail_at=='foreign_cuda' else 2)
  return pause,events,validations

 def test_actual_order_fallback_before_timer_stop_two_checks_then_dispatch(self):
  pause,events,validations=self.execute();self.assertEqual(validations,[False,True])
  runs=[cmd for kind,cmd in events if kind=='run']
  fallback=next(i for i,c in enumerate(runs) if '--on-active=45m' in c)
  stop=next(i for i,c in enumerate(runs) if c[:3]==['systemctl','--user','stop'])
  self.assertLess(fallback,stop)
  dispatch=runs[-1];self.assertIn('--unit='+g.UNIT,dispatch)
  self.assertEqual(dispatch[-4:],['--allocation','smoke','--branch','quiet_priority'])
  self.assertEqual(json.loads((pause/'launch.json').read_text())['command'],dispatch)

 def test_changed_inputs_after_pause_restore_without_stopping_unlaunched_owner(self):
  pause,events,validations=self.execute('second_validation');self.assertEqual(validations,[False,True])
  runs=[cmd for kind,cmd in events if kind=='run']
  self.assertEqual(runs[-1],['/usr/bin/python3',str(pause/'resume_forecasting.py')])
  self.assertFalse(any('--unit='+g.UNIT in cmd for cmd in runs))

 def test_uncertain_dispatch_failure_stops_only_exact_new_owner_then_restores(self):
  pause,events,_=self.execute('dispatch')
  runs=[cmd for kind,cmd in events if kind=='run']
  self.assertEqual(runs[-1],['/usr/bin/python3',str(pause/'resume_forecasting.py'),'--stop-owner'])
  self.assertFalse((pause/'launch.json').exists())

 def test_foreign_cuda_rejected_before_any_pause_or_service_mutation(self):
  pause,events,validations=self.execute('foreign_cuda')
  self.assertEqual(validations,[False]);self.assertFalse(pause.exists())
  self.assertFalse(any(kind=='run' for kind,cmd in events))


if __name__=='__main__':unittest.main()
