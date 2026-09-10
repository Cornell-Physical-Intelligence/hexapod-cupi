"""Read-only helper review: mocked service/Docker/locks; real temporary archive bytes."""
import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'forecast_halo_defer_helper_001/defer_halo.py'
spec=importlib.util.spec_from_file_location('_halo_helper_review',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Tests(unittest.TestCase):
    def test_replacement_invocation_or_pid_never_matches(self):
        base={'ActiveState':'active','InvocationID':m.INVOCATION,'MainPID':'3544965','ControlGroup':m.CGROUP}
        for delta in [{'InvocationID':'replacement'},{'MainPID':'12345'},{'ControlGroup':'/other'}]:
            with patch.object(m,'same_definition') as definition, self.assertRaisesRegex(RuntimeError,'replacement'):
                m.matching_active({**base,**delta})
            definition.assert_not_called()

    def fixture(self,root):
        root=Path(root);work=root/'work';work.mkdir();out=work/'halo';out.mkdir();(out/'forecast.npz').write_bytes(b'partial forecast bytes');(out/'input.npz').write_bytes(b'input bytes')
        log=work/'log.txt';log.write_bytes(b'prior append log');script=work/'halo_replay.py';script.write_text('print("old exact script")\n')
        receipt=root/'receipt';receipt.mkdir()
        state={'LoadState':'not-found','ActiveState':'inactive','MainPID':'0','InvocationID':''}
        record={'helper_freeze_sha256':'freeze','initial_unit_fragment':'exact old fragment','previously_active':True,'stop_requested':True}
        (receipt/'state.json').write_text(json.dumps(record))
        return work,out,log,script,receipt,state,record

    def test_collected_restore_archives_before_same_unit_recreation(self):
        with tempfile.TemporaryDirectory() as root:
            work,out,log,script,receipt,state,record=self.fixture(root);events=[]
            def execute(argv,**kw):
                events.append(('start',argv));self.assertTrue((receipt/'archive_001/outputs/forecast.npz').is_file());return NS(returncode=0,stdout='',stderr='')
            with patch.multiple(m,WORK=work,WEATHER_OUTPUT=out,WEATHER_LOG=log,SCRIPT=script,SCRIPT_SHA=m.sha(script),FRAGMENT_SHA=hashlib.sha256(record['initial_unit_fragment'].encode()).hexdigest(),OUTPUT=receipt),patch.object(m,'bundle',return_value='freeze'),patch.object(m,'inspect_weather',return_value=state),patch.object(m,'prove_hexapod_absent',return_value={'clear':True}),patch.object(m,'await_restart',return_value={'own_weather_lock_verified':True}),patch.object(m,'run',side_effect=execute),patch.object(m.os,'open',side_effect=[11,12]),patch.object(m.os,'close',side_effect=lambda fd:events.append(('close',fd))),patch.object(m.fcntl,'flock'):
                result=m.restore()
            starts=[e for e in events if e[0]=='start'];self.assertEqual(len(starts),1)
            argv=starts[0][1];self.assertIn('--unit='+m.UNIT,argv);self.assertEqual(argv[-3:],['/bin/bash','-c',m.SHELL_COMMAND])
            self.assertEqual(result['method'],'recreate_collected_exact_transient_unit')
            self.assertEqual((receipt/'archive_001/outputs/forecast.npz').read_bytes(),b'partial forecast bytes')
            self.assertEqual((receipt/'archive_001/halo104-1010.log').read_bytes(),b'prior append log')
            # Helper must not hold the lock that the restored script acquires nonblocking.
            self.assertTrue(all(events.index(('close',fd))<events.index(starts[0]) for fd in [11,12]))

    def test_archive_manifest_is_bound_to_original_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            work,out,log,script,receipt,state,record=self.fixture(root)
            with patch.multiple(m,WORK=work,WEATHER_OUTPUT=out,WEATHER_LOG=log,SCRIPT=script,SCRIPT_SHA=m.sha(script),FRAGMENT_SHA=hashlib.sha256(record['initial_unit_fragment'].encode()).hexdigest(),OUTPUT=receipt),patch.object(m,'inspect_weather',return_value=state):
                m.archive(record);archive=receipt/record['archive'];payload=archive/'outputs/forecast.npz';payload.write_bytes(b'replaced')
                manifest=json.loads((archive/'SHA256.json').read_text());manifest['outputs/forecast.npz']=m.sha(payload);(archive/'SHA256.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError,'archive|Archive'):m.archive(record)

    def test_unknown_docker_inspection_is_not_absence(self):
        with tempfile.TemporaryDirectory() as root:
            base=Path(root);jobs=base/'direct_omni_test/jobs';jobs.mkdir(parents=True);(jobs/'standing.json').write_text(json.dumps({'container_name':'hexapod-owned','container_id':'abc123'}))
            def call(argv,**kw):
                if argv[:2]==['docker','inspect']:return NS(returncode=1,stdout='',stderr='permission denied')
                return NS(returncode=0,stdout='',stderr='')
            with patch.object(m,'BASE',base),patch.object(m,'run',side_effect=call),self.assertRaisesRegex(RuntimeError,'unknown'):
                m.prove_hexapod_absent()

    def test_previously_inactive_never_restarts(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);(p/'state.json').write_text(json.dumps({'helper_freeze_sha256':'freeze','previously_active':False,'stop_requested':True}))
            with patch.object(m,'OUTPUT',p),patch.object(m,'bundle',return_value='freeze'),patch.object(m,'run') as run,patch.object(m,'archive') as archive:
                result=m.restore()
            self.assertEqual(result['status'],'no_deferred_weather_to_restart');run.assert_not_called();archive.assert_not_called()

    def test_kernel_lock_proof_requires_new_pid_exact_device_and_inode(self):
        state={'ActiveState':'active','InvocationID':'a'*32,'MainPID':'777','ControlGroup':m.CGROUP,'WorkingDirectory':str(m.WORK),
               'ExecStart':'{ path=/bin/bash ; argv[]=/bin/bash -c '+m.SHELL_COMMAND+' ; ignore_errors=no ; }'}
        stat=NS(st_dev=123,st_ino=456)
        for line,passes in [('1: FLOCK ADVISORY WRITE 777 07:03:456 0 EOF',True),
                            ('1: FLOCK ADVISORY WRITE 778 07:03:456 0 EOF',False),
                            ('1: FLOCK ADVISORY WRITE 777 07:03:457 0 EOF',False)]:
            with patch.object(m,'inspect_weather',return_value=state),patch.object(Path,'stat',return_value=stat),patch.object(Path,'read_text',return_value=line),patch.object(m.os,'major',return_value=7),patch.object(m.os,'minor',return_value=3),patch.object(m.time,'monotonic',side_effect=[0,0,91]),patch.object(m.time,'sleep'):
                if passes:self.assertTrue(m.await_restart()['own_weather_lock_verified'])
                else:
                    with self.assertRaisesRegex(RuntimeError,'within90s'):m.await_restart()
        with patch.object(m,'inspect_weather',return_value={'ActiveState':'failed'}),patch.object(Path,'stat',return_value=stat),patch.object(m.time,'monotonic',side_effect=[0,0]),self.assertRaisesRegex(RuntimeError,'exited'):
            m.await_restart()

if __name__=='__main__':unittest.main()
