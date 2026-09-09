"""CPU-only Docker inventory races; never invokes Docker or queries a real GPU."""
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
loader = importlib.machinery.SourceFileLoader('_fourbar_foreign_race_tests',str(ROOT/'isaaclab/deploy/run-mkii-fourbar'))
spec = importlib.util.spec_from_loader(loader.name,loader)
supervisor = importlib.util.module_from_spec(spec)
loader.exec_module(supervisor)

GONE, CPU, FOREIGN_GPU, OWNED = ('a'*64,'b'*64,'c'*64,'d'*64)
CPU_CONFIG = {'runtime':'runc','device_requests':None,'devices':[],'privileged':False}


class ForeignContainerRaceTests(unittest.TestCase):
    def gate(self, inspections, *, gpu_rows='', utilization='0\n', owned_container=None,
             allow_owned_gpu=False, process_rows=''):
        calls = []
        def run(argv, **kwargs):
            calls.append(list(argv))
            if argv[0] == 'ps':
                result = (0,process_rows,'')
            elif argv[0] == 'systemctl':
                result = (3,'inactive\n','')
            elif argv[:2] == ['docker','ps']:
                result = (0,'\n'.join(inspections)+'\n','')
            elif argv[:2] == ['docker','inspect']:
                result = inspections[argv[-1]]
            elif '--query-compute-apps=pid,process_name' in argv:
                result = gpu_rows if isinstance(gpu_rows,tuple) else (0,gpu_rows,'')
            elif '--query-gpu=utilization.gpu' in argv:
                result = utilization if isinstance(utilization,tuple) else (0,utilization,'')
            else:
                raise AssertionError(f'Unexpected command: {argv}')
            return subprocess.CompletedProcess(argv,*result)
        with patch.object(supervisor,'available_memory_bytes',return_value=32*1024**3), \
             patch.object(supervisor.subprocess,'run',side_effect=run):
            # Exercise the actual command wrapper's return-code handling.
            result = supervisor.resource_gate(owned_container=owned_container,allow_owned_gpu=allow_owned_gpu)
        return result,calls

    def inspect(self, returncode=1, stdout='', stderr=None, identifier=GONE):
        result = subprocess.CompletedProcess(['docker','inspect'],returncode,stdout,
            f'Error: No such object: {identifier}\n' if stderr is None else stderr)
        with patch.object(supervisor.subprocess,'run',return_value=result) as call:
            value = supervisor.inspect_foreign_container_gpu_config(identifier)
        self.assertEqual(call.call_count,1)
        self.assertEqual(call.call_args.args[0][-1],identifier)
        return value

    def test_exact_removal_race_recorded_and_both_gpu_inventories_still_required(self):
        result,calls = self.gate({GONE:(1,'',f'Error: No such object: {GONE}\n'),
                                 CPU:(0,json.dumps(CPU_CONFIG),'')})
        self.assertEqual(result['disappeared_foreign_container_ids'],[GONE])
        self.assertEqual(result['unrelated_cpu_container_ids'],[CPU])
        self.assertEqual(result['gpu_pids'],[])
        self.assertEqual(result['gpu_utilization_percent'],[0])
        self.assertEqual(sum(command[:2] == ['docker','inspect'] for command in calls),2)
        self.assertTrue(any('--query-compute-apps=pid,process_name' in command for command in calls))
        self.assertTrue(any('--query-gpu=utilization.gpu' in command for command in calls))

    def test_exact_docker_missing_object_and_container_messages_only(self):
        for prefix in ('','Error: ','Error response from daemon: '):
            for kind in ('object','container'):
                for ending in ('','\n','\r\n'):
                    with self.subTest(prefix=prefix,kind=kind,ending=ending):
                        self.assertIsNone(self.inspect(stderr=f'{prefix}No such {kind}: {GONE}{ending}'))

    def test_stderr_substrings_wrong_ids_or_nonempty_stdout_cannot_prove_removal(self):
        errors = [f'Error: No such object: {CPU}\n',f'Error: No such object: {GONE[:12]}\n',
                  f'permission denied: No such object: {GONE}\n',
                  f'Error: No such object: {GONE}\nCannot connect to Docker daemon\n',
                  f'Error: No such object: {GONE} trailing\n',f' Error: No such object: {GONE}\n']
        for message in errors:
            with self.subTest(message=message),self.assertRaises(supervisor.Blocked):
                self.inspect(stderr=message)
        for output in ('\n','[]\n',json.dumps(CPU_CONFIG)):
            with self.subTest(stdout=output),self.assertRaises(supervisor.Blocked):
                self.inspect(stdout=output)
        with self.assertRaises(supervisor.Blocked):
            self.inspect(returncode=125)

    def test_daemon_permission_and_unknown_inspection_failures_remain_blocking(self):
        for message in ('Cannot connect to the Docker daemon','permission denied while connecting',
                        'context deadline exceeded',''):
            with self.subTest(message=message),self.assertRaises(supervisor.Blocked):
                self.inspect(stderr=message)
        with patch.object(supervisor.subprocess,'run',side_effect=subprocess.TimeoutExpired('docker',20)):
            with self.assertRaises(subprocess.TimeoutExpired):
                supervisor.inspect_foreign_container_gpu_config(GONE)

    def test_invalid_or_truncated_inventory_id_rejected_before_inspection(self):
        for identifier in ('cpu-reader',GONE[:12],GONE.upper(),GONE+'a'):
            with self.subTest(identifier=identifier),patch.object(supervisor.subprocess,'run') as call:
                with self.assertRaises(supervisor.Blocked):
                    supervisor.inspect_foreign_container_gpu_config(identifier)
                call.assert_not_called()

    def test_success_with_unparseable_or_incomplete_configuration_remains_blocking(self):
        values = ['','not JSON','[]','null','{}',json.dumps({**CPU_CONFIG,'runtime':None}),
                  json.dumps({**CPU_CONFIG,'privileged':0}),json.dumps({**CPU_CONFIG,'devices':[{}]}),
                  json.dumps({**CPU_CONFIG,'device_requests':[{'Driver':'','Capabilities':['gpu']}]}),
                  json.dumps({**CPU_CONFIG,'device_requests':[{'Driver':'','Capabilities':None}]})]
        for value in values:
            with self.subTest(value=value),self.assertRaisesRegex(supervisor.Blocked,'malformed'):
                self.inspect(returncode=0,stdout=value,stderr='')

    def test_disappeared_container_does_not_exempt_foreign_gpu_pid_or_busy_gpu(self):
        gone = {GONE:(1,'',f'No such container: {GONE}\n')}
        with self.assertRaisesRegex(supervisor.Blocked,'Unrelated or premature GPU process'):
            self.gate(gone,gpu_rows='9876, foreign_process\n')
        with self.assertRaisesRegex(supervisor.Blocked,'GPU utilization is nonzero'):
            self.gate(gone,utilization='15\n')

    def test_gpu_inventory_or_utilization_failure_after_race_still_blocks(self):
        gone = {GONE:(1,'',f'No such object: {GONE}\n')}
        cases = ({'gpu_rows':(1,'','GPU driver unavailable')},{'gpu_rows':'not-a-pid, process\n'},
                 {'utilization':(1,'','GPU driver unavailable')},{'utilization':''},{'utilization':'N/A\n'})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs),self.assertRaises(supervisor.Blocked):
                self.gate(gone,**kwargs)

    def test_other_foreign_gpu_capable_container_still_vetoes_after_race(self):
        configurations = [dict(CPU_CONFIG,runtime='nvidia'),dict(CPU_CONFIG,privileged=True),
            dict(CPU_CONFIG,devices=[{'PathOnHost':'/dev/nvidia0'}]),
            dict(CPU_CONFIG,device_requests=[{'Driver':'','Capabilities':[['gpu']]}])]
        for description in configurations:
            with self.subTest(description=description),self.assertRaisesRegex(supervisor.Blocked,'GPU-capable container'):
                self.gate({GONE:(1,'',f'No such object: {GONE}\n'),
                           FOREIGN_GPU:(0,json.dumps(description),'')})

    def test_owned_immutable_id_is_still_skipped_and_owned_gpu_descendant_allowed(self):
        description = {'id':OWNED,'pid':321}
        result,calls = self.gate({OWNED:None,GONE:(1,'',f'No such object: {GONE}\n')},
            owned_container=description,allow_owned_gpu=True,process_rows='321 1 owned-container\n322 321 python\n',
            gpu_rows='322, owned_python\n',utilization='80\n')
        self.assertEqual(result['gpu_pids'],[322])
        self.assertEqual(result['disappeared_foreign_container_ids'],[GONE])
        inspected = [command[-1] for command in calls if command[:2] == ['docker','inspect']]
        self.assertEqual(inspected,[GONE])


if __name__ == '__main__':
    unittest.main()
