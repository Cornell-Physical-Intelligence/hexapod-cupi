from pathlib import Path
import json,sys,hashlib,subprocess,os,time
args=json.load(sys.stdin);base=Path('/home/orionh/HEXAPOD_runs/canonical_direct_20260910/exclusive_reservation_001');coord=Path('/home/orionh/SPARK_COMPUTE_COORDINATION.md')
old=coord.read_bytes();assert hashlib.sha256(old).hexdigest()==args['expected_coord'];assert not base.exists()
active=subprocess.check_output(['systemctl','--user','list-units','--type=service','--state=running','--no-legend','--plain'],text=True)
assert not any(line.split()[0].startswith('hexapod-canonical-native-')for line in active.splitlines()if line.split()),'Owned native allocation still active'
units=[f'stormscope-{name}.{kind}'for name in ['dispatch','scout','monitor','publish','verify']for kind in ['timer','service']]
paths={u:Path('/home/orionh/.config/systemd/user')/(u+'.d')/'90-hexapod-exclusive-reservation-20260910.conf'for u in units}
assert not any(p.exists()for p in paths.values())
snapshot={u:subprocess.check_output(['systemctl','--user','show',u,'-p','LoadState','-p','ActiveState','-p','SubState','-p','UnitFileState','-p','FragmentPath','-p','MainPID'],text=True)for u in units}
base.mkdir();(base/'coordination_before.md').write_bytes(old);(base/'scheduler_before.json').write_text(json.dumps(snapshot,indent=2)+'\n')
(base/'ACTIVE').write_text(json.dumps({'exclusive':True,'authorized_instruction':args['instruction'],'created_unix':time.time(),'release':'Only explicit later user release; preserve exact prior scheduler state.'},indent=2)+'\n')
dropin='[Unit]\n# User-authorized exclusive HEXAPOD reservation; remove ACTIVE only on release.\nConditionPathExists=!'+str(base/'ACTIVE')+'\n'
for p in paths.values():p.parent.mkdir(parents=True,exist_ok=True);p.write_text(dropin)
subprocess.run(['systemctl','--user','daemon-reload'],check=True)
subprocess.run(['systemctl','--user','stop',*[u for u in units if u.endswith('.timer')]],check=True)
# These named dispatch/monitor/publish/verify services are scheduler/control work.
# Stop only live named controllers; completed/exited historical jobs are untouched.
controllers=[u for u in units if u.endswith('.service')and 'SubState=running' in snapshot[u]]
if controllers:subprocess.run(['systemctl','--user','stop',*controllers],check=True)
new=args['header'].encode()+old;temp=coord.with_name('.SPARK_COMPUTE_COORDINATION.hexapod-exclusive.tmp');assert not temp.exists();temp.write_bytes(new);temp.chmod(coord.stat().st_mode&0o777);os.replace(temp,coord)
after={u:subprocess.check_output(['systemctl','--user','show',u,'-p','ActiveState','-p','SubState','-p','Conditions'],text=True)for u in units}
assert all('ActiveState=inactive' in after[u]or 'ActiveState=failed'in after[u]for u in units if u.endswith('.timer'))
receipt={'authorized_instruction':args['instruction'],'phase':'persistent_scheduler_deferral_and_coordination','reservation_path':str(base),'created_unix':time.time(),'coordination_before_sha256':hashlib.sha256(old).hexdigest(),'coordination_sha256':hashlib.sha256(new).hexdigest(),'dropins':{u:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}for u,p in paths.items()},'stopped_live_controller_units':controllers,'scheduler_after':after,'hardware_partition_claimed':False,'general_compute_enforcement':'preparing; no claim that manual competing CUDA launches are yet prevented'}
(base/'initial_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'receipt':receipt,'coordination_before_b64':__import__('base64').b64encode(old).decode(),'coordination_after_b64':__import__('base64').b64encode(new).decode(),'scheduler_before':snapshot}))
