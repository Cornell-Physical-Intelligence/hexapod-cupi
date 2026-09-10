from pathlib import Path
import hashlib,json,shutil
base=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
original=base/'omni_control_probe_source_002_no_noise'
prep=base/'ppo_repair_003_preparation'
hashes=json.loads((original/'campaign_source_hashes.json').read_text())
for relative,expected in hashes.items():
    if hashlib.sha256((original/relative).read_bytes()).hexdigest()!=expected:raise RuntimeError(relative)
changed=['tools/train_length_study.py','tools/omni_flat_env.py','tools/omni_diagnostics.py',
         'tools/omni_repair_training.py','tools/launch_omni_repair_pair_spark.py']
plans=[]
for branch,cost in [('a',0.),('b',-2.)]:
    dest=base/f'omni_repair_source_003_{branch}'
    if dest.exists():raise RuntimeError(f'Never overwrite {dest}')
    shutil.copytree(original,dest,ignore=shutil.ignore_patterns('__pycache__'))
    for relative in changed:shutil.copy2(prep/relative,dest/relative)
    deploy=dest/'isaaclab/deploy/hexapod-rl'
    text=deploy.read_text().replace('  omni-repair)\n','  omni-repair-pair)\n    shift\n    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)\n    exec python3 "$script_dir/../../tools/launch_omni_repair_pair_spark.py" "$@"\n    ;;\n  omni-repair)\n')
    deploy.write_text(text)
    path=dest/'robot/hexapod_mkii_length_study/training_plan.json';plan=json.loads(path.read_text())
    plan['training_iterations']=50
    o=plan['omni'];o['overrides'].update(observation_noise_scale=1.,target_filter_time_constant_s=0.)
    o['overrides']['reward_weights']['stand_raw_action']=cost
    o['repair_training']={'checkpoint_sha256':'1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8','exploration_std':.1,'entropy_coef':0.,'optimizer':'reset','learning_rate':5e-5}
    path.write_text(json.dumps(plan,indent=2)+'\n');plans.append(plan)
    manifest={relative:hashlib.sha256((dest/relative).read_bytes()).hexdigest() for relative in sorted(set(hashes)|set(changed))}
    (dest/'campaign_source_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
    unchanged_runtime={relative:sha for relative,sha in manifest.items() if relative.startswith('isaaclab/hexapod_rl/')}
    assert all(hashes[r]==sha for r,sha in unchanged_runtime.items())
    print(json.dumps({'source':str(dest),'hashes':len(manifest),'legacy_runtime_files_unchanged':len(unchanged_runtime),'manifest_sha256':hashlib.sha256((dest/'campaign_source_hashes.json').read_bytes()).hexdigest()}))
import sys
sys.path.insert(0,str(prep/'tools'))
from launch_omni_repair_pair_spark import assert_plan_pair
assert_plan_pair(plans,o['repair_training']['checkpoint_sha256'])
