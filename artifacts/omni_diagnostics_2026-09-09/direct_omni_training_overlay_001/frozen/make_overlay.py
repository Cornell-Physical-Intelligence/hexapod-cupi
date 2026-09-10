"""Create reviewable training files/plans outside every frozen source."""
import argparse,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).parent
PLAN='robot/hexapod_mkii_length_study/training_plan.json'
ENTRY='tools/train_length_study.py'
def one(text,old,new):
 if text.count(old)!=1:raise ValueError('Entry seam differs: '+old[:60])
 return text.replace(old,new)
def patch_entry(text):
 text=one(text,'        from omni_flat_env import OmniFlatEnv\n        env = OmniFlatEnv(','''        from omni_flat_env import OmniFlatEnv
        if args.mode == "train":
            from curriculum import make_environment
            OmniFlatEnv = make_environment(OmniFlatEnv)
        env = OmniFlatEnv(''')
 old='''        dump_yaml(str(args.output/"agent.yaml"),runner_cfg)
        wrapped = RslRlVecEnvWrapper(env,clip_actions=runner_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped,runner_cfg.to_dict(),log_dir=str(args.output/"policy"),device=env.device)'''
 new='''        run_cfg = runner_cfg.to_dict()
        if args.mode == "train":
            from training_config import configure
            run_cfg = configure(run_cfg, omni["direct_recovery_training"], runner_cfg.max_iterations)
        dump_yaml(str(args.output/"agent.yaml"),run_cfg)
        wrapped = RslRlVecEnvWrapper(env,clip_actions=runner_cfg.clip_actions)
        if args.mode == "train":
            from caps import CapsPairWrapper
            wrapped = CapsPairWrapper(wrapped)
        runner = OnPolicyRunner(wrapped,run_cfg,log_dir=str(args.output/"policy"),device=env.device)'''
 return one(text,old,new)
def plan(old,branch):
 from training_config import options
 p=json.loads(json.dumps(old))
 if p['omni']['overrides']['target_slew_rad_per_20ms']!=.04:raise ValueError('Matched new formal.04 baseline required')
 p['training_num_envs']=128;p['training_iterations']=10
 p['omni']['direct_recovery_training']=options(branch)
 return p
if __name__=='__main__':
 p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--baseline-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 manifest=json.loads((a.baseline_source/'campaign_source_hashes.json').read_text())
 for k,v in manifest.items():
  if hashlib.sha256((a.baseline_source/k).read_bytes()).hexdigest()!=v:raise ValueError('Baseline source mismatch')
 a.output.mkdir(parents=True);entry=patch_entry((a.baseline_source/ENTRY).read_text());(a.output/'train_length_study.py').write_text(entry)
 for b in ['curriculum','caps']:(a.output/(b+'_plan.json')).write_text(json.dumps(plan(json.loads((a.baseline_source/PLAN).read_text()),b),indent=2)+'\n')
 print('Overlay only; root must build/pin new source and authorize each allocation. No launch performed.')
