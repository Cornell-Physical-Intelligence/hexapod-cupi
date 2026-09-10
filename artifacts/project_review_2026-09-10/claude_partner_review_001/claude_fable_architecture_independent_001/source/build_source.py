"""Build a successor once; all branches/allocations reuse the same immutable bytes."""
import argparse, ast, hashlib, json, shutil
from pathlib import Path
from direct_config import protocol

HERE = Path(__file__).resolve().parent
PARENT = '4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b'
PLAN = 'robot/hexapod_mkii_length_study/training_plan.json'
RUNTIME = ('caps.py', 'caps_ppo.py', 'curriculum.py', 'direct_config.py',
           'direct_training.py', 'direct_stop_evaluation.py', 'direct_quiet_metrics.py', 'direct_contract.py',
           'inputs/RSL_LICENSE')

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def one(text, old, new):
    if text.count(old) != 1: raise ValueError('Exact source seam differs: ' + old[:80])
    return text.replace(old, new)

def patch_entry(text):
    text = one(text, 'parser = argparse.ArgumentParser(description=__doc__)', 'parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)')
    text = one(text, 'AppLauncher.add_app_launcher_args(parser)', '''parser.add_argument("--direct-allocation", choices=("smoke", "pilot"))
parser.add_argument("--direct-branch", choices=("curriculum", "caps"))
parser.add_argument("--direct-evaluation", choices=("constant", "stop"))
AppLauncher.add_app_launcher_args(parser)''')
    text = one(text, 'args = parser.parse_args()', '''args = parser.parse_args()
from direct_config import selection, protocol
direct_selection = selection(args.mode, args.direct_allocation, args.direct_branch, args.direct_evaluation, args.iterations)''')
    text = one(text, '    record = next(r for r in manifest["variants"] if r["variant"]==args.variant)', '''    if plan["omni"].get("direct_recovery_training") != protocol():
        raise ValueError("Plan does not bind the native direct recovery protocol")
    record = next(r for r in manifest["variants"] if r["variant"]==args.variant)''')
    text = one(text, '    if args.mode != "validate":', '    state["direct_selection"] = direct_selection\n    if args.mode != "validate":')
    text = one(text, '    cfg.sim.device = args.device', '''    # Only allocation differs; inherited physical cfg assignments stay exact.
    cfg.scene.num_envs = direct_selection["replicas"]
    cfg.sim.device = args.device''')
    text = one(text, '        from omni_flat_env import OmniFlatEnv\n        env = OmniFlatEnv(', '''        from omni_flat_env import OmniFlatEnv
        if args.mode == "train":
            from curriculum import make_environment
            from direct_training import audited_environment
            OmniFlatEnv = audited_environment(make_environment(OmniFlatEnv))
        env = OmniFlatEnv(''')
    text = one(text, '        runner_cfg.max_iterations = args.iterations or plan["training_iterations"]', '        runner_cfg.max_iterations = direct_selection.get("updates", plan["training_iterations"])')
    text = one(text, '''        dump_yaml(str(args.output/"agent.yaml"),runner_cfg)
        wrapped = RslRlVecEnvWrapper(env,clip_actions=runner_cfg.clip_actions)
        runner = OnPolicyRunner(wrapped,runner_cfg.to_dict(),log_dir=str(args.output/"policy"),device=env.device)''', '''        run_cfg = runner_cfg.to_dict()
        if args.mode == "train":
            from direct_config import configure
            run_cfg = configure(run_cfg, direct_selection)
        dump_yaml(str(args.output/"agent.yaml"),run_cfg)
        wrapped = RslRlVecEnvWrapper(env,clip_actions=runner_cfg.clip_actions)
        if args.mode == "train":
            from caps import CapsPairWrapper
            wrapped = CapsPairWrapper(wrapped)
        runner = OnPolicyRunner(wrapped,run_cfg,log_dir=str(args.output/"policy"),device=env.device)''')
    text = one(text, '''            runner.learn(num_learning_iterations=runner_cfg.max_iterations,init_at_random_ep_len=True)
            checkpoint = args.output/"policy"/"final.pt"
            runner.save(str(checkpoint))''', '''            from direct_training import learn_and_verify
            state["training_receipt"] = learn_and_verify(env, wrapped, runner, direct_selection, args.output)
            checkpoint = args.output/"policy"/"final.pt"''')
    text = one(text, '''                function = record_omni if args.mode=="video" else evaluate_omni
                function(env, runner, plan, args.output, digest(args.checkpoint))''', '''                function = evaluate_omni
                if direct_selection["evaluation"] == "stop":
                    from direct_stop_evaluation import evaluate_stop
                    function = evaluate_stop
                function(env, runner, plan, args.output, digest(args.checkpoint))''')
    return text

def build(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists() or source in output.parents: raise ValueError('Fresh independent output required')
    if sha(source/'campaign_source_hashes.json') != PARENT: raise ValueError('Exact cold589 parent required')
    before = json.loads((source/'campaign_source_hashes.json').read_text())
    actual = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob('*') if p.is_file()}
    actual.pop('campaign_source_hashes.json')
    if actual != before or any(p.is_symlink() for p in source.rglob('*')): raise ValueError('Changed parent input')
    shutil.copytree(source, output)
    (output/'tools/train_length_study.py').write_text(patch_entry((source/'tools/train_length_study.py').read_text()))
    plan = json.loads((source/PLAN).read_text())
    if plan['omni']['overrides']['target_slew_rad_per_20ms'] != .04: raise ValueError('Formal004 parent required')
    plan.update(training_num_envs=1024, training_iterations=50)
    plan['omni']['direct_recovery_training'] = protocol()
    (output/PLAN).write_text(json.dumps(plan, indent=2)+'\n')
    for name in RUNTIME:
        (output/'tools'/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(HERE/name, output/'tools'/name)
    origin = {'schema': protocol()['schema'], 'cold_parent_source_sha256': PARENT,
              'cold_parent_origin_sha256': sha(source/'source_origin.json'),
              'preserved_runtime': 'legacy exact16; no production/source009 physics substitution',
              'runtime_overlays': {name: sha(HERE/name) for name in RUNTIME},
              'scope': 'Bounded new training, never automatic admission or continuation'}
    (output/'source_origin.json').write_text(json.dumps(origin, indent=2)+'\n')
    manifest = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='campaign_source_hashes.json'}
    (output/'campaign_source_hashes.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return {'source_manifest_sha256': sha(output/'campaign_source_hashes.json'), 'files': len(manifest),
            'plan_sha256': sha(output/PLAN), 'origin_sha256': sha(output/'source_origin.json')}

if __name__ == '__main__':
    p=argparse.ArgumentParser(allow_abbrev=False); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(build(a.source,a.output),indent=2))
