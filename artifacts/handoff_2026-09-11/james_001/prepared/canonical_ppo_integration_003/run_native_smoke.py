#!/usr/bin/env python3
"""Fresh canonical native standing followed by bounded405/408 PPO integration only."""
import argparse,json,sys
from pathlib import Path
from canonical_direct_ppo.native_contract import verify_inputs,finish
from canonical_direct_ppo.native_entry_adapter import load,exact_helper_imports
from canonical_direct_ppo.smoke_config import SCHEMA


def main(argv=None):
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ['asset','admission','standing-source','standing-one','standing32','bindings','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--headless',action='store_true')
    args=p.parse_args(argv);identity=verify_inputs(args)
    if args.preflight_only:print(json.dumps(identity,sort_keys=True));return 0
    if args.device!='cuda:0'or not args.headless:raise ValueError('Declared fresh native GPU/headless composition required')
    def native_verify(current):
        if current.num_envs!=32 or any(Path(getattr(current,k)).resolve()!=Path(getattr(args,k)).resolve()for k in ['asset','admission','standing_one','output']):raise ValueError('Native entry arguments changed')
        return verify_inputs(args)
    def native_finish(output,state,current,actual_identity,app):
        return finish(output,state,args,actual_identity,app)
    def callback(session,model,out,state,errors,current):
        from native_support import save
        from standing_score import score
        from canonical_direct_ppo.neutral_prefix import snapshot
        if errors:raise ValueError('Native errors before neutral-prefix admission')
        state['fresh_neutral_prefix']=snapshot(session,out/'neutral_prefix',score,save)
        state['checks']['fresh_neutral_prefix_admitted']=True
        save(out/'entry_composition.json',composition)
        state['quality_admitted']=False;state['Stage2_complete']=False;save(out/'state.json',state)
        from canonical_direct_ppo.run_bound_smoke import run
        result=run(session,model,args.standing_source,args.standing_one,args.standing32,args.bindings,out/'learner',device=args.device,native_errors=errors)
        if errors:raise ValueError('Native errors during learner')
        state['learner_updates_completed']=result['updates_completed'];state['learner_integration_completed']=True
        state['checks']['learner_2_updates_strict_reload']=True
    with exact_helper_imports(args.standing_source):
        native_main,composition=load(args.standing_source,native_verify,native_finish,callback,SCHEMA)
        native_argv=['--asset',str(args.asset),'--admission',str(args.admission),'--output',str(args.output),
            '--num-envs','32','--standing-one',str(args.standing_one),'--device',args.device,'--headless']
        return native_main(native_argv)

if __name__=='__main__':sys.exit(main())
