#!/usr/bin/env python3
"""Actual full native-main import/preflight path, with a labeled CPU identity stub.
No simulator allocation or claim of actual native input admission.
"""
from pathlib import Path
import argparse,json,sys
from canonical_direct_ppo.native_entry_adapter import load


def main():
 p=argparse.ArgumentParser();p.add_argument('--standing-source',type=Path,required=True);args=p.parse_args()
 events=[]
 def verified_fixture(a):
  events.append({'num_envs':a.num_envs,'preflight_only':a.preflight_only})
  return {'CPU_ENTRYPOINT_FIXTURE_ONLY':True}
 run,composition=load(args.standing_source,verified_fixture,None,None,'CPU_ENTRYPOINT_FIXTURE')
 result=run(['--asset','/CPU_ASSET','--admission','/CPU_ADMISSION','--output','/CPU_OUTPUT','--num-envs','32','--standing-one','/CPU_ONE','--preflight-only'])
 assert result==0 and events==[{'num_envs':32,'preflight_only':True}]
 prohibited=[x for x in sys.modules if x in ('torch','numpy')or x.startswith('isaaclab')or x.startswith('omni.')]
 assert not prohibited,prohibited
 print(json.dumps({'test':'exact_full_source005_load_and_preflight_only','passed':True,'native_inputs_stubbed':True,'native_executed':False,'new_native_output_created':False,'prohibited_imports':prohibited,'composition':composition},sort_keys=True))
if __name__=='__main__':main()
