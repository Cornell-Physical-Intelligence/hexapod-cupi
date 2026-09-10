"""Verify or reconstruct exact reference005 from immutable reference004 + delta.

Without --output this checks the virtual merged source, writing nothing.
No simulator, GPU, subprocess, or job launch is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
PARENT='a433e529d29d5360c828b406a3dfd769e078d69fc9deaf03f4fe110eb6fa7a63'
TARGET='c557b71636c8b1e10883ab8a6b2b49f40b661988f9b09faeaefe4821a72844b4'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-source',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    parent=args.parent_source.resolve()
    if sha(parent/'campaign_source_hashes.json')!=PARENT:raise ValueError('Wrong immutable source004')
    old=json.loads((parent/'campaign_source_hashes.json').read_text())
    for name,digest in old.items():
        path=(parent/name).resolve()
        if parent not in path.parents or sha(path)!=digest:raise ValueError('Parent source changed: '+name)
    if {str(p.relative_to(parent)) for p in parent.rglob('*') if p.is_file()}!=set(old)|{'campaign_source_hashes.json'}:
        raise ValueError('Parent contains extra/missing files')
    final_path=HERE/'source_identity/campaign_source_hashes.json'
    if sha(final_path)!=TARGET:raise ValueError('Target manifest changed')
    final=json.loads(final_path.read_text())
    overlay={str(p.relative_to(HERE/'source_overlays')):p for p in (HERE/'source_overlays').rglob('*') if p.is_file()}
    if set(overlay)!={'tools/solver_comparison.py','tools/reference_physics_env.py','tools/screen_contract.py','tools/launch_reference_physics_spark.py'}:
        raise ValueError('Only the reviewed standing-only solver comparison delta is permitted')
    overlay['source_origin.json']=HERE/'source_identity/source_origin.json'
    virtual={**old,**{name:sha(path) for name,path in overlay.items()}}
    if virtual!=final:raise ValueError('Merged source does not equal frozen source005')
    if args.output is not None:
        output=args.output.resolve()
        if output==parent or parent in output.parents or output.exists():raise ValueError('Fresh output outside parent required')
        output.mkdir(parents=True)
        for name in final:
            target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(overlay.get(name,parent/name),target)
            if sha(target)!=final[name]:raise ValueError('Copy mismatch: '+name)
        shutil.copy2(final_path,output/'campaign_source_hashes.json')
    print(json.dumps({'verified':True,'mode':'verify_only' if args.output is None else 'reconstruct',
        'parent_files':len(old),'target_files':len(final),'overlay_files':len(overlay),
        'source_manifest_sha256':TARGET,'parent_unchanged':True},indent=2))


if __name__=='__main__':main()
