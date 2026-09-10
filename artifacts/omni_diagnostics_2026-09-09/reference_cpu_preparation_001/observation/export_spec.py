"""Compute slices and source/profile identity from the validated CPU fixture."""
import argparse
import json
from pathlib import Path
from observation import ObservationBuilder,Limits,SOURCE_CONTRACT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('formal_004','diagnostic_003'),default='formal_004')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.out is not None and args.out.exists():raise FileExistsError('Use a fresh spec path')
    packet=json.loads(Path(__file__).with_name('fixtures.json').read_text())['landing']
    limits=Limits(args.profile,.02,.25,2. if args.profile=='formal_004' else 1.5)
    b=ObservationBuilder(packet['joint_names_runtime'],SOURCE_CONTRACT['nominal_joint_positions'],1,limits=limits)
    b.reset([0],[packet['episode_id']],[0.])
    spec=b.build([packet])['schema']
    text=json.dumps(spec,indent=2,sort_keys=True)+'\n'
    if args.out is None:print(text,end='')
    else:args.out.write_text(text)


if __name__=='__main__':main()
