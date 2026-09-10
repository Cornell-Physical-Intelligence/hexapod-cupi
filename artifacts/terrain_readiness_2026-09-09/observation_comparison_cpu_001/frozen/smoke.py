"""Small CPU execution example; no Isaac application or actor import."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from terrain_fixture_checks import load_catalog
from hexapod_terrain.support_queries import TerrainSupportQueries
from course_queries import CourseQueries, UnresolvedRuntime
from observation_modes import TerrainObservationModes
from perception_replay import WorldPoints


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--fixture-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Fresh output required')
    torch.set_num_threads(1)
    records=load_catalog(args.catalog,[args.fixture_id])
    oracle=TerrainSupportQueries(records,device='cpu',dtype=torch.float64)
    runtime=UnresolvedRuntime(actor_sha256=None,source_sha256=None,physics_sha256=None)
    courses=CourseQueries(oracle,fixture_indices=[0],origins_world=[[0,0,0]],yaw_rad=[0.],
                          boundary_margin_m=.02,runtime=runtime)
    modes=TerrainObservationModes(courses,map_origins_xy=[[-1.5,-1.5]],map_sizes_xy=[[3,3]],
        world_frame='odom',clock_id='cpu_smoke',reset_time_s=0.,registration='oracle_localization')
    pose=np.array([[0.,-1,0,0],[1,0,0,0],[0,0,1,.13653251856352808],[0,0,0,1]])
    # One explicitly synthetic observed point, with declared uncertainty. This
    # exercises causal delivery; it does not claim a calibrated sensor/noise fit.
    q=courses.query_world([0],[[[.01,.01,0.]]])
    point=WorldPoints(np.array([[.01,.01,q.height_m.item()]]),np.array([.003**2]),np.array([1.]),
                      'odom','synthetic_fixture_point','cpu_example_only',{'synthetic':True})
    modes.enqueue(0,point,receive_time_s=1.04,clock_id='cpu_smoke',reset_epoch=0,
                  provenance='synthetic_corrupted_map')
    before=modes.packet(0,'corrupted_map',pose,now_s=1.03)
    packets={mode:modes.packet(0,mode,pose,now_s=1.04) for mode in sorted(modes.MODES)}
    expired=modes.packet(0,'corrupted_map',pose,now_s=1.250001)
    result={'scope':'CPU adapter execution only','runtime':runtime.receipt(),'fixture_id':args.fixture_id,
            'before_receipt_usable_cells':int(before.channels[...,1].sum()),
            'modes':{mode:{'shape':list(packet.channels.shape),'usable_cells':int(packet.channels[...,1].sum()),
                           'geometry_truth_exposed':packet.geometry_truth is not None,
                           'registration':packet.registration,'provenance':packet.provenance}
                     for mode,packet in packets.items()},
            'expired_usable_cells':int(expired.channels[...,1].sum()),'physics_started':False}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
