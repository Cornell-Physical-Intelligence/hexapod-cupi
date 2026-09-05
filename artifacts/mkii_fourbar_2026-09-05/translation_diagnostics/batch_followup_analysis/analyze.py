#!/usr/bin/env python3
"""Verify noisy single-location and explicit-filtering follow-up controls."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[1]/"batch_analysis/analyze.py"
spec=importlib.util.spec_from_file_location("matched_standing_reader",BASE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def file_changes(before,after):
    return {p:{"before":before.get(p),"after":after.get(p)}
            for p in sorted(set(before)|set(after)) if before.get(p)!=after.get(p)}


def compare(before_path,after_path,*,before_row,after_row,filter_change):
    bt,b,bs=base.load(before_path);at,a,ass=base.load(after_path)
    br,ar=bt.report,at.report
    if bt.columns!=at.columns:
        raise ValueError("Named trace layouts differ")
    old,new=copy.deepcopy(br["runtime_manifest"]),copy.deepcopy(ar["runtime_manifest"])
    isolation=new.pop("resolved_collision_isolation",None) if filter_change else None
    if old!=new:
        raise ValueError("Runtime changed beyond the expected explicit collision-isolation descriptor")
    if filter_change and isolation is None:
        raise ValueError("Filtered candidate lacks verified isolation descriptor")
    if not filter_change and (br["contract"]!=ar["contract"] or bs["source_manifest"]!=ass["source_manifest"]):
        raise ValueError("Single-location control changed source")
    before_selected=b[:,before_row:before_row+1] if before_row is not None else b
    after_selected=a[:,after_row:after_row+1] if after_row is not None else a
    if before_selected.shape!=after_selected.shape:
        raise ValueError("Selected trace shapes do not match")
    fields={key:float(abs(before_selected[:,:,[bt.fields[key][n] for n in bt.fields[key]]]
                           -after_selected[:,:,[at.fields[key][n] for n in bt.fields[key]]]).max())
            for key in bt.fields}
    for key in ("target","processed_target","velocity_target","feedforward","terrain_origin_w"):
        if fields[key]!=0:
            raise ValueError(f"Control inputs differ: {key}")
    bi=range(bt.envs) if before_row is None else [before_row]
    ai=range(at.envs) if after_row is None else [after_row]
    initial_before=[br["placement"]["initial_reset_root_positions_m"][i] for i in bi]
    initial_after=[ar["placement"]["initial_reset_root_positions_m"][i] for i in ai]
    if initial_before!=initial_after:
        raise ValueError("Initial root placements differ")
    for category in ("motor_readback","solver_readback"):
        for key in br[category]:
            x=np.asarray(br[category][key])[list(bi)]
            y=np.asarray(ar[category][key])[list(ai)]
            if not np.array_equal(x,y):raise ValueError(f"Backend readback differs: {category}/{key}")
    qidx=[bt.fields['pre_q'][name] for name in bt.joints]
    if not np.array_equal(before_selected[0,:,qidx],after_selected[0,:,qidx]):
        raise ValueError("Initial all30 joint coordinates differ")
    return {
        "before":{"report_path":str(bt.path),"report_sha256":bt.report_hash,"trace_sha256":bt.files[0]["sha256"],
                  "source_commit":bs["source_commit"],"functional_contract_sha256":br["contract"]["sha256"],
                  "source_manifest_sha256":bs["source_manifest"]["sha256"],"selected_row":before_row},
        "after":{"report_path":str(at.path),"report_sha256":at.report_hash,"trace_sha256":at.files[0]["sha256"],
                 "source_commit":ass["source_commit"],"functional_contract_sha256":ar["contract"]["sha256"],
                 "source_manifest_sha256":ass["source_manifest"]["sha256"],"selected_row":after_row},
        "samples_compared_per_environment":bt.samples,"field_max_absolute_differences":fields,
        "physics_runtime_equal_except_declared_isolation":True,
        "backend_motor_and_solver_readbacks_equal":True,
        "all_initial30_joint_coordinates_equal":True,"initial_actual_root_xyz_equal":True,
        "both_supervisors_confirm_unchanged_source_and_exact_owned_cleanup":True,
        "functional_file_changes":file_changes(br["contract"]["files"],ar["contract"]["files"]),
        "new_collision_isolation_descriptor":isolation,
        "after_primary_settled_window":ar["windows"]["settled"],
        "overlapping_world_isolation_proven":False,
    }


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("batch_report",type=Path);p.add_argument("single_report",type=Path)
    p.add_argument("filtered_report",type=Path);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result={"schema":"hexapod.matched_location_and_filter_followups.v1",
            "reader_sha256":base.digest(BASE),
            "single_at_noisy_location":compare(a.batch_report,a.single_report,before_row=7,after_row=0,filter_change=False),
            "explicit_filtering_spaced_world_control":compare(a.batch_report,a.filtered_report,before_row=None,after_row=None,filter_change=True)}
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
