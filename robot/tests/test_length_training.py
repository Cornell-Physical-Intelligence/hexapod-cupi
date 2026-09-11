"""CPU checks for admission inputs and honest comparison summaries."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("rank",ROOT/"experiments/c_length_study/tools/rank_length_study.py")
rank=importlib.util.module_from_spec(spec);spec.loader.exec_module(rank)
sys.path.insert(0,str(ROOT/"tools"))
from experiments.c_length_study.tools.launch_length_training_spark import live_competitors


class TrainingPlanTests(unittest.TestCase):
    def test_gpu_shutdown_race_does_not_hide_live_foreign_work(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for pid,group in (("12","/system.slice/docker-owned.scope"),("13","/weather.service")):
                path=root/pid;path.mkdir();(path/"cgroup").write_text("0::"+group)
            found=live_competitors("11, exited\n12, own\n13, weather",set(),"docker-owned",root)
            self.assertEqual([v["process"] for v in found],["13, weather"])

    def test_every_geometry_has_an_eligible_stance_with_action_headroom(self):
        package=ROOT/"robot/hexapod_mkii_length_study"
        plan=json.loads((package/"training_plan.json").read_text())
        self.assertEqual(len(plan["variants"]),49)
        for variant,record in plan["variants"].items():
            self.assertTrue(record["stances"],variant)
            limits={j.get("name"):tuple(float(j.find("limit").get(k)) for k in ("lower","upper"))
                    for j in ET.parse(package/"urdf"/(variant+".urdf")).getroot().findall("joint")}
            for stance in record["stances"]:
                self.assertTrue(stance["six_foot_geometry_eligible"])
                self.assertLessEqual(stance["max_abs_hold_torque_nm"],1.3)
                self.assertGreaterEqual(stance["root_height_at_contact_m"],.070)
                self.assertGreaterEqual(stance["nonfoot_mesh_vertex_clearance_m"],.005)
                for joint,q in stance["joint_positions_rad"].items():
                    low,high=limits[joint];margin=(high-low)*.025
                    self.assertGreaterEqual(q-.2,low+margin)
                    self.assertLessEqual(q+.2,high-margin)

    def test_failed_motor_or_fall_constraints_cannot_win_pareto_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name,scale,bad in (("good",1.,None),("dominated",2.,None),("overloaded",.1,"saturation_fraction"),("falling",.1,"fall_fraction"),("standing_still",.1,"abs_forward_error_mps")):
                speeds=[]
                for speed in (.1,.2,.3):
                    row={k:.001*scale for k in rank.METRICS}
                    row.update(command_mps=speed,fall_fraction=0.,nonfoot_contact_fraction=0.)
                    if bad:row[bad]=speed if bad=="abs_forward_error_mps" else .5
                    speeds.append(row)
                path=root/name/"evaluate";path.mkdir(parents=True)
                (path/"evaluation.json").write_text(json.dumps({"variant":name,"complete":True,"speeds":speeds}))
            result=rank.summarize(root)
            self.assertEqual(result["pareto_candidates"],["good"])
            self.assertFalse(result["all_49_evaluated"])

    def test_partial_speed_evaluations_are_not_ranked(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/"partial"/"evaluate";path.mkdir(parents=True)
            (path/"evaluation.json").write_text(json.dumps({"complete":True,"speeds":[]}))
            self.assertEqual(rank.summarize(root)["completed_evaluations"],0)


if __name__=="__main__":unittest.main()
