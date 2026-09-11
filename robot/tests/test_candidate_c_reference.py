"""Independent geometry checks on the full robot's stepping reference."""
import json
import hashlib
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools"))
from experiments.c_length_study.tools import screen_length_mechanics as s

class CandidateReferenceTests(unittest.TestCase):
    def test_reference_preserves_asset_and_joint_identity(self):
        package=ROOT/"robot/hexapod_mkii_length_study"
        path=ROOT/"artifacts/length_study_2026-09-09/candidate_c_training_inputs/candidate_c_reference.json"
        r=json.loads(path.read_text());xml=ET.parse(package/"urdf/f050_t060.urdf").getroot()
        self.assertEqual(r["urdf_sha256"],hashlib.sha256((package/"urdf/f050_t060.urdf").read_bytes()).hexdigest())
        self.assertEqual(set(r["joint_names"]),{j.get("name") for j in xml.findall("joint")})
        q=np.array(r["positions_rad"]);self.assertEqual(q.shape,(256,18))
        for i,name in enumerate(r["joint_names"]):
            limit=xml.find(f"joint[@name='{name}']/limit")
            self.assertTrue(np.all(q[:,i]>=float(limit.get("lower"))))
            self.assertTrue(np.all(q[:,i]<=float(limit.get("upper"))))

    def test_stance_feet_cancel_commanded_forward_body_velocity(self):
        package=ROOT/"robot/hexapod_mkii_length_study"
        r=json.loads((ROOT/"artifacts/length_study_2026-09-09/candidate_c_training_inputs/candidate_c_reference.json").read_text())
        m=json.loads((package/"manifest.json").read_text());xml=ET.parse(package/"urdf/f050_t060.urdf").getroot()
        robot=s.Robot(xml,m["link_joint_mapping"],s.mesh_clouds(xml,package))
        q=np.array(r["positions_rad"]).reshape(256,6,3);toes=np.array(r["stance"]["toes"])
        # Reference is sampled in the body frame: stance foot +Y velocity
        # must cancel anatomical forward body velocity (-Y).
        for i,leg in enumerate(robot.legs):
            feet=np.array([leg.fk(row[i],toes[i])["foot"] for row in q])
            phase=((np.arange(256)+.5)/256+s.PHASE["tripod"][i])%1
            stance=(phase>.02)&(phase<.63)
            velocity=(np.roll(feet,-1,axis=0)-np.roll(feet,1,axis=0))*128*1.3
            self.assertLess(np.max(np.abs(velocity[stance]-[0,.2,0])),2e-5)

if __name__=="__main__":unittest.main()
