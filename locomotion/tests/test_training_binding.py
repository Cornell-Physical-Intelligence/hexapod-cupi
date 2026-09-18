"""File-backed admission protocol fixtures, never claimed as physics evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from locomotion.env_config import EnvConfig, MODEL_SHA256, USD_SHA256, sha
from locomotion.admission import require_admission


class TrainingBindingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cfg = EnvConfig(num_envs=32)
        self.identity = {
            "model_sha256": MODEL_SHA256, "usd_sha256": USD_SHA256,
            "physics_source_files": {"env.py": "a"*64, "env_config.py": "b"*64},
            "stance_sha256": "c"*64, "geometry_sha256": "d"*64,
            "geometry_extrema_sha256": "e"*64,
            "physics_config": {"physics_dt": .0025, "decimation": 8, "spacing_m": 2.,
                "target_slew_rad": .04, "action_scale_rad": .35,
                "neutral_joint_position_rad": [0., -.3, .4]*6,
                "reset_root_height_m": .10280231400684256,
                "solver_position_iterations": 32, "solver_velocity_iterations": 0,
                "floor": "80m_two_triangle_mesh_y_equals_x_seam", "material_friction": [1., 1.],
                "restitution": 0., "external_forces_every_iteration": True}}
        self.envelope = {"schema": "hexapod_locomotion_standing_admission_v1",
                         **deepcopy(self.identity), "num_envs": 32}
        self.states, self.reports = {}, {}
        for name, count in (("one", 1), ("batch", 32)):
            self.states[name] = {"schema": "canonical_paper_walk_run_v1", "mode": "diagnostic",
                "status": "completed", "standing_gate_pass": True, "errors": [],
                "identity": {**deepcopy(self.identity), "config": EnvConfig(num_envs=count).declaration()}}
            self.reports[name] = {"num_envs": count, "controls": 1000, "substeps": 8000,
                "all_pass": True, "replicas": [{"env": e, "pass": True,
                    "failed_physical_bounds": [], "quiet": {"pass": True, "failed_bounds": []}}
                    for e in range(count)]}
            self.refresh(name)
        self.path = self.root/"admission.json"
        self.write_envelope()

    def refresh(self, name):
        row = self.envelope.setdefault(name, {})
        for kind, value in (("state", self.states[name]), ("report", self.reports[name])):
            path = self.root/(name+"_"+kind+".json")
            path.write_text(json.dumps(value, indent=2)+"\n")
            row[kind+"_path"] = str(path)
            row[kind+"_sha256"] = sha(path)

    def write_envelope(self):
        self.path.write_text(json.dumps(self.envelope, indent=2)+"\n")

    def accepted(self, cfg=None):
        self.write_envelope()
        return require_admission(self.path, self.identity, self.cfg if cfg is None else cfg)

    def test_matching_state_report_bytes_pass_for_batch_and_video_one(self):
        self.assertEqual(self.accepted()["num_envs"], 32)
        self.assertEqual(self.accepted(EnvConfig(num_envs=1))["num_envs"], 32)

    def test_other_batch_layout_cannot_borrow_admission(self):
        with self.assertRaisesRegex(ValueError, "replica layout"):
            self.accepted(EnvConfig(num_envs=64))

    def test_envelope_source_claim_cannot_override_actual_state(self):
        self.states["batch"]["identity"]["physics_source_files"]["env.py"] = "f"*64
        self.refresh("batch")
        with self.assertRaisesRegex(ValueError, "Native standing identity differs"):
            self.accepted()

    def test_actual_model_pose_and_geometry_mismatch_rejected_after_rehash(self):
        for name in ("one", "batch"):
            for key in ("model_sha256", "usd_sha256", "stance_sha256", "geometry_sha256", "geometry_extrema_sha256"):
                before = deepcopy(self.states[name])
                self.states[name]["identity"][key] = "f"*64
                self.refresh(name)
                with self.subTest(name=name, key=key), self.assertRaisesRegex(ValueError, "Native standing identity differs"):
                    self.accepted()
                self.states[name] = before
                self.refresh(name)
        self.states["batch"]["identity"]["physics_config"]["neutral_joint_position_rad"][1] = 0.
        self.refresh("batch")
        with self.assertRaisesRegex(ValueError, "Native standing identity differs"):
            self.accepted()

    def test_claimed_pass_cannot_hide_failed_replica(self):
        self.reports["batch"]["replicas"][17]["pass"] = False
        self.refresh("batch")
        with self.assertRaisesRegex(ValueError, "Standing quality rejected"):
            self.accepted()

    def test_duplicate_replica_or_contradictory_failed_fields_rejected(self):
        for mutation in ("duplicate", "physical", "quiet", "quiet_pass"):
            before = deepcopy(self.reports["batch"])
            row = self.reports["batch"]["replicas"][17]
            if mutation == "duplicate": row["env"] = 0
            elif mutation == "physical": row["failed_physical_bounds"] = ["six_toe_support"]
            elif mutation == "quiet": row["quiet"]["failed_bounds"] = ["joint_speed"]
            else: row["quiet"]["pass"] = False
            self.refresh("batch")
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "Standing quality rejected"):
                self.accepted()
            self.reports["batch"] = before
            self.refresh("batch")

    def test_short_acquisition_or_missing_replica_rejected(self):
        for name in ("one", "batch"):
            for key, value in (("controls", 999), ("substeps", 7999), ("replicas", [])):
                before = deepcopy(self.reports[name])
                self.reports[name][key] = value
                self.refresh(name)
                with self.subTest(name=name, key=key), self.assertRaisesRegex(ValueError, "Standing quality rejected"):
                    self.accepted()
                self.reports[name] = before
                self.refresh(name)

    def test_native_failure_or_wrong_count_rejected_even_with_pass_report(self):
        for key, value in (("status", "failed"), ("standing_gate_pass", False), ("errors", ["native error"])):
            before = deepcopy(self.states["one"])
            self.states["one"][key] = value
            self.refresh("one")
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Native standing acquisition failed"):
                self.accepted()
            self.states["one"] = before
            self.refresh("one")
        self.states["batch"]["identity"]["config"]["num_envs"] = 1
        self.refresh("batch")
        with self.assertRaisesRegex(ValueError, "Native standing replica count differs"):
            self.accepted()

    def test_changed_report_and_state_bytes_rejected(self):
        for kind in ("report", "state"):
            path = Path(self.envelope["batch"][kind+"_path"])
            before = path.read_bytes()
            path.write_bytes(before+b" ")
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "Missing/changed"):
                self.accepted()
            path.write_bytes(before)

    def test_missing_files_or_unsigned_envelope_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires fresh"):
            require_admission(None, self.identity, self.cfg)
        self.envelope["one"]["state_path"] = str(self.root/"missing.json")
        with self.assertRaisesRegex(ValueError, "Missing/changed"):
            self.accepted()



if __name__ == "__main__":
    unittest.main()
