"""Small synthetic alignment, arithmetic and independently timed event regressions."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("prefix_analyzer", Path(__file__).with_name("analyze.py"))
analyzer = importlib.util.module_from_spec(spec); spec.loader.exec_module(analyzer)
KIN = json.loads((analyzer.ROOT/"configs/mkii_fourbar_v3_kinematics.json").read_text())
NAMES, JOINTS = KIN["active_joint_names"], KIN["tree_joint_names"]


def add_file(directory, name, values, columns, first, field):
    path = directory/name
    np.savez_compressed(path, values=values, columns=np.array(columns))
    return {"file": name, "sha256": analyzer.digest(path), "shape": list(values.shape), field: first,
            "segment": {"phase": "driven", "motors": ["lm_tibia_lever_pivot"], "offset_rad": .04, "steps": len(values)//16}}


def phases_fixture():
    q = np.zeros((2500, 32, 18), np.float32)
    report = {"active_motor_names": NAMES, "num_envs": 32, "individual_phase_means": [], "individual_motor_response_by_env": {}}
    baseline = {"individual_motor_positive_minus_negative_rad": {}}
    for motor, name in enumerate(NAMES[:15]):
        for sign in (0, 1):
            start = 1000+100*motor+50*sign
            q[start:start+45, :, motor] = 987.  # Any inclusion before the end hold makes the test fail.
            q[start+45:start+50, :, motor] = (np.linspace(-.883, .731, 32, dtype=np.float32)[None, :]
                +np.arange(5, dtype=np.float32)[:, None]*.0073 + (.019 if sign == 0 else -.017))
            if sign == 1: q[start+45:start+50, 11, motor] += .1
        means, delta, minimum, worst = analyzer.float32_phase_response(q, motor, motor)
        for sign in (0, 1):
            report["individual_phase_means"].append({"motor": name, "first_control_step": 1000+100*motor+50*sign,
                "offset_rad": .04 if sign == 0 else -.04, "control_steps": 50, "mean_control_indices": [45,46,47,48,49],
                "mean_dtype": "float32", "mean_joint_position_rad": means[sign]})
        report["individual_motor_response_by_env"][name] = {"positive_minus_negative_rad": delta}
        baseline["individual_motor_positive_minus_negative_rad"][name] = minimum
    return q, report, baseline


class PrefixAnalysisTests(unittest.TestCase):
    def test_last_five_float32_reduction_and_environment_minimum(self):
        q, report, baseline = phases_fixture()
        results = analyzer.phase_analysis(q, report, baseline)
        self.assertEqual(len(results), 15)
        for row in results:
            self.assertEqual(row["worst_env_index"], 11)
            self.assertEqual(row["report_mean_max_abs_recompute_difference_rad"], [0., 0.])
            self.assertTrue(row["exact_campaign008_minimum_match"])
        report["individual_phase_means"][26]["first_control_step"] += 1
        with self.assertRaises(ValueError): analyzer.phase_analysis(q, report, baseline)

    def test_shifted_end_hold_changes_response_and_cpu_gpu_difference_is_disclosed(self):
        q, report, baseline = phases_fixture()
        q[2345, 11, 13] += .25
        row = analyzer.phase_analysis(q, report, baseline)[13]
        self.assertGreater(row["report_mean_max_abs_recompute_difference_rad"][0], .049)
        self.assertFalse(row["exact_campaign008_minimum_match"])

    def test_sparse_reader_preserves_global_sample_index_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            report = {"num_envs": 2, "numerical_recipe": {"decimation": 16}}
            values = np.zeros((4800, 2, 2), np.float32)
            report["trace_files"] = [add_file(directory, "trace_000.npz", values, ["q/a", "qd/a"], 35200, "first_physics_sample")]
            traces = analyzer.TraceSet(directory/"report.json", report, "trace")
            self.assertEqual(traces.first, 35200)
            self.assertEqual(traces.end, 40000)
            self.assertEqual(traces.read(traces.records[0]).shape, values.shape)
            bad = copy.deepcopy(report); bad["trace_files"][0]["first_physics_sample"] = 0
            with self.assertRaises(ValueError): analyzer.TraceSet(directory/"report.json", bad, "trace")
            bad = copy.deepcopy(report); bad["trace_files"][0]["file"] = "../trace_000.npz"
            with self.assertRaises(ValueError): analyzer.TraceSet(directory/"report.json", bad, "trace")
            (directory/"trace_000.npz").write_bytes(b"tampered")
            with self.assertRaises(ValueError): traces.read(traces.records[0])

    def test_control_coverage_requires_all_2500_rows_and_unique_named_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            report = {"num_envs": 2, "numerical_recipe": {"decimation": 16}}
            report["control_trace_files"] = [add_file(directory, "control_000.npz", np.zeros((2500,2,2),np.float32), ["q/a","q/a"],0,"first_control_step")]
            traces = analyzer.TraceSet(directory/"report.json", report,"control")
            with self.assertRaises(ValueError): traces.read(traces.records[0])
            report["control_trace_files"][0]["shape"][0] = 2499
            with self.assertRaises(ValueError): analyzer.TraceSet(directory/"report.json", report,"control")

    def test_braking_intent_is_distinct_from_delivered_torque_and_limit(self):
        speed = np.array([60., 60., -60., 10.])
        demand = np.array([-30., 30., 30., -4.])
        applied = np.array([0., 0., 0., -2.])
        result = analyzer.motor_dynamics(speed, demand, applied, np.array([0.,0.,0.,2.]), 480*np.pi/30)
        self.assertEqual(result["overspeed_with_opposing_demand"].tolist(), [True,False,True,False])
        self.assertEqual(result["raw_minus_applied_abs_nm"].tolist(), [30.,30.,30.,2.])
        self.assertEqual(result["applied_above_recorded_limit_nm"].tolist(), [0.,0.,0.,0.])

    def test_shuffled_passive_velocity_mapping_has_no_position_offset(self):
        names = ["rod", "source", "knee"]
        kin = {"passive_relations": {"rod": {"source_joint":"source","multiplier":-1.,"offset_rad":.4},
                                     "knee":{"source_joint":"source","multiplier":1.,"offset_rad":.3}}}
        v = np.array([[[ -2.,2.,2.]], [[-2.,2.,3.]]])
        result = analyzer.passive_residual(v,names,kin)
        self.assertEqual(result.tolist(), [[[0.,0.]],[[0.,1.]]])

    def test_control_event_time_is_last_substep_endpoint_and_keeps_motor_identity(self):
        fields = [(key,NAMES) for key in ("joint_vel","computed_torque","applied_torque","instantaneous_limit_nm")]
        fields += [("tree_joint_vel",JOINTS), ("body_contact_force_w",[f"{leg}_tibia_{axis}" for leg in analyzer.LEGS for axis in "xyz"])]
        columns = [f"{key}/{name}" for key,names in fields for name in names]
        values = np.zeros((50,2,len(columns)),np.float32)
        name = "lr_tibia_lever_pivot"
        values[7,1,columns.index("joint_vel/"+name)] = 70.
        class Reader:
            def block(self, values, key, names): return values[..., [columns.index(f"{key}/{name}") for name in names]]
            def vectors(self, values, key, names): return self.block(values,key,[f"{name}_{axis}" for name in names for axis in "xyz"]).reshape(50,2,len(names),3)
        report = {"active_motor_names":NAMES,"joint_names":JOINTS,"numerical_recipe":{"decimation":16},
                  "runtime_manifest":{"policy_dt_s":.02,"motor_contract":{"configuration":{"vendor":{"no_load_rpm":480.,"peak_output_torque_nm":5.5}}}}}
        events = {}
        analyzer.control_events(Reader(),values,{"first_control_step":1000,"file":"control_001.npz","sha256":"synthetic"},report,KIN,events)
        event = events["first_endpoint_speed_above_no_load"]
        self.assertEqual(event["control_step"],1007)
        self.assertEqual(event["last_physics_sample"],1008*16-1)
        self.assertEqual(event["post_time_s"],1008*.02)
        self.assertEqual(event["environment"],1)
        self.assertEqual(event["name"],name)

    def test_metric_peaks_and_first_events_are_chronological_not_assumed_coincident(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fields = [(key, JOINTS) for key in ("pre_q","post_q","pre_qd","post_qd","direct_pre_qd")]
            fields += [(key,NAMES) for key in ("target","processed_target","p_term","d_term","feedforward","demand","applied","instantaneous_limit","headroom")]
            fields += [(key,[f"{leg}_{a}" for leg in analyzer.LEGS for a in "xyz"]) for key in ("hinge_gap_local","hinge_relative_point_velocity_local","foot_force_w")]
            columns = [f"{key}/{name}" for key,names in fields for name in names]
            values = np.zeros((4800,2,len(columns)),np.float32)
            for leg in analyzer.LEGS: values[:,:,columns.index(f"foot_force_w/{leg}_z")] = 2.
            for name in NAMES: values[:,:,columns.index("instantaneous_limit/"+name)] = 5.5
            name = "lm_tibia_lever_pivot"
            values[11,1,columns.index("pre_qd/"+name)] = 60.
            values[11,1,columns.index("demand/"+name)] = -20.
            values[17,1,columns.index("demand/"+name)] = 85.
            values[19,0,columns.index("hinge_gap_local/lr_x")] = .0003
            values[23,1,columns.index("applied/"+name)] = 5.5
            values[11,1,columns.index("post_q/"+name)] = .00125*5.
            record = add_file(directory,"trace_000.npz",values,columns,35200,"first_physics_sample")
            report = {"num_envs":2,"trace_files":[record],"active_motor_names":NAMES,"joint_names":JOINTS,
                "numerical_recipe":{"decimation":16,"physics_dt_s":.00125},"runtime_manifest":{"motor_contract":{"configuration":{"vendor":{"no_load_rpm":480.,"peak_output_torque_nm":5.5}}}}}
            traces = analyzer.TraceSet(directory/"report.json",report,"trace")
            events, trajectories = analyzer.detailed_analysis(traces,report,KIN,{"lf":0,"lm":1,"lr":0})
            mapping = {row["event"]:row for row in events}
            self.assertEqual(mapping["first_pre_speed_above_no_load_with_opposing_demand"]["physics_sample"],35211)
            self.assertEqual(mapping["peak_raw_demand_nm"]["physics_sample"],35217)
            self.assertEqual(mapping["peak_C_pin_gap_m"]["physics_sample"],35219)
            self.assertEqual(mapping["peak_applied_torque_nm"]["physics_sample"],35223)
            self.assertEqual([r["physics_sample"] for r in events],sorted(r["physics_sample"] for r in events))
            onset = mapping["first_pre_speed_above_no_load_with_opposing_demand"]["nearby_same_environment_samples"][2]
            self.assertAlmostEqual(onset["motors"][name]["q_finite_difference_interval_average_rad_s"],5.,places=5)
            self.assertEqual(onset["motors"][name]["pre_qd"],60.)
            self.assertEqual(len(trajectories),3)


if __name__ == "__main__": unittest.main()
