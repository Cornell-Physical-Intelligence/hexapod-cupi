"""Evaluation allocation, uninterrupted timing and incomplete-prefix fixtures."""
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import torch
from experiments.paper_walk import evaluate as run
from experiments.paper_walk import evaluation as scoring
from experiments.paper_walk.env_config import EnvConfig


class FakeNative:
    def __init__(self, terminate_at=None):
        self.num_envs=1;self.device="cpu";self.cfg=run.EvaluationEnvConfig(num_envs=1)
        self.commands=torch.zeros(1,3);self.current={"toe_world":torch.zeros(1,6,3)};self.counter=0;self.reset_count=0
        self.capture=None;self.native_errors=[];self.terminate_at=terminate_at;self.steps=0
        self.sim=SimpleNamespace(get_physics_step_count=lambda:self.counter)

    def reset(self):
        self.reset_count+=1
        return self._observations(self.current)

    def _observations(self, state):
        obs=torch.zeros(1,231);obs[:,210:213]=self.commands
        return {"obs":obs,"critic":torch.cat((obs,torch.zeros(1,3)),-1),"amp":torch.zeros(1,61)}

    def step(self, action):
        self.steps+=1;self.counter+=8
        self.capture.count+=8
        return {**self._observations(self.current),"terminated":torch.tensor([self.steps==self.terminate_at]),"truncated":torch.zeros(1,dtype=torch.bool)}

    def verify_native_recipe(self, phase):
        pass


class FakeCapture:
    def __init__(self, env, output, geometry):
        self.env=env;self.count=0;env.capture=self
        self.last_control=[{"distal_contact":np.ones((1,6),bool)}]

    def control_record(self, result, command, control):
        n=1;root=np.zeros((n,7));root[:,2]=.1;root[:,6]=1.
        row={"root_pose_xyzw":root,"time_s":np.array((control+1)*.02),"command":np.asarray(command),
             "requested_command":np.asarray(command),"velocity_navigation_mps":np.zeros((n,3)),
             "velocity_world_mps":np.zeros((n,3)),"gyro_body_rad_s":np.zeros((n,3)),
             "terminated":result["terminated"].numpy(),"truncated":result["truncated"].numpy(),"reset":np.zeros(n,bool),
             "minimum_non_toe_floor_m":np.full(n,.01)}
        for key in ("joint_position_rad","joint_velocity_rad_s","joint_target_rad","computed_torque_nm","applied_torque_nm","saturation_count_400hz"):
            row[key]=np.zeros((n,18))
        for key in ("applied_torque_abs_max_400hz","nonfoot_contact","nonfoot_contact_count_400hz","missing_six_toe_count_400hz"):
            row[key]=np.zeros(n)
        return row

    def close(self):
        self.env.capture=None
        return {"steps":self.count,"failure":None,"joint_bound_violation_steps":[0],"speed_bound_violation_steps":[0],
                "nonfoot_contact_steps_400hz":[0],"maximum_applied_nm":[0.],
                "minimum_non_toe_floor_m":[.01],"minimum_plate_height_m":[.1]}


class EvaluationOrchestrationTests(unittest.TestCase):
    def test_long_timeout_changes_no_physical_config(self):
        cfg=run.EvaluationEnvConfig(num_envs=1)
        values=asdict(cfg)
        self.assertEqual(values.pop("episode_seconds"),90.)
        ordinary=asdict(EnvConfig(num_envs=1,episode_seconds=60.));ordinary.pop("episode_seconds")
        self.assertEqual(values,ordinary)
        for kwargs in ({"episode_seconds":70.},{"physics_dt":.005},{"target_slew_rad":.05}):
            with self.assertRaises(ValueError):run.EvaluationEnvConfig(**kwargs)

    def test_fixed_full_manifest_and_pilot_stays_incomplete(self):
        full=run.case_manifest()
        self.assertEqual(len(full),96)
        self.assertEqual(len(set(x["case_id"] for x in full)),96)
        self.assertEqual(sum(x["profile"]=="omni_static" for x in full),77)
        self.assertEqual(sum(x["profile"]=="stop_to_stand" for x in full),12)
        pilot=run.selected_cases(run.EvaluationConfig(static_indices=(0,),include_transitions=False,include_quiet=False,
            include_stops=False,include_formal=False,record_video=False))
        report=scoring.summarize_suite([{"case_id":pilot[0]["case_id"],"pass":True}],required_cases=[x["case_id"] for x in full])
        self.assertFalse(report["stage2_complete"])
        self.assertEqual(len(report["missing_cases"]),95)

    def test_transition_keeps_all14_segments_in_one70_second_trial(self):
        case=next(x for x in run.case_manifest() if x["profile"]=="transition")
        self.assertEqual(case["controls"],3500)
        offset=0
        for name,duration,command in scoring.transition_sequence():
            count=round(duration/.02)
            for local in (0,count//2,count-1):
                self.assertEqual(run.command_at(case,offset+local),scoring.trajectory_command(name,local*.02,duration,command))
            offset+=count
        self.assertEqual(offset,3500)

    def test_stop_command_uses_fixed8_second_boundary(self):
        case=next(x for x in run.case_manifest() if x["case_id"]=="stop:left")
        self.assertEqual(case["controls"],1050)
        self.assertEqual(run.command_at(case,399),[0.,.1,0.])
        self.assertEqual(run.command_at(case,400),[0.,0.,0.])
        self.assertEqual(run.command_at(case,1049),[0.,0.,0.])

    def test_no_missing_or_unselected_video_substitute(self):
        with self.assertRaisesRegex(ValueError,"video case"):
            run.selected_cases(run.EvaluationConfig(static_indices=(0,)))
        with self.assertRaises(ValueError):
            run.selected_cases(run.EvaluationConfig(static_indices=(0,0),record_video=False))

    def test_failed_terminal_prefix_preserved_without_reset(self):
        native=FakeNative(terminate_at=2)
        case={"case_id":"static:stand","profile":"omni_static","command":[0.,0.,0.],"controls":1000}
        with TemporaryDirectory() as temporary:
            geometry=Path(temporary)/"geometry.npz";geometry.write_bytes(b"CPU fixture")
            with patch.object(run,"ExactEvaluationCapture",FakeCapture):
                report=run.run_batch(native,lambda obs:torch.zeros(1,18),[case],Path(temporary)/"run",geometry,
                                     checkpoint_sha256="CPU fixture",source_sha256="CPU fixture")
            self.assertEqual(native.reset_count,1)
            self.assertEqual(native.steps,2)
            self.assertEqual(report["controls"],2)
            self.assertEqual(report["recorded_physics_steps"],16)
            self.assertEqual(report["failure_kind"],"native_terminal_prefix")
            self.assertFalse(report["results"][0]["pass"])
            with np.load(Path(temporary)/"run/control_trace.npz") as raw:
                self.assertTrue(raw["terminated"][1,0])
                self.assertEqual(raw["reset"].sum(),0)
                self.assertEqual(raw["policy_observation"].shape,(2,1,231))
                np.testing.assert_array_equal(raw["policy_observation"][:,:,210:213],raw["command"])
                self.assertEqual(raw["toe_xyz_world_m"].shape,(2,1,6,3))

    def test_native_capture_retains_mid_hold_events_and_com_frame(self):
        native=FakeNative()
        native.current={"q":torch.zeros(1,18)}
        native.geometry_meta={};native.native_body_names=["body"];native.sensor_map=[]
        native.contact=SimpleNamespace(get_contact_data=lambda dt:[])
        native.native_readback={"limits":np.tile([[-1.,1.]],(1,18,1)).tolist(),
                               "native_max_velocity":np.full((1,18),20.).tolist()}
        fake_geometry=SimpleNamespace(clearance=lambda poses:(np.zeros(1),np.full(1,.01)))
        root=np.array([[0.,0.,.1,0.,0.,np.sqrt(.5),np.sqrt(.5)]],np.float32)
        state={"q":torch.zeros(1,18),"dq":torch.zeros(1,18),"root":root,
               "root_velocity":np.array([[1.,2.,3.,0.,0.,.2]],np.float32),"link":root[:,None,:],
               "toe_world":torch.zeros(1,6,3),"toe_body":torch.zeros(1,6,3)}
        def classify(*args):
            feet=np.ones((1,6),bool)
            if native.counter==3:feet[0,1]=False
            return {"distal_contact":feet,"distal_force_world":np.zeros((1,6,3)),
                    "nonfoot_contact":np.array([native.counter==5]),
                    "nonfoot_force_world":np.zeros((1,4,3)),"patches":[]}
        with TemporaryDirectory() as temporary:
            geometry=Path(temporary)/"geometry.npz";np.savez(geometry,cloud=np.zeros(1))
            with patch.object(run,"_DiagnosticGeometry",return_value=fake_geometry),patch.object(run,"_classify_patches",side_effect=classify):
                capture=run.ExactEvaluationCapture(native,Path(temporary)/"raw",geometry)
                for substep in range(8):
                    native.counter+=1
                    q=np.zeros((1,18),np.float32);target=q.copy()
                    if substep==3:target[0,0]=.2
                    requested,applied,ceiling=run._diagnostic_servo(q,q,target,np.full(18,12.,np.float32),np.asarray(run.KD,np.float32))
                    capture(native,state,q,q,requested,applied,ceiling,target,None,substep,applied)
                result={"terminated":torch.zeros(1,dtype=torch.bool),"truncated":torch.zeros(1,dtype=torch.bool)}
                row=capture.control_record(result,[[0.,0.,0.]],0)
                receipt=capture.close()
            self.assertEqual(row["missing_six_toe_count_400hz"].tolist(),[1])
            self.assertEqual(row["nonfoot_contact_count_400hz"].tolist(),[1])
            self.assertEqual(row["nonfoot_contact"].tolist(),[False])
            self.assertEqual(row["saturation_count_400hz"][0,0],1)
            self.assertAlmostEqual(row["computed_torque_abs_max_400hz"][0],2.4,places=6)
            self.assertEqual(row["computed_torque_nm"].max(),0.)
            np.testing.assert_allclose(row["velocity_navigation_mps"],[[1.,2.,3.]],atol=1e-6)
            self.assertEqual(receipt["steps"],8)
            self.assertEqual(receipt["nonfoot_contact_steps_400hz"],[1])
            self.assertFalse(run._native_contact_screen(receipt,0,{"profile":"omni_static","command":[.05,0.,0.]})["pass"])
            with np.load(Path(temporary)/"raw/substeps_000.npz") as saved:
                self.assertEqual(len(saved["sequence"]),8)
                self.assertFalse(saved["distal_contact"][2,0,1])

    def test_custom_diagnostic_does_not_replace_required_cases(self):
        native=FakeNative(terminate_at=2)
        with TemporaryDirectory() as temporary:
            geometry=Path(temporary)/"geometry.npz";geometry.write_bytes(b"CPU fixture")
            checkpoint=Path(temporary)/"checkpoint";checkpoint.write_bytes(b"CPU fixture")
            with patch.object(run,"ExactEvaluationCapture",FakeCapture):
                report=run.run_diagnostic_trial(native,lambda obs:torch.zeros(1,18),Path(temporary)/"run",geometry,
                                               checkpoint,command=(.05,0.,0.),record_video=False)
            self.assertFalse(report["stage2_complete"])
            self.assertEqual(len(report["required_cases_not_evaluated"]),96)
            self.assertEqual(report["cases"][0]["command"],[.05,0.,0.])

    def test_deadline_preserves_prefix_and_does_not_start_next_case(self):
        native=FakeNative()
        config=run.EvaluationConfig(static_indices=(0,1),include_transitions=False,include_quiet=False,
            include_stops=False,include_formal=False,record_video=False)
        with TemporaryDirectory() as temporary:
            geometry=Path(temporary)/"geometry.npz";geometry.write_bytes(b"CPU fixture")
            checkpoint=Path(temporary)/"checkpoint";checkpoint.write_bytes(b"CPU fixture")
            with patch.object(run,"ExactEvaluationCapture",FakeCapture),patch.object(run.time,"monotonic",side_effect=lambda:float(native.steps)):
                summary=run.evaluate_suite(native,lambda obs:torch.zeros(1,18),Path(temporary)/"run",geometry,
                    checkpoint,config=config,max_wall_seconds=50.)
            self.assertEqual(native.steps,100)
            self.assertEqual(native.reset_count,1)
            self.assertTrue(summary["allocation_limit_reached"])
            self.assertFalse(summary["stage2_complete"])
            self.assertEqual(len(summary["results"]),1)
            self.assertEqual(len(summary["missing_cases"]),95)
            self.assertFalse(summary["results"][0]["native_capture_complete"])
            with np.load(Path(temporary)/"run/batch_000/control_trace.npz") as saved:
                self.assertEqual(len(saved["time_s"]),100)
                self.assertEqual(saved["reset"].sum(),0)

    def test_400hz_contact_fraction_boundary_and_missing_evidence(self):
        moving={"profile":"omni_static","command":[.1,0.,0.]}
        receipt={"steps":8000,"nonfoot_contact_steps_400hz":[8]}
        self.assertTrue(run._native_contact_screen(receipt,0,moving)["pass"])
        receipt["nonfoot_contact_steps_400hz"]=[9]
        self.assertFalse(run._native_contact_screen(receipt,0,moving)["pass"])
        receipt["nonfoot_contact_steps_400hz"]=[1]
        quiet={"profile":"quiet_stand","command":[0.,0.,0.]}
        self.assertFalse(run._native_contact_screen(receipt,0,quiet)["pass"])
        self.assertFalse(run._native_contact_screen({"steps":8000},0,moving)["pass"])

    def test_learning_probe_manifest_keeps_signed_cases_and_original_windows(self):
        cases=run.learning_probe_cases()
        self.assertEqual(len(cases),13)
        self.assertEqual(sum(c["controls"] for c in cases),13650)
        self.assertFalse(set(c["case_id"] for c in cases)&set(c["case_id"] for c in run.case_manifest()))
        for i in range(8):
            np.testing.assert_allclose(cases[i]["command"],[-x for x in cases[(i+4)%8]["command"]],atol=1e-15)
            self.assertAlmostEqual(np.linalg.norm(cases[i]["command"]),.05)
        self.assertEqual([c["command"][2] for c in cases[8:10]],[-.2,.2])
        self.assertEqual([(c["profile"],c["controls"]) for c in cases[10:]],
            [("quiet_stand",1000),("stage2_long_quiet",1600),("stop_to_stand",1050)])
        self.assertEqual(run.command_at(cases[-1],399),[.05,0.,0.])
        self.assertEqual(run.command_at(cases[-1],400),[0.,0.,0.])

    def test_learning_probes_keep_failed_prefix_then_run_other_trials(self):
        native=FakeNative(terminate_at=2)
        with TemporaryDirectory() as temporary:
            geometry=Path(temporary)/"geometry.npz";geometry.write_bytes(b"CPU fixture")
            checkpoint=Path(temporary)/"checkpoint";checkpoint.write_bytes(b"CPU fixture")
            with patch.object(run,"ExactEvaluationCapture",FakeCapture):
                summary=run.run_learning_probe_suite(native,lambda obs:torch.zeros(1,18),Path(temporary)/"run",geometry,checkpoint)
            self.assertEqual(native.reset_count,13)
            self.assertEqual(native.steps,13650-998)
            self.assertEqual(len(summary["results"]),13)
            self.assertEqual(summary["missing_probe_cases"],[])
            self.assertEqual(len(summary["missing_cases"]),96)
            self.assertFalse(summary["stage2_complete"])
            self.assertFalse(summary["results"][0]["native_capture_complete"])
            self.assertTrue(summary["results"][1]["native_capture_complete"])
            with np.load(Path(temporary)/"run/batch_000/control_trace.npz") as raw:
                self.assertEqual(len(raw["time_s"]),2)
                self.assertTrue(raw["terminated"][-1,0])
                self.assertEqual(raw["reset"].sum(),0)
            with np.load(Path(temporary)/"run/batch_012/control_trace.npz") as raw:
                np.testing.assert_array_equal(raw["command"][:400,0],np.tile([.05,0.,0.],(400,1)))
                self.assertEqual(abs(raw["command"][400:]).max(),0.)
                self.assertEqual(raw["reset"].sum(),0)

    def test_learning_probe_deadline_retains_all_missing_qualification_cases(self):
        native=FakeNative()
        with TemporaryDirectory() as temporary:
            geometry=Path(temporary)/"geometry.npz";geometry.write_bytes(b"CPU fixture")
            checkpoint=Path(temporary)/"checkpoint";checkpoint.write_bytes(b"CPU fixture")
            with patch.object(run,"ExactEvaluationCapture",FakeCapture),patch.object(run.time,"monotonic",side_effect=lambda:float(native.steps)):
                summary=run.run_learning_probe_suite(native,lambda obs:torch.zeros(1,18),Path(temporary)/"run",geometry,checkpoint,max_wall_seconds=50.)
            self.assertEqual(native.reset_count,1)
            self.assertEqual(native.steps,100)
            self.assertTrue(summary["allocation_limit_reached"])
            self.assertEqual(len(summary["missing_probe_cases"]),12)
            self.assertEqual(len(summary["missing_cases"]),96)

    def test_learning_probe_video_is_optional_single_case_and_never_qualifies(self):
        native=FakeNative();native.cfg=replace(native.cfg,render=True)
        cases=run.learning_probe_cases();chosen=cases[3]["case_id"]
        calls=[]
        def batch(env,policy,selected,output,geometry,**kwargs):
            calls.append(kwargs.get("video_case_id"))
            return {"results":[{"case_id":selected[0]["case_id"],"pass":True}],
                    "video_frames":1 if kwargs.get("video_case_id") else 0,"native_capture_failure":None,
                    "acquisition_complete":True,"failure":None,"failure_kind":None}
        with TemporaryDirectory() as temporary:
            geometry=Path(temporary)/"geometry.npz";geometry.write_bytes(b"CPU fixture")
            checkpoint=Path(temporary)/"checkpoint";checkpoint.write_bytes(b"CPU fixture")
            with patch.object(run,"run_batch",side_effect=batch):
                summary=run.run_learning_probe_suite(native,None,Path(temporary)/"run",geometry,checkpoint,
                    record_video=True,video_case_id=chosen)
            self.assertEqual(calls,[chosen if i==3 else None for i in range(13)])
            self.assertTrue(summary["diagnostic_screens_passed"])
            self.assertFalse(summary["stage2_complete"])
            self.assertEqual(len(summary["missing_cases"]),96)
            self.assertEqual(summary["video_files"],["batch_003/rollout.mp4"])
            self.assertEqual(summary["allocation"]["selected_case_ids"],[c["case_id"] for c in cases])
            self.assertEqual(summary["allocation"]["unselected_probe_case_ids"],[])
            self.assertEqual(summary["missing_selected_probe_cases"],[])

    def test_learning_probe_subset_preserves_order_declarations_and_missing_cases(self):
        selected=["learning:quiet_20s","learning:forward_0.05_to_stop","learning:translate_0.05_0deg"]
        calls=[]
        def batch(env,policy,cases,output,geometry,**kwargs):
            calls.append((cases[0],Path(output).name))
            return {"results":[{"case_id":cases[0]["case_id"],"pass":True}],
                "native_capture_failure":None,"acquisition_complete":True,"failure":None,"failure_kind":None}
        with TemporaryDirectory() as temporary:
            checkpoint=Path(temporary)/"checkpoint";checkpoint.write_bytes(b"CPU fixture")
            with patch.object(run,"run_batch",side_effect=batch):
                summary=run.run_learning_probe_suite(FakeNative(),None,Path(temporary)/"run",None,checkpoint,
                    selected_case_ids=selected)
            self.assertEqual([c[0]["case_id"] for c in calls],selected)
            self.assertEqual([c[1] for c in calls],["batch_000","batch_001","batch_002"])
            canonical=run.learning_probe_cases();by_id={c["case_id"]:c for c in canonical}
            self.assertEqual([c[0] for c in calls],[by_id[case_id] for case_id in selected])
            self.assertEqual(summary["allocation"]["probe_cases"],canonical)
            self.assertEqual(summary["allocation"]["selected_case_ids"],selected)
            self.assertEqual(summary["allocation"]["requested_controls"],3050)
            missing=[c["case_id"] for c in canonical if c["case_id"] not in selected]
            self.assertEqual(summary["missing_probe_cases"],missing)
            self.assertEqual(summary["allocation"]["unselected_probe_case_ids"],missing)
            self.assertEqual(summary["missing_selected_probe_cases"],[])
            self.assertTrue(all(r["pass"] for r in summary["results"]))
            self.assertFalse(summary["diagnostic_screens_passed"])
            self.assertFalse(summary["stage2_complete"])
            self.assertEqual(len(summary["missing_cases"]),96)
            import json
            self.assertEqual(json.loads((Path(temporary)/"run/summary.json").read_text()),summary)

    def test_learning_probe_invalid_selection_cannot_create_output_or_reset(self):
        first=run.learning_probe_cases()[0]["case_id"]
        for selected in ([],(),[first,first],["unknown"],first,{first},[None]):
            with self.subTest(selected=selected),TemporaryDirectory() as temporary:
                native=FakeNative();output=Path(temporary)/"run"
                with patch.object(run,"run_batch") as batch,self.assertRaises(ValueError):
                    run.run_learning_probe_suite(native,None,output,None,None,selected_case_ids=selected)
                self.assertFalse(output.exists());self.assertEqual(native.reset_count,0);batch.assert_not_called()

    def test_learning_probe_subset_video_must_be_selected_and_defaults_to_first(self):
        selected=["learning:quiet_20s","learning:translate_0.05_180deg"]
        native=FakeNative();native.cfg=replace(native.cfg,render=True)
        with TemporaryDirectory() as temporary:
            checkpoint=Path(temporary)/"checkpoint";checkpoint.write_bytes(b"CPU fixture")
            with self.assertRaisesRegex(ValueError,"outside the selected"):
                run.run_learning_probe_suite(native,None,Path(temporary)/"invalid",None,checkpoint,
                    selected_case_ids=selected,record_video=True,video_case_id="learning:quiet_32s")
            self.assertFalse((Path(temporary)/"invalid").exists())
            calls=[]
            def batch(env,policy,cases,output,geometry,**kwargs):
                calls.append(kwargs["video_case_id"])
                return {"results":[{"case_id":cases[0]["case_id"],"pass":True}],"acquisition_complete":True,
                    "video_frames":1 if kwargs["video_case_id"] else 0,"native_capture_failure":None,"failure":None}
            with patch.object(run,"run_batch",side_effect=batch):
                summary=run.run_learning_probe_suite(native,None,Path(temporary)/"run",None,checkpoint,
                    selected_case_ids=selected,record_video=True)
            self.assertEqual(calls,[selected[0],None])
            self.assertEqual(summary["video_files"],["batch_000/rollout.mp4"])
            self.assertFalse(summary["diagnostic_screens_passed"])

    def test_learning_probe_stops_on_actor_integrity_error(self):
        native=FakeNative()
        policy=SimpleNamespace(act=lambda obs,deterministic:torch.zeros(1,18),
            model=SimpleNamespace(actor=lambda obs:(torch.ones(1,18),torch.zeros(1,3))))
        with TemporaryDirectory() as temporary:
            geometry=Path(temporary)/"geometry.npz";geometry.write_bytes(b"CPU fixture")
            checkpoint=Path(temporary)/"checkpoint";checkpoint.write_bytes(b"CPU fixture")
            with patch.object(run,"ExactEvaluationCapture",FakeCapture):
                summary=run.run_learning_probe_suite(native,policy,Path(temporary)/"run",geometry,checkpoint)
            self.assertEqual(native.reset_count,1)
            self.assertEqual(native.steps,1)
            self.assertIn("differs from its actual actor",summary["acquisition_failure"])
            self.assertFalse(summary["diagnostic_screens_passed"])
            self.assertEqual(len(summary["missing_probe_cases"]),12)
            self.assertEqual(len(summary["missing_cases"]),96)
            import json
            report=json.loads((Path(temporary)/"run/batch_000/report.json").read_text())
            self.assertEqual(report["failure_kind"],"acquisition_error")
            self.assertEqual(report["recorded_physics_steps"],8)

    def test_learning_probe_does_not_treat_arbitrary_failure_as_native_terminal(self):
        for failure,kind in (("Native terminal/timeout: untrusted text",None),
                             ("Video finalization failed after native termination","acquisition_error"),
                             (None,None)):
            with self.subTest(failure=failure),TemporaryDirectory() as temporary:
                geometry=Path(temporary)/"geometry.npz";geometry.write_bytes(b"CPU fixture")
                checkpoint=Path(temporary)/"checkpoint";checkpoint.write_bytes(b"CPU fixture")
                def batch(env,policy,selected,output,geometry,**kwargs):
                    return {"results":[{"case_id":selected[0]["case_id"],"pass":False}],
                        "native_capture_failure":None,"acquisition_complete":False,
                        "failure":failure,"failure_kind":kind}
                with patch.object(run,"run_batch",side_effect=batch) as calls:
                    summary=run.run_learning_probe_suite(FakeNative(),None,Path(temporary)/"run",geometry,checkpoint)
                self.assertEqual(calls.call_count,1)
                self.assertTrue(summary["acquisition_failure"])
                self.assertFalse(summary["diagnostic_screens_passed"])
                self.assertEqual(len(summary["missing_probe_cases"]),12)

    def test_learning_probe_final_pass_requires_intact_inputs_and_wall_allocation(self):
        for final_change in ("hash","missing","deadline"):
            with self.subTest(final_change=final_change),TemporaryDirectory() as temporary:
                geometry=Path(temporary)/"geometry.npz";geometry.write_bytes(b"CPU fixture")
                checkpoint=Path(temporary)/"checkpoint";checkpoint.write_bytes(b"CPU fixture")
                calls=[]
                def batch(env,policy,selected,output,geometry,**kwargs):
                    calls.append(selected[0]["case_id"])
                    final=len(calls)==13
                    if final and final_change=="hash":checkpoint.write_bytes(b"Changed checkpoint")
                    if final and final_change=="missing":checkpoint.unlink()
                    return {"results":[{"case_id":selected[0]["case_id"],"pass":True}],
                        "native_capture_failure":None,"acquisition_complete":True,"failure":None,
                        "failure_kind":None,"allocation_limit_reached":final and final_change=="deadline"}
                with patch.object(run,"run_batch",side_effect=batch):
                    summary=run.run_learning_probe_suite(FakeNative(),None,Path(temporary)/"run",geometry,checkpoint)
                self.assertEqual(len(calls),13)
                self.assertTrue(all(r["pass"] for r in summary["results"]))
                self.assertFalse(summary["diagnostic_screens_passed"])
                self.assertEqual(summary["missing_probe_cases"],[])
                self.assertEqual(len(summary["missing_cases"]),96)
                self.assertFalse(summary["stage2_complete"])
                if final_change=="deadline":self.assertTrue(summary["allocation_limit_reached"])
                else:self.assertTrue(summary["acquisition_failure"])


if __name__=="__main__":unittest.main()
