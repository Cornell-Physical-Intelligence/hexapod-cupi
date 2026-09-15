"""Native policy evaluation orchestration; physics and acceptance gates unchanged.

The full manifest is frozen independently of any pilot subset. Every trial has
one initial reset, then uninterrupted policy-driven physics. Failed prefixes,
400Hz state/motor/contact records and any actual video remain in their trial.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np
import torch
from .env_config import JOINT_NAMES, KD, MODEL_SHA256, URDF_SHA256, USD_SHA256
from .evaluation_config import EvaluationEnvConfig
from .env import _DiagnosticGeometry, _classify_patches, _diagnostic_rotation, _diagnostic_servo
from . import evaluation as scoring


@dataclass(frozen=True)
class EvaluationConfig:
    static_indices: tuple | None = None
    include_transitions: bool = True
    include_quiet: bool = True
    include_stops: bool = True
    include_formal: bool = True
    record_video: bool = True
    video_case_id: str = "static:translate_0.10_22.5deg"
    seed: int = 27057
    allow_unisolated_batch_pilot: bool = False


class _NativeTrialTermination(RuntimeError):
    """The recorded native endpoint terminated this trial without resetting it."""


def stop_scenarios():
    return [{"name": name, "command": list(command)} for name, command in (
        ("stand", (0., 0., 0.)), ("forward", (.10, 0., 0.)),
        ("reverse", (-.10, 0., 0.)), ("left", (0., .10, 0.)),
        ("right", (0., -.10, 0.)), ("forward_fast", (.20, 0., 0.)),
        ("turn_left", (0., 0., .20)), ("turn_right", (0., 0., -.20)),
        ("arc_left", (.10, 0., .20)), ("arc_right", (.10, 0., -.20)),
        ("strafe_arc", (0., .10, .20)), ("diagonal", (.071, .071, 0.)))]


def case_manifest():
    cases = [{"case_id": "static:"+r["name"], "profile": "omni_static", "command": r["command"], "controls": 1000}
             for r in scoring.evaluation_scenarios()]
    cases += [{"case_id": "transition:full_program", "profile": "transition", "command": [0., 0., 0.],
               "controls": round(sum(x[1] for x in scoring.transition_sequence())/.02)}]
    cases += [{"case_id": "quiet:stand", "profile": "quiet_stand", "command": [0., 0., 0.], "controls": 1000}]
    cases += [{"case_id": "quiet:stage2_long", "profile": "stage2_long_quiet", "command": [0., 0., 0.], "controls": 1600}]
    cases += [{"case_id": "stop:"+r["name"], "profile": "stop_to_stand", "command": r["command"], "controls": 1050}
              for r in stop_scenarios()]
    cases += [{"case_id": f"formal:{speed:.2f}", "profile": "stage2c_formal", "command": [speed, 0., 0.], "controls": 500, "seed": 60}
              for speed in (0., .16, .20, .30)]
    return cases


def selected_cases(config):
    full = case_manifest()
    indices = tuple(range(77)) if config.static_indices is None else tuple(config.static_indices)
    if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < 77 for i in indices):
        raise ValueError("Static subset must contain unique valid indices from the frozen77-case matrix")
    selected = [full[i] for i in indices]
    enabled = {"transition": config.include_transitions, "quiet_stand": config.include_quiet,
               "stage2_long_quiet": config.include_quiet,
               "stop_to_stand": config.include_stops, "stage2c_formal": config.include_formal}
    selected += [c for c in full[77:] if enabled[c["profile"]]]
    if not selected:
        raise ValueError("Empty evaluation allocation")
    if config.record_video and config.video_case_id not in {x["case_id"] for x in selected}:
        raise ValueError("Requested actual video case is not in this evaluation allocation")
    return selected


def command_at(case, control):
    if not 0 <= control < case["controls"]:
        raise ValueError("Control lies outside declared case")
    if case["profile"] == "transition":
        offset = 0
        for name, duration, command in scoring.transition_sequence():
            count = round(duration/.02)
            if control < offset+count:
                return scoring.trajectory_command(name, (control-offset)*.02, duration, command)
            offset += count
        raise AssertionError("Incomplete transition schedule")
    if case["profile"] == "stop_to_stand" and control >= 400:
        return [0., 0., 0.]
    return list(case["command"])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def _cpu(value):
    return value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.array(value, copy=True)


def _deadline(seconds):
    if seconds is None:
        return math.inf
    if not math.isfinite(seconds) or seconds<=0:
        raise ValueError("Evaluation wall-clock allocation must be finite and positive")
    return time.monotonic()+seconds


class ExactEvaluationCapture:
    """Capture all eight native samples, including full unedited patch packets."""
    def __init__(self, env, output, geometry_extrema):
        if env.capture is not None:
            raise ValueError("Another owner already occupies the native capture hook")
        self.env, self.output = env, Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        with np.load(geometry_extrema, allow_pickle=False) as data:
            self.geometry = _DiagnosticGeometry(env.geometry_meta, {k: data[k] for k in data.files}, env.native_body_names)
        self.counter = env.sim.get_physics_step_count()
        self.initial_counter = self.counter
        self.count = 0
        self.rows, self.last_control, self.files = [], [], []
        self.failure = None
        self.contacts = (self.output/"contacts.jsonl").open("w")
        self.maximum_applied = np.zeros(env.num_envs)
        self.maximum_requested = np.zeros(env.num_envs)
        self.joint_bound_violations = np.zeros(env.num_envs, dtype=int)
        self.speed_bound_violations = np.zeros(env.num_envs, dtype=int)
        self.nonfoot_steps = np.zeros(env.num_envs, dtype=int)
        self.minimum_clearance = np.full(env.num_envs, np.inf)
        self.minimum_plate = np.full(env.num_envs, np.inf)
        self.previous_q = _cpu(env.current["q"]).astype(float)
        self.native_limits = np.asarray(env.native_readback["limits"])
        self.native_speed = np.asarray(env.native_readback["native_max_velocity"])
        _save(self.output/"initial_state.json", {k: _cpu(v).tolist() for k, v in env.current.items()})
        env.capture = self

    def __call__(self, env, state, q, dq, requested, applied, ceiling, target, forces, substep, native_pre):
        if substep == 0:
            self.last_control = []
        row = {"sequence": np.array(self.count), "control_index": np.array(self.count//8),
            "substep_index": np.array(substep), "time_s": np.array((self.count+1)*.0025),
            "explicit_counter": np.array(env.sim.get_physics_step_count()),
            "pre_joint_position_rad": _cpu(q), "pre_joint_velocity_rad_s": _cpu(dq),
            "joint_position_rad": _cpu(state["q"]), "joint_velocity_rad_s": _cpu(state["dq"]),
            "joint_target_rad": _cpu(target), "computed_torque_nm": _cpu(requested),
            "applied_torque_nm": _cpu(applied), "effort_ceiling_nm": _cpu(ceiling),
            "native_input_pre_nm": _cpu(native_pre), "root_pose_xyzw": _cpu(state["root"]),
            "root_com_velocity": _cpu(state["root_velocity"]), "link_pose_xyzw": _cpu(state["link"]),
            "command": _cpu(env.commands), "toe_xyz_world_m": _cpu(state["toe_world"]),
            "toe_xyz_body_m": _cpu(state["toe_body"])}
        try:
            buffers = [np.array(x.numpy(), copy=True) for x in env.contact.get_contact_data(.0025)]
            contacts = _classify_patches(buffers, env.sensor_map, row["link_pose_xyzw"], self.geometry, env.num_envs)
            minimum, non_toe = self.geometry.clearance(row["link_pose_xyzw"])
            row.update(distal_contact=contacts["distal_contact"], distal_force_world_n=contacts["distal_force_world"],
                nonfoot_contact=contacts["nonfoot_contact"], nonfoot_force_world_n=contacts["nonfoot_force_world"],
                minimum_mesh_floor_m=minimum, minimum_non_toe_floor_m=non_toe,
                interval_angle_rate_rad_s=(row["joint_position_rad"].astype(float)-self.previous_q)/.0025)
            self.rows.append(row)
            self.last_control.append(row)
            self.contacts.write(json.dumps({"sequence": self.count, "explicit_counter": int(row["explicit_counter"]),
                "patches": contacts["patches"]}, allow_nan=False)+"\n")
            self.count += 1
            if int(row["explicit_counter"]) != self.counter+1 or substep != (self.count-1)%8:
                raise RuntimeError("Native evaluation sample/counter continuity differs")
            self.counter += 1
            if not all(np.isfinite(v).all() for v in row.values()):
                raise FloatingPointError("Nonfinite native evaluation evidence")
            if not np.array_equal(row["native_input_pre_nm"], row["applied_torque_nm"]):
                raise RuntimeError("Actual native motor input differs")
            wanted = _diagnostic_servo(row["pre_joint_position_rad"], row["pre_joint_velocity_rad_s"],
                                      row["joint_target_rad"], np.full(18,12.,np.float32), np.asarray(KD,np.float32))
            for name, expected in zip(("computed_torque_nm", "applied_torque_nm", "effort_ceiling_nm"), wanted):
                if not np.array_equal(row[name], expected):
                    raise RuntimeError("Canonical motor recurrence differs: "+name)
            self.maximum_applied = np.maximum(self.maximum_applied, abs(row["applied_torque_nm"]).max(-1))
            self.maximum_requested = np.maximum(self.maximum_requested, abs(row["computed_torque_nm"]).max(-1))
            self.joint_bound_violations += ((row["joint_position_rad"] < self.native_limits[:,:,0]-2e-6) |
                                             (row["joint_position_rad"] > self.native_limits[:,:,1]+2e-6)).any(-1)
            self.speed_bound_violations += (abs(row["joint_velocity_rad_s"]) > self.native_speed+2e-6).any(-1)
            self.nonfoot_steps += row["nonfoot_contact"]
            self.minimum_clearance = np.minimum(self.minimum_clearance, non_toe)
            self.minimum_plate = np.minimum(self.minimum_plate, row["root_pose_xyzw"][:,2])
            self.previous_q = row["joint_position_rad"].astype(float)
            if self.maximum_applied.max() > 1.60001:
                raise RuntimeError("Native applied torque cap exceeded")
            if len(self.rows) >= 800:
                self.flush()
        except BaseException as error:
            self.failure = repr(error)
            _save(self.output/"failed_partial_step.json", {k: np.asarray(v).tolist() for k,v in row.items()})
            self.flush()
            raise

    def control_record(self, result, requested_command, control):
        if len(self.last_control) != 8:
            raise RuntimeError("A complete eight-sample hold is required")
        raw = {key: np.stack([r[key] for r in self.last_control]) for key in self.last_control[0]}
        last = self.last_control[-1]
        world_velocity = last["root_com_velocity"][:,:3]
        rotations = np.stack([_diagnostic_rotation(q) for q in last["root_pose_xyzw"][:,3:]])
        body_velocity = np.einsum("nji,nj->ni", rotations, world_velocity)
        gyro = np.einsum("nji,nj->ni", rotations, last["root_com_velocity"][:,3:])
        nav = np.stack((-body_velocity[:,1], body_velocity[:,0], body_velocity[:,2]), -1)
        return {"time_s": np.array((control+1)*.02), "root_pose_xyzw": last["root_pose_xyzw"],
            "velocity_world_mps": world_velocity, "velocity_navigation_mps": nav,
            "gyro_body_rad_s": gyro, "joint_position_rad": last["joint_position_rad"],
            "joint_velocity_rad_s": last["joint_velocity_rad_s"], "joint_target_rad": last["joint_target_rad"],
            "computed_torque_nm": last["computed_torque_nm"], "applied_torque_nm": last["applied_torque_nm"],
            "saturation_count_400hz": (abs(raw["computed_torque_nm"])>1.6).sum(0),
            "applied_torque_squared_sum_400hz": np.square(raw["applied_torque_nm"]).sum(0),
            "computed_torque_abs_max_400hz": abs(raw["computed_torque_nm"]).max((0,2)),
            "applied_torque_abs_max_400hz": abs(raw["applied_torque_nm"]).max((0,2)),
            "nonfoot_contact": last["nonfoot_contact"],
            "nonfoot_contact_count_400hz": raw["nonfoot_contact"].sum(0),
            "missing_six_toe_count_400hz": (~raw["distal_contact"].all(-1)).sum(0),
            "minimum_non_toe_floor_m": raw["minimum_non_toe_floor_m"].min(0),
            "command": last["command"], "requested_command": np.asarray(requested_command,np.float32),
            "terminated": _cpu(result["terminated"]), "truncated": _cpu(result["truncated"]),
            "reset": np.zeros(self.env.num_envs,bool)}

    def flush(self):
        if self.rows:
            name = f"substeps_{len(self.files):03d}.npz"
            np.savez_compressed(self.output/name, **{k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]})
            self.files.append(name)
            self.rows = []
        self.contacts.flush()

    def close(self):
        self.env.capture = None
        self.flush()
        self.contacts.close()
        receipt = {"schema":"canonical_exact_evaluation_capture_v1", "steps":self.count,
            "initial_counter":self.initial_counter, "final_counter":self.counter,
            "failure":self.failure, "substep_files":self.files,
            "maximum_applied_nm":self.maximum_applied.tolist(), "maximum_requested_nm":self.maximum_requested.tolist(),
            "joint_bound_violation_steps":self.joint_bound_violations.tolist(),
            "speed_bound_violation_steps":self.speed_bound_violations.tolist(),
            "nonfoot_contact_steps_400hz":self.nonfoot_steps.tolist(),
            "minimum_non_toe_floor_m": [None if not np.isfinite(x) else float(x) for x in self.minimum_clearance],
            "minimum_plate_height_m": [None if not np.isfinite(x) else float(x) for x in self.minimum_plate],
            "body_names":self.env.native_body_names, "joint_names":list(JOINT_NAMES),
            "files":{p.name:_sha(p) for p in sorted(self.output.iterdir()) if p.is_file()},
            "contact_classification":"exact_distal_points", "stage2_complete":False}
        _save(self.output/"capture.json", receipt)
        return receipt


def _set_command(env, commands):
    env.commands.copy_(torch.as_tensor(commands, dtype=torch.float32, device=env.device))
    # History contains proprio only; update command-bearing observations before
    # the actor, without inserting a history sample or native physics step.
    return env._observations(env.current)


def _native_contact_screen(receipt, index, case):
    """Common canonical400Hz screen, separate from historical sampled metrics."""
    quiet=case["profile"] in {"quiet_stand","stage2_long_quiet","stop_to_stand"}
    quiet=quiet or (case["profile"]!="transition" and not np.any(case["command"]))
    bound=0. if quiet else .001
    count=None;fraction=None
    values=receipt.get("nonfoot_contact_steps_400hz")
    steps=receipt.get("steps",0)
    if isinstance(values,list) and index<len(values) and type(steps) is int and steps>0:
        value=values[index]
        if type(value) is int and 0<=value<=steps:
            count=value;fraction=count/steps
    return {"schema":"canonical_native_contact_screen_v1", "frequency_hz":400,
        "window":"complete trial, including initial settling and every native substep",
        "recorded_substeps":steps,"nonfoot_substeps":count,"nonfoot_fraction":fraction,
        "maximum_nonfoot_fraction":bound,"pass":fraction is not None and fraction<=bound,
        "scope":"Common canonical native contact evidence; historical sampled metrics remain separate."}


@torch.inference_mode()
def run_batch(env, policy, cases, output, geometry_extrema, *, checkpoint_sha256, source_sha256,
              seed=27057, video_case_id=None, progress=None, max_wall_seconds=None):
    """One initial reset, full fixed-duration trials; a failure preserves prefixes."""
    if not cases or len(cases)>env.num_envs or len({x["controls"] for x in cases})!=1:
        raise ValueError("Batch cases must fit the environment and have one fixed duration")
    duration = cases[0]["controls"]*.02
    if env.cfg.episode_seconds <= duration:
        raise ValueError("Declared episode timeout would truncate the requested evaluation")
    deadline=_deadline(max_wall_seconds)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    assignments = cases + [cases[-1]]*(env.num_envs-len(cases))
    torch.manual_seed(seed);np.random.seed(seed)
    env.commands.zero_();env.reset()
    capture=ExactEvaluationCapture(env,out/"native400hz",geometry_extrema)
    rows=[];failure=None;failure_kind=None;writer=None;frames=0;allocation_limit_reached=False
    video_index=next((i for i,c in enumerate(cases) if c["case_id"]==video_case_id),None)
    declaration={"schema":"canonical_stage2_native_batch_v1", "cases":cases,
        "assigned_case_ids":[c["case_id"] for c in assignments], "padding_replicas":env.num_envs-len(cases),
        "seed":seed, "episode_timeout_seconds":env.cfg.episode_seconds,"max_wall_seconds":max_wall_seconds,
        "checkpoint_sha256":checkpoint_sha256,"source_sha256":source_sha256,
        "model_sha256":MODEL_SHA256,"urdf_sha256":URDF_SHA256,"usd_sha256":USD_SHA256,
        "geometry_extrema_sha256":_sha(geometry_extrema),"contact_classification":"exact_distal_points",
        "velocity_measurement_point":"native root center_of_mass; actual world velocity rotated into body/navigation",
        "initial_resets":1,"resets_during_trial":0,"pose_forcing":False,"stage2_complete":False}
    _save(out/"declaration.json",declaration)
    try:
        if video_index is not None:
            if not env.cfg.render:
                raise ValueError("Native cameras must be enabled before constructing a video evaluation environment")
            import imageio.v2 as imageio
            env.render(index=video_index)
            writer=imageio.get_writer(str(out/"rollout.mp4"),fps=25,codec="libx264",quality=8)
        for control in range(cases[0]["controls"]):
            if control%100==0 and time.monotonic()>=deadline:
                allocation_limit_reached=True
                raise RuntimeError("Evaluation wall-clock allocation exhausted; preserved prefix")
            requested=[command_at(c,control) for c in assignments]
            observation=_set_command(env,requested)
            action=policy.act(observation["obs"],deterministic=True) if hasattr(policy,"act") else policy(observation["obs"])
            result=env.step(action)
            row=capture.control_record(result,requested,control)
            row["policy_action"]=_cpu(action)
            row["policy_observation"]=_cpu(observation["obs"])
            row["critic_observation"]=_cpu(observation["critic"])
            row["amp_state_before"]=_cpu(observation["amp"])
            row["amp_state_after"]=_cpu(result["amp"])
            row["toe_xyz_world_m"]=_cpu(env.current["toe_world"])
            row["distal_contact"]=capture.last_control[-1]["distal_contact"].copy()
            if hasattr(policy, "model"):
                mean, estimate = policy.model.actor(observation["obs"])
                if not torch.equal(mean, action):
                    raise RuntimeError("Recorded deterministic policy differs from its actual actor")
                row["estimated_velocity_navigation_mps"]=_cpu(estimate)
            if hasattr(policy, "amp"):
                row["motion_prior_style_reward"]=_cpu(policy.amp.style_reward(observation["amp"], result["amp"]))
                row["motion_prior_discriminator_score"]=_cpu(policy.amp.discriminator(
                    policy.amp.features(observation["amp"], result["amp"])).squeeze(-1))
            rows.append(row)
            if writer is not None and control%2==1:
                frame=env.render(index=video_index);writer.append_data(frame);frames+=1
                if frames==1:imageio.imwrite(str(out/"first_frame.png"),frame)
            if progress is not None and (control+1)%100==0:
                progress({"case_ids":[c["case_id"] for c in cases],"controls":control+1,"requested_controls":cases[0]["controls"]})
            if bool(result["terminated"].any() or result["truncated"].any()):
                raise _NativeTrialTermination("Native terminal/timeout: preserved failed prefix; no automatic reset")
        env.verify_native_recipe("after_stage2_evaluation")
    except BaseException as error:
        failure=repr(error)
        failure_kind=("native_terminal_prefix" if isinstance(error,_NativeTrialTermination) else
                      "allocation_deadline" if allocation_limit_reached else "acquisition_error")
    finally:
        try:
            if writer is not None:writer.close()
        except Exception as error:
            failure=(failure+"; " if failure else "")+"Video finalization: "+repr(error)
            failure_kind="acquisition_error"
        receipt=capture.close()
        data={k:np.stack([r[k] for r in rows]) for k in rows[0]} if rows else {}
        if data:np.savez_compressed(out/"control_trace.npz",**data)
    results=[]
    for index,case in enumerate(cases):
        metadata={**declaration,"profile":case["profile"],"env_index":index,"command":case["command"],
                  "target_slew_rad":.04,"control_dt_s":.02,"seed":case.get("seed",seed)}
        try:
            if len(rows)>=2:
                result=scoring.score_transitions(data,metadata) if case["profile"]=="transition" else scoring.score_recording(data,metadata)
            else:
                result={"pass":False,"checks":{},"failed_bounds":["incomplete_native_prefix"],"missing_evidence":[]}
        except Exception as error:
            result={"pass":False,"checks":{},"failed_bounds":["invalid_recorded_evidence"],
                    "missing_evidence":[],"scoring_error":repr(error)}
        complete=failure is None and len(rows)==case["controls"] and receipt["steps"]==8*len(rows) and receipt["failure"] is None
        contact_screen=_native_contact_screen(receipt,index,case)
        native_ok=(complete and receipt["joint_bound_violation_steps"][index]==0
                   and receipt["speed_bound_violation_steps"][index]==0
                   and receipt["maximum_applied_nm"][index]<=1.60001
                   and receipt["minimum_non_toe_floor_m"][index]>=-.001
                   and receipt["minimum_plate_height_m"][index]>=.055
                   and contact_screen["pass"])
        result.update(case_id=case["case_id"],native_capture_complete=complete,native_motor_and_joint_checks_pass=native_ok,
                      native_contact_screen=contact_screen,batch_failure=failure,
                      passed_numeric_screen=result["pass"],stage2_complete=False)
        result["pass"]=bool(result["pass"] and native_ok)
        if not native_ok:
            result.setdefault("failed_bounds",[]).append("native_capture_or_original_physical_bounds")
        if not contact_screen["pass"]:
            result.setdefault("failed_bounds",[]).append("canonical_full_trial_400hz_nonfoot_fraction")
        results.append(result)
    report={**declaration,"acquisition_complete":failure is None,"failure":failure,"failure_kind":failure_kind,"controls":len(rows),
            "allocation_limit_reached":allocation_limit_reached,
            "native_capture_failure":receipt["failure"],
            "recorded_physics_steps":receipt["steps"],"results":results,"video_frames":frames,
            "video_case_id":video_case_id if video_index is not None else None,
            "files":{p.name:_sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    _save(out/"report.json",report)
    return report


def evaluate_suite(env, policy, output, geometry_extrema, checkpoint, *, config=None, evidence=None,
                   progress=None, max_wall_seconds=None):
    """Run selected cases while always reporting against the complete96-case manifest.

    Default is one robot to avoid inter-replica collision contamination. A
    multi-robot pilot is allowed only when explicitly labeled unisolated; it can
    never qualify Stage2. A future isolated batch needs separately bound physics.
    """
    deadline=_deadline(max_wall_seconds)
    config=EvaluationConfig() if config is None else config
    chosen=selected_cases(config);full=case_manifest()
    if env.cfg.episode_seconds!=90.:
        raise ValueError("Construct EvaluationEnvConfig with its declared90-second timeout")
    if env.num_envs!=1 and not config.allow_unisolated_batch_pilot:
        raise ValueError("Current scene has no verified inter-replica collision isolation; use one robot for formal evaluation")
    if config.record_video and not env.cfg.render:
        raise ValueError("Actual held-out video requires native cameras before startup")
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    checkpoint_sha=_sha(checkpoint);source_sha=_sha(__file__)
    declaration={"schema":"canonical_stage2_evaluation_allocation_v1","config":asdict(config),
        "full_required_cases":full,"selected_case_ids":[x["case_id"] for x in chosen],
        "checkpoint_sha256":checkpoint_sha,"orchestrator_sha256":source_sha,
        "evaluator_sha256":_sha(Path(scoring.__file__)),"num_envs":env.num_envs,
        "cross_replica_interference_excluded":env.num_envs==1,"episode_timeout_seconds":90.,
        "max_wall_seconds":max_wall_seconds,
        "held_out_video_scope":"22.5-degree command between training bearings; actual deterministic policy, no prescribed pose.",
        "stage2_complete":False}
    _save(out/"allocation.json",declaration)
    reports=[];results=[];started=time.monotonic();allocation_limit_reached=False
    # Group equal-duration/profile phases only; each transition remains one
    # contiguous70-second episode, never14 independent starts.
    for profile in ("omni_static","transition","quiet_stand","stage2_long_quiet","stop_to_stand","stage2c_formal"):
        group=[x for x in chosen if x["profile"]==profile]
        for offset in range(0,len(group),env.num_envs):
            remaining=deadline-time.monotonic()
            if remaining<=0:
                allocation_limit_reached=True
                break
            batch=group[offset:offset+env.num_envs]
            report=run_batch(env,policy,batch,out/f"batch_{len(reports):03d}",geometry_extrema,
                checkpoint_sha256=checkpoint_sha,source_sha256=source_sha,seed=60 if profile=="stage2c_formal" else config.seed,
                video_case_id=config.video_case_id if config.record_video else None,progress=progress,
                max_wall_seconds=None if not math.isfinite(remaining) else remaining)
            reports.append(report);results.extend(report["results"])
            _save(out/"progress.json",{"completed_batches":len(reports),"completed_case_ids":[x["case_id"] for x in results],
                                       "failed_case_ids":[x["case_id"] for x in results if not x["pass"]],"stage2_complete":False})
            allocation_limit_reached=report.get("allocation_limit_reached",False) or time.monotonic()>=deadline
            if env.native_errors or report["native_capture_failure"] or allocation_limit_reached:
                break
        if env.native_errors or (reports and reports[-1]["native_capture_failure"]) or allocation_limit_reached:
            break
    if _sha(checkpoint)!=checkpoint_sha or _sha(__file__)!=source_sha:
        raise RuntimeError("Evaluation input changed during collection")
    ids={x["case_id"] for x in results}
    facts=dict(evidence or {})
    facts.update(exact_model_bound=True,
        full_direction_matrix=all(x["case_id"] in ids for x in full if x["profile"]=="omni_static"),
        full_transition_matrix="transition:full_program" in ids and next(x for x in results if x["case_id"]=="transition:full_program").get("native_capture_complete",False),
        all_stop_cases=all(x["case_id"] in ids for x in full if x["profile"]=="stop_to_stand"),
        exact_contact_and_motor_evidence=env.num_envs==1 and bool(results) and all(x.get("native_capture_complete") and x.get("native_motor_and_joint_checks_pass") for x in results))
    summary=scoring.summarize_suite(results,required_cases=[x["case_id"] for x in full],evidence=facts)
    summary.update(allocation=declaration,results=results,wall_seconds=time.monotonic()-started,
                   allocation_limit_reached=allocation_limit_reached,
                   video_files=[str(Path(f"batch_{i:03d}")/"rollout.mp4") for i,r in enumerate(reports) if r["video_frames"]],
                   numerical_pass_is_not_visual_acceptance=True)
    _save(out/"summary.json",summary)
    return summary


def run_diagnostic_trial(env, policy, output, geometry_extrema, checkpoint, *,
                         command=(.05, 0., 0.), record_video=True, seed=27057, progress=None,
                         max_wall_seconds=None):
    """An explicitly additional20-second command screen; never substitutes a case.

    This supports early learning inspection at a training command such as.05m/s.
    The existing static scorer still owns its numerical bounds. Every one of the
    frozen96 required cases remains missing from this diagnostic-only result.
    """
    command=np.asarray(command,float)
    if command.shape!=(3,) or not np.isfinite(command).all():
        raise ValueError("Diagnostic command must be a finite navigation vector3")
    if env.num_envs!=1:
        raise ValueError("Diagnostic policy inspection uses one robot")
    if record_video and not env.cfg.render:
        raise ValueError("Actual video requires native cameras before startup")
    case={"case_id":"diagnostic:custom_command", "profile":"omni_static",
          "command":command.tolist(), "controls":1000}
    report=run_batch(env,policy,[case],output,geometry_extrema,
        checkpoint_sha256=_sha(checkpoint),source_sha256=_sha(__file__),seed=seed,
        video_case_id=case["case_id"] if record_video else None,progress=progress,
        max_wall_seconds=max_wall_seconds)
    report.update(schema="canonical_policy_diagnostic_trial_v1", stage2_complete=False,
        scope="Additional command inspection only; no required Stage2 case is replaced or admitted.",
        required_case_count=len(case_manifest()),
        required_cases_not_evaluated=[x["case_id"] for x in case_manifest()])
    _save(Path(output)/"diagnostic_result.json",report)
    return report


def learning_probe_cases():
    """Additional low-speed learning screens; distinct from every required case."""
    cases=[]
    for bearing in range(8):
        angle=bearing*math.pi/4
        cases.append({"case_id":f"learning:translate_0.05_{45*bearing}deg",
            "profile":"omni_static","command":[.05*math.cos(angle),.05*math.sin(angle),0.],"controls":1000})
    cases += [{"case_id":f"learning:yaw_{sign:+d}","profile":"omni_static",
               "command":[0.,0.,sign*.2],"controls":1000} for sign in (-1,1)]
    cases += [{"case_id":"learning:quiet_20s","profile":"quiet_stand","command":[0.,0.,0.],"controls":1000},
              {"case_id":"learning:quiet_32s","profile":"stage2_long_quiet","command":[0.,0.,0.],"controls":1600},
              {"case_id":"learning:forward_0.05_to_stop","profile":"stop_to_stand","command":[.05,0.,0.],"controls":1050}]
    return cases


def run_learning_probe_suite(env,policy,output,geometry_extrema,checkpoint,*,
                             record_video=False,video_case_id=None,seed=27057,
                             progress=None,max_wall_seconds=None):
    """Run 13 diagnostic trials in one existing native app without admitting Stage2.

    Each trial uses run_batch's one initial reset and unchanged numeric scorer.
    Falls preserve their trial prefixes; subsequent independent trials may run.
    Native acquisition corruption or the wall deadline stops further trials.
    """
    started=time.monotonic();deadline=_deadline(max_wall_seconds)
    cases=learning_probe_cases();full=case_manifest()
    if env.num_envs!=1:
        raise ValueError("Learning probes require one native robot")
    if env.cfg.episode_seconds<=32.:
        raise ValueError("Learning probes require an episode timeout greater than32 seconds")
    if record_video and not env.cfg.render:
        raise ValueError("Native cameras must be enabled before a video probe")
    if video_case_id is not None and not record_video:
        raise ValueError("A video case requires record_video=True")
    video_case_id=(video_case_id or cases[0]["case_id"]) if record_video else None
    if video_case_id is not None and video_case_id not in {c["case_id"] for c in cases}:
        raise ValueError("Video case is outside the declared learning probes")
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    checkpoint_sha=_sha(checkpoint);source_sha=_sha(__file__);scorer_sha=_sha(scoring.__file__)
    declaration={"schema":"canonical_learning_probe_allocation_v1","probe_cases":cases,
        "full_required_case_ids":[c["case_id"] for c in full],"seed":seed,"num_envs":1,
        "episode_timeout_seconds":env.cfg.episode_seconds,"max_wall_seconds":max_wall_seconds,
        "checkpoint_sha256":checkpoint_sha,"orchestrator_sha256":source_sha,"evaluator_sha256":scorer_sha,
        "record_video":record_video,"video_case_id":video_case_id,"requested_controls":sum(c["controls"] for c in cases),
        "trial_initialization":"One canonical state reset per trial; no reset within a trial and no solver-cache reset implied.",
        "acceptance_scope":"Additional diagnostic screens only; none substitute for the full 96 required cases.",
        "stage2_complete":False}
    _save(out/"allocation.json",declaration)
    reports=[];results=[];allocation_limit_reached=False;acquisition_failure=None

    def summary():
        ids={r["case_id"] for r in results}
        value=scoring.summarize_suite([],required_cases=declaration["full_required_case_ids"])
        value.update(schema="canonical_learning_probe_suite_v1",allocation=declaration,results=results,
            missing_probe_cases=[c["case_id"] for c in cases if c["case_id"] not in ids],
            failed_probe_cases=[r["case_id"] for r in results if not r["pass"]],
            diagnostic_screens_passed=not acquisition_failure and not allocation_limit_reached and
                len(results)==len(cases) and all(r["pass"] for r in results),
            allocation_limit_reached=allocation_limit_reached,acquisition_failure=acquisition_failure,
            wall_seconds=time.monotonic()-started,
            video_files=[str(Path(f"batch_{i:03d}")/"rollout.mp4") for i,r in enumerate(reports) if r.get("video_frames",0)],
            stage2_complete=False,scope="Low-speed learning comparison only; all 96 required Stage2 cases remain unevaluated by this allocation.")
        return value

    _save(out/"summary.json",summary())
    for index,case in enumerate(cases):
        remaining=deadline-time.monotonic()
        if remaining<=0:
            allocation_limit_reached=True
            break
        if env.native_errors:
            acquisition_failure="Native physics errors prevent another probe"
            break
        def trial_progress(update):
            if progress is not None:
                progress({"probe_index":index,"probe_count":len(cases),**update})
        try:
            report=run_batch(env,policy,[case],out/f"batch_{index:03d}",geometry_extrema,
                checkpoint_sha256=checkpoint_sha,source_sha256=source_sha,seed=seed,
                video_case_id=video_case_id if case["case_id"]==video_case_id else None,progress=trial_progress,
                max_wall_seconds=None if not math.isfinite(remaining) else remaining)
        except Exception as error:
            acquisition_failure=repr(error)
            _save(out/"acquisition_exception.json",{"case":case,"error":acquisition_failure,"stage2_complete":False})
            results.append({"case_id":case["case_id"],"pass":False,"native_capture_complete":False,
                            "acquisition_exception":acquisition_failure,"stage2_complete":False})
            break
        reports.append(report);results.extend(report["results"])
        allocation_limit_reached=report.get("allocation_limit_reached",False) or time.monotonic()>=deadline
        acquisition_failure=report.get("native_capture_failure")
        if report.get("failure") and report.get("failure_kind")!="native_terminal_prefix":
            acquisition_failure=acquisition_failure or report["failure"]
        elif not report.get("failure") and report.get("acquisition_complete") is not True:
            acquisition_failure=acquisition_failure or "Batch acquisition did not complete"
        if env.native_errors:
            acquisition_failure=acquisition_failure or "Native physics errors prevent another probe"
        _save(out/"summary.json",summary())
        if acquisition_failure or allocation_limit_reached or env.native_errors:
            break
    try:
        if _sha(checkpoint)!=checkpoint_sha or _sha(__file__)!=source_sha or _sha(scoring.__file__)!=scorer_sha:
            acquisition_failure="Learning probe input changed during collection"
    except OSError as error:
        acquisition_failure="Learning probe input unavailable after collection: "+repr(error)
    value=summary();_save(out/"summary.json",value)
    return value
