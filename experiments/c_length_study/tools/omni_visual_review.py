"""Actual-physics visual coverage of bearings, turns, paths and quiet standing.

Command-only clips deliberately have no position feedback that could conceal
drift. Path-follower clips identify their ideal simulator localization explicitly.
"""

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
import math
from pathlib import Path
import numpy as np
import torch

from experiments.c_length_study.tools.omni_flat_math import integrate_body_twist, slew_commands
from experiments.c_length_study.tools.omni_path_demo import GroundDrawing, actual_pose, path_follower


def visual_review_specs():
    specs = []
    for speed in (.10, .20):
        for bearing in range(16):
            angle = math.pi * bearing / 8
            specs.append({"name": f"bearing_{bearing*22.5:g}_{speed:.2f}",
                          "title": f"TRAVEL {bearing*22.5:g} DEG / {speed:.2f} M/S",
                          "command": [speed*math.cos(angle), speed*math.sin(angle), 0.],
                          "moving_s": 4., "controller": "command"})
    for yaw in (-.4, -.2, .2, .4):
        specs.append({"name": f"turn_{yaw:+.2f}", "title": f"TURN {yaw:+.2f} RAD/S",
                      "command": [0., 0., yaw], "moving_s": 4., "controller": "command"})
    for speed, yaw, label in ((.15, .05, "gentle"), (.20, .4, "tight")):
        for sign in (-1, 1):
            specs.append({"name": f"{label}_arc_{sign:+d}", "title": f"{label.upper()} ARC / YAW {sign*yaw:+.2f}",
                          "command": [speed, 0., sign*yaw], "moving_s": 6., "controller": "command"})
    for sign in (-1, 1):
        specs.append({"name": f"strafe_arc_{sign:+d}", "title": f"STRAFE ARC / YAW {sign*.2:+.2f}",
                      "command": [0., .1, sign*.2], "moving_s": 6., "controller": "command"})
    specs += [
        {"name": "s_curve", "title": "S CURVE / IDEAL POSE FOLLOWER", "moving_s": 10., "controller": "pose_follower"},
        {"name": "fixed_heading_curve", "title": "CURVE WITH FIXED HEADING / IDEAL POSE FOLLOWER", "moving_s": 10., "controller": "pose_follower"},
        {"name": "reversals", "title": "FORWARD / REVERSE / LEFT / RIGHT / STOP", "moving_s": 12., "controller": "command"},
        {"name": "quiet_stand", "title": "ZERO COMMAND / QUIET STAND / NO POSE FOLLOWER", "moving_s": 30., "controller": "command"},
    ]
    return specs


def review_command(spec, time_s):
    local = time_s - 1.
    if local < 0 or local >= spec["moving_s"] or spec["name"] == "quiet_stand":
        return [0., 0., 0.]
    if spec["name"] == "s_curve":
        return [.12, 0., .15 * math.sin(2*math.pi*local/spec["moving_s"])]
    if spec["name"] == "fixed_heading_curve":
        angle = .5*math.pi*local/spec["moving_s"]
        return [.12*math.cos(angle), .12*math.sin(angle), 0.]
    if spec["name"] == "reversals":
        return [[.1, 0., 0.], [-.1, 0., 0.], [0., .1, 0.], [0., -.1, 0.]][min(int(local//3), 3)]
    return spec["command"]


def review_reference(spec, dt, initial_pose):
    pose = initial_pose.clone()
    command = torch.zeros_like(pose)
    poses = []
    commands = []
    requests = []
    for step in range(round((spec["moving_s"] + 2.) / dt)):
        request = torch.tensor(review_command(spec, step*dt), dtype=pose.dtype, device=pose.device).reshape(1, 3)
        command = slew_commands(command, request, dt)
        pose = integrate_body_twist(pose, command, dt)
        requests.append(request[0].clone())
        commands.append(command[0].clone())
        poses.append(pose[0].clone())
    return torch.stack(poses), torch.stack(commands), torch.stack(requests)


@torch.inference_mode()
def record_visual_review(env, runner, plan, output, checkpoint_sha):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from tensordict import TensorDict
    from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
    from experiments.c_length_study.tools.omni_flat_evaluation import save
    if env.num_envs != 1:
        raise ValueError("Visual review requires one full robot per rendered trial")
    if abs(env.step_dt - .02) > 1e-8:
        raise ValueError("This 25 fps recorder requires the established 20 ms control period")
    options = plan["omni"].get("visual_review", {})
    if set(options) - {"seed", "cases"}:
        raise ValueError("Visual review accepts only seed and explicit case subset")
    all_specs = visual_review_specs()
    selected = options.get("cases")
    if selected is not None:
        names = {spec["name"] for spec in all_specs}
        if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or set(selected) - names:
            raise ValueError("Visual-review cases must be unique known names")
        specs = [spec for spec in all_specs if spec["name"] in selected]
    else:
        specs = all_specs
    policy = runner.get_inference_policy(device=env.device)
    output = Path(output)
    global_step = frames = 0
    trials = []
    trace = []
    trace_case = []
    env.omni_diagnostic_enabled = True
    try:
        with imageio.get_writer(str(output / "rollout.mp4"), fps=25, codec="libx264", quality=8) as writer:
            for index, spec in enumerate(specs):
                env.reset(seed=options.get("seed", 37057) + all_specs.index(spec))
                env.episode_length_buf.zero_()
                if env.render() is None:
                    raise RuntimeError("No Isaac RGB output for visual review")
                poses, commands, requests = review_reference(spec, env.step_dt, actual_pose(env).clone())
                drawing = GroundDrawing()
                drawing.reference(poses.cpu().numpy(), commands.cpu().numpy(), spec["title"])
                xy = poses[:, :2].cpu().numpy()
                center = xy.mean(0)
                span = max(float(np.ptp(xy, axis=0).max()), .75)
                set_kit_renderer_camera_view(eye=(float(center[0])+.85, float(center[1])+1.2, max(1.65, span*1.8)),
                                             target=(float(center[0]), float(center[1])+.13, 0.))
                errors = []
                actual_trail = []
                nterm = ntrunc = 0
                first_frame = frames
                for step in range(len(poses)):
                    request = requests[step:step+1]
                    if spec["controller"] == "pose_follower":
                        request = path_follower(actual_pose(env), poses[step:step+1], commands[step:step+1])
                    env.set_evaluation_targets(request)
                    obs = TensorDict(env._get_observations(), batch_size=[1])
                    _, _, terminated, truncated, _ = env.step(policy(obs))
                    env.set_evaluation_targets(request)
                    snapshot = env.omni_diagnostic_sample
                    trace.append(snapshot)
                    trace_case.append(index)
                    nterm += int(terminated[0])
                    ntrunc += int(truncated[0])
                    actual = actual_pose(env)
                    error = float((actual[0, :2] - poses[step, :2]).norm())
                    errors.append(error)
                    actual_trail.append(actual[0, :2].cpu().numpy())
                    global_step += 1
                    if global_step % 2 == 0:
                        if step % 10 == 1:
                            drawing.line("ActualTrail", np.array(actual_trail), (1., .25, .08), width=.007, z=.008)
                        raw = env.render()
                        if raw is None or raw.ndim != 3 or float(np.std(raw)) < 1:
                            raise RuntimeError("Blank visual-review frame")
                        frame = Image.fromarray(raw[:, :, :3])
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((8, 8, 1150, 83), fill=(10, 18, 28))
                        draw.text((18, 17), f"{index+1}/{len(specs)}  {spec['title']}", fill="white")
                        mode = "COMMAND ONLY: no position-feedback correction" if spec["controller"] == "command" else "PATH FOLLOWER: ideal simulator localization"
                        draw.text((18, 39), mode + f" | failures {nterm+ntrunc} | path error {error:.3f} m", fill="white")
                        draw.text((18, 61), "Actual PPO joint control and contact physics | independent trials reset at labeled boundaries", fill="white")
                        writer.append_data(np.asarray(frame))
                        frames += 1
                trials.append({**spec, "start_frame": first_frame, "end_frame_exclusive": frames,
                               "duration_s": len(poses)*env.step_dt, "terminations": nterm, "truncations": ntrunc,
                               "mean_path_error_m": float(np.mean(errors)), "p95_path_error_m": float(np.quantile(errors, .95)),
                               "visual_review_status": "awaiting_human_or_agent_frame_and_video_inspection"})
                save(output / "visual_review_progress.json", {"complete": False, "checkpoint_sha256": checkpoint_sha,
                                                              "completed_cases": trials, "frames": frames})
    finally:
        env.omni_diagnostic_enabled = False
    np.savez_compressed(output / "visual_review_trace.npz",
                        **{key: np.stack([row[key][0] for row in trace]) for key in trace[0]},
                        case_index=np.array(trace_case), joint_names=np.array(env._robot.joint_names),
                        time_s=(np.arange(len(trace))+1)*env.step_dt)
    report = {"complete": True, "kind": "all_direction_visual_review_v1", "checkpoint_sha256": checkpoint_sha,
              "fps": 25, "frames": frames, "duration_s": global_step*env.step_dt,
              "all_cases_recorded": len(specs) == len(all_specs), "cases": trials,
              "stage2_complete": False, "trace_file": "visual_review_trace.npz",
              "ground_annotations": {"blue": "reference path/travel", "gold": "body heading", "orange": "actual path"},
              "notes": ["Recording all cases does not itself establish smoothness or pass qualification.",
                        "Short command clips inspect motion; separate quiet/stop evaluator establishes sustained stillness.",
                        "Accepted forward benchmark and current straight trial remain visual comparison references."]}
    save(output / "visual_review.json", report)
    save(output / "video.json", report)
    return report
