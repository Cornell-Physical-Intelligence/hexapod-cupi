"""Candidate-only 495/498 diagnostic schema; shared pre-reset scoring helpers."""
from pathlib import Path
import json
import numpy as np
import torch
from omni_diagnostics import diagnostic_options,diagnostic_scenarios,sample_masks,_summary

@torch.inference_mode()
def evaluate_candidate_diagnostics(env, runner, plan, output, checkpoint_sha):
    from tensordict import TensorDict
    from omni_flat_evaluation import save
    options = diagnostic_options(plan["omni"]["diagnostics"])
    scenarios = diagnostic_scenarios()
    if env.num_envs % len(scenarios):
        raise ValueError("Diagnostic environments must contain equal replicas of 12 scenarios")
    replicas = env.num_envs // len(scenarios)
    if options["trace_envs_per_scenario"] > replicas:
        raise ValueError("trace_envs_per_scenario exceeds the available replicas")
    targets = torch.tensor([row["command"] for row in scenarios], device=env.device).repeat_interleave(replicas, 0)
    policy = runner.get_inference_policy(device=env.device)
    env.omni_diagnostic_enabled = True
    env.reset(seed=options["seed"])
    env.episode_length_buf.zero_()
    env.set_evaluation_targets(targets)
    samples = []
    observation_audit = {"actor_width": 495, "critic_width": 498,
                         "max_same_step_repeat_difference": 0., "max_command_slice_difference": 0.,
                         "max_history_shift_difference": 0., "max_executable_state_difference": 0.}
    previous_actor = previous_done = None
    try:
        for step in range(round(options["duration_s"] / env.step_dt)):
            obs = env._get_observations()
            repeat = env._get_observations()
            actor = obs["policy"]
            observation_audit["max_same_step_repeat_difference"] = max(
                observation_audit["max_same_step_repeat_difference"], float((actor - repeat["policy"]).abs().max()))
            scaled_command = env._commands * torch.tensor([5., 5., 2.5], device=env.device)
            observation_audit["max_command_slice_difference"] = max(
                observation_audit["max_command_slice_difference"], float((actor[:, -99+6:-99+9] - scaled_command).abs().max()))
            if previous_actor is not None and (~previous_done).any():
                valid = ~previous_done
                observation_audit["max_history_shift_difference"] = max(
                    observation_audit["max_history_shift_difference"],
                    float((actor[valid, :-99] - previous_actor[valid, 99:]).abs().max()))
            expected_state=env.target_velocity_controller.observable_state(env._robot.data.default_joint_pos.torch)
            observation_audit["max_executable_state_difference"]=max(observation_audit["max_executable_state_difference"],float((actor[:,-36:]-expected_state).abs().max()))
            if actor.shape!=(env.num_envs,495) or obs["critic"].shape!=(env.num_envs,498):raise RuntimeError("Candidate observation schema mismatch")
            previous_actor = actor.clone()
            action = policy(TensorDict(obs, batch_size=[env.num_envs]))
            if options["controller"] == "zero":
                action = torch.zeros_like(action)
            _, _, terminated, truncated, _ = env.step(action)
            previous_done = (terminated | truncated).clone()
            env.set_evaluation_targets(targets)
            sample = env.omni_diagnostic_sample
            if not all(np.isfinite(value).all() for value in sample.values()):
                raise RuntimeError("Nonfinite diagnostic sample")
            # Verify the preserved event sample matches the returned reset flags.
            if not np.array_equal(sample["terminated"], terminated.cpu().numpy()):
                raise RuntimeError("Diagnostic snapshot does not match pre-reset termination events")
            samples.append(sample)
    finally:
        env.omni_diagnostic_enabled = False
    data = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
    joint_names = list(env._robot.joint_names)
    masks = sample_masks(data["age_s"], data["terminated"], options["settle_s"])
    rows = []
    for index, scenario in enumerate(scenarios):
        selected = slice(index * replicas, (index + 1) * replicas)
        subset = {key: value[:, selected] for key, value in data.items()}
        row = {**scenario, "replicas": replicas,
               "terminations": int(subset["terminated"].sum()), "truncations": int(subset["truncated"].sum()),
               "termination_reasons": {key.removeprefix("reason_"): int((value & subset["terminated"]).sum())
                                       for key, value in subset.items() if key.startswith("reason_")},
               "windows": {name: _summary(subset, mask[:, selected], joint_names) for name, mask in masks.items()},
               "per_replica": [{"replica":replica,"windows":{
                   name:_summary({k:v[:,replica:replica+1] for k,v in subset.items()},mask[:,index*replicas+replica:index*replicas+replica+1],joint_names)
                   for name,mask in masks.items()}} for replica in range(replicas)]}
        rows.append(row)
    trace_ids = [index * replicas + replica for index in range(len(scenarios))
                 for replica in range(options["trace_envs_per_scenario"])]
    trace_path = Path(output) / "diagnostic_trace.npz"
    np.savez_compressed(trace_path, **{key: value[:, trace_ids] for key, value in data.items()},
                        trace_env_ids=np.array(trace_ids), joint_names=np.array(joint_names),
                        time_s=(np.arange(len(samples)) + 1) * env.step_dt)
    report = {"complete": True, "kind": "diagnostic_not_qualification", "checkpoint_sha256": checkpoint_sha,
              "stage2_complete":False,"controller":env.candidate_contract,"options": options, "overrides": plan["omni"].get("overrides", {}),
              "reward_weights": env.cfg.omni_reward_weights,
              "joint_names": joint_names, "control_sample_dt_s": env.step_dt,
              "physics_dt_s": env.cfg.sim.dt, "trace_file": trace_path.name,
              "observation_audit": observation_audit, "scenarios": rows,
              "notes": ["State captured in reward calculation before command advance and automatic reset.",
                        "All failure counts retained; post-settle windows exclude each episode's reset transient.",
                        "Computed torque is requested; applied torque is the motor-clipped simulator command.",
                        "Finite-difference velocity spans one control step; reported velocity is its endpoint.",
                        "Control-rate trace cannot identify oscillation above its Nyquist frequency.",
                        "Termination reason counts overlap when multiple causes coincide."]}
    save(Path(output) / "diagnostics.json", report)
    print("OMNI_DIAGNOSTICS_DONE " + json.dumps({"checkpoint": checkpoint_sha,
          "terminations": sum(row["terminations"] for row in rows), "trace": str(trace_path)}), flush=True)
    return report
