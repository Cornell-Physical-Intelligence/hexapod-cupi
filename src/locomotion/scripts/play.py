"""Evaluate a PPO or AMP checkpoint on the admitted canonical simulator."""

from __future__ import annotations

import argparse
import json
import time

from locomotion.evaluation.camera import NativePolicyCamera
from locomotion.evaluation.checkpoint import (
    require_compatibility_configuration,
    require_embedded_declaration,
    require_learner_configuration,
)
from locomotion.scripts.cli import (
    add_app_arguments,
    add_common_arguments,
    add_learner_arguments,
    learner_config,
    print_progress,
    resolve_options,
    run,
)


def parse_args(argv=None):
    """Parse the command line and resolve the recipe and the evaluation gait-clock default."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_common_arguments(parser)
    add_learner_arguments(parser)
    add_app_arguments(parser, argv)
    return resolve_options(parser.parse_args(argv), "evaluate")


def actor_input(env, args, observation, tracker=None):
    """Scale the actor observation and append the gait clock and the destination inputs, as the training
    wrapper does. ``tracker`` follows the destination once per control; None appends the neutral inputs."""
    import torch

    from locomotion.omniwalk.mdp.destination import NEUTRAL_FEATURES
    from locomotion.rl.wrapper import clock_features, scale_observation

    scaled = scale_observation(observation, args.observation_scaling)
    if args.gait_clock:
        clock = clock_features(env.episode_steps, args.gait_clock, observation[:, 210:213]).to(
            scaled
        )
        scaled = torch.cat((scaled, clock), -1)
    if args.experiment is not None:
        extra = (
            torch.tensor(NEUTRAL_FEATURES).expand(len(scaled), -1)
            if tracker is None
            else tracker.inputs(env.current["root"], observation[:, 210:213], env.episode_steps)
        )
        scaled = torch.cat((scaled, extra.to(scaled)), -1)
    return scaled


def evaluation_policy(env, args, config, identity, checkpoint_record, compatibility):
    """Check the checkpoint configuration, load its actor and return the policy that the evaluator calls."""
    import torch
    from tensordict import TensorDict

    from locomotion.rl.ppo import load_actor

    check = (
        require_compatibility_configuration
        if compatibility is not None
        else require_learner_configuration
    )
    check(checkpoint_record, identity, config)
    if config.get("environment_wrapper", {}).get("destination_input") == "ghost_v10":
        raise ValueError(
            "Native evaluation does not build the ghost_v10 inputs yet; "
            "evaluate this checkpoint with locomotion.surrogate.evaluate"
        )
    if config.get("environment_wrapper", {}).get("gait_clock_scaling") is not None:
        raise ValueError(
            "Native evaluation does not follow a speed-scaled gait clock yet; "
            "evaluate this checkpoint with locomotion.surrogate.evaluate"
        )
    observation = env.observations(env.current)["obs"]
    scaled = actor_input(env, args, observation)
    if args.networks == "paper":
        from locomotion.amp.networks import actor_observation

        observations = actor_observation(observation)
    else:
        observations = TensorDict({"policy": scaled}, batch_size=[env.num_envs])
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    require_embedded_declaration(payload.get("infos"), checkpoint_record)
    actor, _ = load_actor(payload, config, observations, args.device)
    if args.networks == "paper":

        def evaluate(observation):
            return actor(actor_observation(observation))
    else:
        from locomotion.omniwalk.mdp.destination import DestinationTracker

        tracker = (
            DestinationTracker(env.num_envs, args.device) if args.experiment is not None else None
        )

        def evaluate(observation):
            return actor(
                TensorDict(
                    {"policy": actor_input(env, args, observation, tracker)},
                    batch_size=[env.num_envs],
                )
            )

    return smoothed_policy(evaluate, env, args.action_smoothing)


def smoothed_policy(evaluate, env, action_smoothing):
    """Wrap the actor call with the training wrapper's two-control mean, restarted with each episode."""
    import torch

    from locomotion.rl.wrapper import smoothed_action

    previous = None

    def policy(observation):
        nonlocal previous
        with torch.inference_mode():
            action = evaluate(observation)
            if action_smoothing == "mean2":
                last = torch.zeros_like(action) if previous is None else previous
                previous = action.clone()
                action = smoothed_action(action, last, env.episode_steps)
            return action

    return policy


def run_evaluation(env, args, policy, state, started):
    """Score the policy on the selected cases, then require a complete set of load summaries."""
    from locomotion.amp.features import extract_features
    from locomotion.evaluation.runner import evaluate_suite, run_learning_probe_suite

    env.render = NativePolicyCamera(env)
    remaining = max(0.001, args.max_wall_seconds - (time.monotonic() - started))
    options = dict(
        progress=print_progress, max_wall_seconds=remaining, motion_features=extract_features
    )
    if args.eval_scope == "full":
        result = evaluate_suite(
            env,
            policy,
            args.output / "evaluation",
            args.geometry_extrema,
            args.checkpoint,
            **options,
        )
    else:
        selected = (
            ["learning:translate_0.05_0deg", "learning:quiet_20s", "learning:forward_0.05_to_stop"]
            if args.eval_scope == "focus"
            else None
        )
        result = run_learning_probe_suite(
            env,
            policy,
            args.output / "evaluation",
            args.geometry_extrema,
            args.checkpoint,
            selected_case_ids=selected,
            record_video=True,
            video_case_id=args.video_case,
            seed=args.seed,
            **options,
        )
    state["evaluation"] = result
    summaries = list((args.output / "evaluation").rglob("force_metrics.json"))
    if not summaries or any(json.loads(p.read_text())["status"] != "available" for p in summaries):
        raise ValueError("Evaluation has no complete set of load summaries")


def evaluate(env, session):
    """Load the checkpoint actor and score it on the selected cases."""
    args = session.args
    config = learner_config(args, session.identity, session.state)
    policy = evaluation_policy(
        env, args, config, session.identity, session.checkpoint_record, session.compatibility
    )
    run_evaluation(env, args, policy, session.state, session.started)


def main(argv=None):
    return run(parse_args(argv), "evaluate", evaluate)


if __name__ == "__main__":
    raise SystemExit(main())
