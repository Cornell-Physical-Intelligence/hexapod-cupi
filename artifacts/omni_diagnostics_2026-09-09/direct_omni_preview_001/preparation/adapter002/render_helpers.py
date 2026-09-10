"""Reviewed native render warmup and non-colliding ground command helpers."""
import json,math,time
import numpy as np

def draw_live_command(drawing, actual, requested, label):
    """Reuse only non-colliding USD mesh helpers for on-ground commands."""
    x, y, heading = map(float, actual)
    forward, left, yaw = requested
    drawing.text('LiveCommand', f'{label}: FWD {forward:+.3f}  LEFT {left:+.3f} M/S  YAW {yaw:+.2f} RAD/S',
                 x, y + .48, color=(.8, .95, 1.), pitch=.0025)
    travel = []
    if math.hypot(forward, left) > 1e-8:
        travel = [(x, y, heading + math.atan2(left, forward))]
    drawing.arrows('LiveTravelArrow', travel, (.08, .48, 1.), length=.20, z=.010)
    if abs(yaw) > 1e-8:
        angles = np.linspace(heading, heading + math.copysign(math.pi / 2, yaw), 14)
        turn = np.column_stack((x + .34*np.cos(angles), y + .34*np.sin(angles)))
        tip_heading = float(angles[-1] + math.copysign(math.pi / 2, yaw))
        yaw_arrow = [(float(turn[-1, 0]), float(turn[-1, 1]), tip_heading)]
    else:
        turn = np.empty((0, 2))
        yaw_arrow = []
    drawing.line('LiveYawArc', turn, (1., .65, .05), width=.006, z=.011)
    drawing.arrows('LiveYawArrow', yaw_arrow, (1., .65, .05), length=.065, z=.011)

def initial_rgb_frame(env, *, max_attempts=24):
    """Warm the lazily attached RGB annotator with renders only, no physics."""
    attempts = []
    deadline = time.monotonic() + 30.
    for index in range(max_attempts):
        raw = env.render()
        valid = raw is not None and raw.ndim == 3 and np.std(raw) >= 1
        attempts.append(dict(attempt=index + 1, shape=None if raw is None else list(raw.shape),
                             std=None if raw is None or raw.size == 0 else float(np.std(raw)), valid=bool(valid)))
        if valid:
            return raw, dict(render_attempts=attempts, physics_steps=0, ready=True)
        if time.monotonic() >= deadline:
            break
        # DirectRLEnv's first RGB call creates the render product/annotator.
        # A later render is required before data exist; this API advances no physics.
        env.sim.render()
    raise RuntimeError('RGB annotator did not become ready after bounded render-only warmup: ' + json.dumps(attempts))

