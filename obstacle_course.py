"""Isaac Sim standalone script: build a walled obstacle course with randomly placed obstacles.

The course is a corridor along +X with a green start pad and a red goal pad. Between them it
places a random mix of:
    - walls      static boxes with random size and yaw
    - pillars    static cylinders
    - bumps      static half-buried cylinders lying across the path (speed bumps)
    - crates     dynamic boxes the robot can push
    - balls      dynamic spheres that roll around

Obstacles keep at least --gap metres between their footprints, so a robot narrower than the gap
can always squeeze between any two of them.

Usage:
    source ~/Desktop/env_isaacsim/bin/activate
    python obstacle_course.py                       # with GUI, random layout
    python obstacle_course.py --seed 42             # reproducible layout
    python obstacle_course.py --num-obstacles 40 --length 30
    python obstacle_course.py --robot               # also drop a Jetbot at the start pad
    python obstacle_course.py --drive --livestream  # drive the Jetbot through the course with the keyboard

Drive controls (--drive; click the viewport first so it has keyboard focus):
    Arrow keys / WASD   forward, back, turn left, turn right
    Shift               hold for double speed
    C                   toggle between the chase camera and a free camera (orbit with the mouse)
    R                   put the robot back on the start pad and reset the movable obstacles
    python obstacle_course.py --headless            # no window
    python obstacle_course.py --livestream          # stream over WebRTC (use when connected over SSH)
    python obstacle_course.py --capture renders     # headless: save PNG views of the course to ./renders
"""

import argparse
import math
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run without a window")
parser.add_argument("--livestream", action="store_true", help="Stream over WebRTC instead of opening a window")
parser.add_argument("--steps", type=int, default=300, help="Number of frames to simulate")
parser.add_argument("--seed", type=int, default=None, help="Random seed (omit for a new layout every run)")
parser.add_argument("--num-obstacles", type=int, default=20, help="Number of obstacles to place")
parser.add_argument("--length", type=float, default=20.0, help="Course length in metres (along X)")
parser.add_argument("--width", type=float, default=6.0, help="Course width in metres (along Y)")
parser.add_argument("--gap", type=float, default=0.6, help="Minimum free space between obstacles in metres")
parser.add_argument("--robot", action="store_true", help="Spawn a Jetbot at the start (needs asset server access)")
parser.add_argument("--drive", action="store_true", help="Spawn a Jetbot and drive it with the keyboard (implies --robot)")
parser.add_argument("--capture", metavar="DIR", default=None, help="Render PNG views of the course into DIR and exit (implies --headless)")
args = parser.parse_args()
if args.capture:
    args.headless, args.livestream = True, False
if args.drive:
    args.robot = True

# SimulationApp must be created before importing any other Isaac Sim / Omniverse modules
from isaacsim import SimulationApp

if args.livestream:
    # The streaming extensions bundle their own charset_normalizer, which clashes with the venv's copy
    # if both get imported (breaking isaacsim.core.experimental.*). Load the venv's first so it wins.
    import charset_normalizer  # noqa: F401

    # Use the streaming experience; EXP_PATH is set when the isaacsim package is imported
    experience = os.path.join(os.environ["EXP_PATH"], "isaacsim.exp.full.streaming.kit")
    simulation_app = SimulationApp({"headless": True, "hide_ui": False}, experience=experience)
else:
    simulation_app = SimulationApp({"headless": args.headless})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import Cube, Cylinder, DomeLight, GroundPlane, Sphere
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim

seed = args.seed if args.seed is not None else random.randrange(1_000_000)
rng = random.Random(seed)
print(f"Building obstacle course with seed {seed}")

START_CLEAR = 2.0  # metres kept free at each end for the start / goal pads
WALL_HEIGHT = 0.5
WALL_THICKNESS = 0.2


def yaw_quat(yaw):
    """Quaternion (wxyz) for a rotation of `yaw` radians about +Z."""
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def static_box(path, center, size, color, yaw=0.0):
    """Unit cube scaled to `size` (x, y, z), with collision but no rigid body, so it never moves."""
    Cube(path, sizes=1.0, colors=color, positions=[center], orientations=[yaw_quat(yaw)], scales=[size])
    GeomPrim(path).apply_collision_apis()


# ---------------------------------------------------------------------------------------------
# Build the scene
# ---------------------------------------------------------------------------------------------
stage_utils.create_new_stage()
stage_utils.define_prim("/World", "Xform")
stage_utils.define_prim("/World/PhysicsScene", "PhysicsScene")
stage_utils.define_prim("/World/Course", "Xform")
stage_utils.define_prim("/World/Course/Obstacles", "Xform")

GroundPlane("/World/GroundPlane")
DomeLight("/World/DomeLight").set_intensities(1000.0)

# Boundary walls: the corridor spans x in [0, length], y in [-width/2, width/2]
L, W = args.length, args.width
static_box("/World/Course/WallLeft", [L / 2, W / 2 + WALL_THICKNESS / 2, WALL_HEIGHT / 2], [L, WALL_THICKNESS, WALL_HEIGHT], "#606060")
static_box("/World/Course/WallRight", [L / 2, -W / 2 - WALL_THICKNESS / 2, WALL_HEIGHT / 2], [L, WALL_THICKNESS, WALL_HEIGHT], "#606060")
static_box("/World/Course/WallBack", [-WALL_THICKNESS / 2, 0.0, WALL_HEIGHT / 2], [WALL_THICKNESS, W + 2 * WALL_THICKNESS, WALL_HEIGHT], "#606060")
static_box("/World/Course/WallFront", [L + WALL_THICKNESS / 2, 0.0, WALL_HEIGHT / 2], [WALL_THICKNESS, W + 2 * WALL_THICKNESS, WALL_HEIGHT], "#606060")

# Start / goal pads: thin coloured tiles with no collision so the robot drives over them freely
Cube("/World/Course/StartPad", sizes=1.0, colors="#22aa44", positions=[[START_CLEAR / 2, 0.0, 0.005]], scales=[[START_CLEAR * 0.8, W * 0.8, 0.01]])
Cube("/World/Course/GoalPad", sizes=1.0, colors="#cc2222", positions=[[L - START_CLEAR / 2, 0.0, 0.005]], scales=[[START_CLEAR * 0.8, W * 0.8, 0.01]])

# ---------------------------------------------------------------------------------------------
# Random obstacles
# ---------------------------------------------------------------------------------------------
# Each entry: (kind, relative weight). Footprint radius is computed per obstacle for spacing.
OBSTACLE_KINDS = [("wall", 3), ("pillar", 3), ("bump", 1), ("crate", 2), ("ball", 1)]
placed = []  # (x, y, footprint_radius) of every obstacle placed so far
dynamic_paths = []  # obstacles with rigid bodies, so they can be reset later


def try_place(radius, max_tries=200):
    """Rejection-sample an (x, y) inside the course that keeps `args.gap` clear of everything else."""
    x_min, x_max = START_CLEAR + radius, L - START_CLEAR - radius
    y_min, y_max = -W / 2 + radius, W / 2 - radius
    if x_min >= x_max or y_min >= y_max:
        return None
    for _ in range(max_tries):
        x, y = rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)
        if all(math.hypot(x - px, y - py) >= radius + pr + args.gap for px, py, pr in placed):
            placed.append((x, y, radius))
            return x, y
    return None


kinds, weights = zip(*OBSTACLE_KINDS)
counts = {kind: 0 for kind in kinds}
for i in range(args.num_obstacles):
    kind = rng.choices(kinds, weights)[0]
    path = f"/World/Course/Obstacles/{kind.capitalize()}_{i:02d}"
    color = "#{:02x}{:02x}{:02x}".format(*(rng.randint(60, 230) for _ in range(3)))
    yaw = rng.uniform(0, math.pi)

    if kind == "wall":
        size = [rng.uniform(0.8, 2.0), rng.uniform(0.15, 0.3), rng.uniform(0.3, 0.8)]
        pos = try_place(math.hypot(size[0], size[1]) / 2)
        if pos is None:
            continue
        static_box(path, [pos[0], pos[1], size[2] / 2], size, color, yaw)

    elif kind == "pillar":
        radius, height = rng.uniform(0.1, 0.35), rng.uniform(0.5, 1.5)
        pos = try_place(radius)
        if pos is None:
            continue
        Cylinder(path, radii=radius, heights=height, axes="Z", colors=color, positions=[[pos[0], pos[1], height / 2]])
        GeomPrim(path).apply_collision_apis()

    elif kind == "bump":
        # Cylinder lying on its side, sunk into the ground so only a low hump sticks out
        radius, length = rng.uniform(0.05, 0.12), rng.uniform(1.0, 2.5)
        pos = try_place(length / 2)
        if pos is None:
            continue
        Cylinder(path, radii=radius, heights=length, axes="Y", colors=color, positions=[[pos[0], pos[1], radius * 0.4]], orientations=[yaw_quat(rng.uniform(-0.4, 0.4))])
        GeomPrim(path).apply_collision_apis()

    elif kind == "crate":
        side = rng.uniform(0.2, 0.5)
        pos = try_place(side * math.sqrt(2) / 2)
        if pos is None:
            continue
        Cube(path, sizes=side, colors=color, positions=[[pos[0], pos[1], side / 2]], orientations=[yaw_quat(yaw)])
        GeomPrim(path).apply_collision_apis()
        RigidPrim(path, masses=[rng.uniform(0.5, 3.0)])
        dynamic_paths.append(path)

    elif kind == "ball":
        radius = rng.uniform(0.1, 0.25)
        pos = try_place(radius)
        if pos is None:
            continue
        Sphere(path, radii=radius, colors=color, positions=[[pos[0], pos[1], radius]])
        GeomPrim(path).apply_collision_apis()
        RigidPrim(path, masses=[rng.uniform(0.2, 1.0)])
        dynamic_paths.append(path)

    counts[kind] += 1

total = sum(counts.values())
print(f"Placed {total}/{args.num_obstacles} obstacles: " + ", ".join(f"{n} {k}s" for k, n in counts.items()))
if total < args.num_obstacles:
    print("  (ran out of room for the rest; try a longer/wider course or a smaller --gap)")

# Optional robot at the start pad, facing the goal
ROBOT_START = [START_CLEAR / 2, 0.0, 0.05]
robot = None
if args.robot:
    from isaacsim.storage.native import get_assets_root_path

    assets_root = get_assets_root_path()
    if assets_root is None:
        print("Could not reach the Isaac Sim asset server; skipping the robot")
        args.drive = False
    elif args.drive:
        from isaacsim.robot.experimental.wheeled_robots.controllers import DifferentialController
        from isaacsim.robot.experimental.wheeled_robots.robots import WheeledRobot

        robot = WheeledRobot(
            "/World/Jetbot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            usd_path=assets_root + "/Isaac/Robots_Multiphysics/NVIDIA/Jetbot/jetbot.usda",
            positions=[ROBOT_START],
        )
        # Jetbot wheel radius and track width in metres
        controller = DifferentialController(wheel_radius=0.0335, wheel_base=0.1125)
    else:
        stage_utils.add_reference_to_stage(assets_root + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd", "/World/Jetbot")
        from isaacsim.core.experimental.prims import XformPrim

        XformPrim("/World/Jetbot", positions=[ROBOT_START], reset_xform_op_properties=True)

dynamic = RigidPrim(dynamic_paths) if dynamic_paths else None

# Cameras for --capture: name -> (camera position, look-at point)
if args.capture:
    import omni.replicator.core as rep
    from PIL import Image

    views = {
        "overview": ((-4.0, -W - 3.0, 7.0), (L / 2, 0.0, 0.0)),
        "topdown": ((L / 2, -0.01, (L / 2 + 1) / math.tan(math.radians(23)) + 1), (L / 2, 0.0, 0.0)),
        "start": ((-1.5, 0.0, 1.5), (L / 2, 0.0, 0.0)),
        "goal": ((L + 1.5, 0.0, 2.0), (L / 2, 0.0, 0.0)),
    }
    annotators = {}
    for name, (position, look_at) in views.items():
        camera = rep.create.camera(position=position, look_at=look_at)
        annotators[name] = rep.AnnotatorRegistry.get_annotator("rgb")
        annotators[name].attach(rep.create.render_product(camera, (1600, 900)))

# ---------------------------------------------------------------------------------------------
# Run the simulation
# ---------------------------------------------------------------------------------------------
app_utils.play()
for _ in range(args.steps):
    simulation_app.update()

if args.capture:
    # Let the renderer settle (the first frames are often blank or noisy), then save each view
    for _ in range(30):
        simulation_app.update()
    os.makedirs(args.capture, exist_ok=True)
    for name, annotator in annotators.items():
        path = os.path.join(args.capture, f"{name}.png")
        Image.fromarray(annotator.get_data()[:, :, :3]).save(path)
        print(f"Saved {os.path.abspath(path)}")
    simulation_app.close()
    raise SystemExit

if dynamic is not None:
    # Remember where the movable obstacles settled so they can be put back if knocked away
    rest_positions, rest_orientations = dynamic.get_world_poses()
    rest_positions, rest_orientations = rest_positions.numpy(), rest_orientations.numpy()



def reset_obstacles():
    if dynamic is not None:
        n = len(dynamic_paths)
        dynamic.set_world_poses(positions=rest_positions, orientations=rest_orientations)
        dynamic.set_velocities(linear_velocities=[[0.0, 0.0, 0.0]] * n, angular_velocities=[[0.0, 0.0, 0.0]] * n)


if args.drive:
    import carb.input
    import numpy as np
    import omni.appwindow
    import time
    from isaacsim.core.rendering_manager import ViewportManager

    BASE_SPEED, BASE_TURN = 1.0, 3.0  # m/s, rad/s
    # Ramp speed changes over several frames (m/s per frame); sudden jumps flip the Jetbot over
    SPEED_UP, SLOW_DOWN = 0.03, 0.05
    MAX_SIDEWAYS_ACCEL = 2.0  # m/s^2; turning harder than this at speed rolls the robot over
    K = carb.input.KeyboardInput
    FORWARD, BACK, LEFT, RIGHT = {K.UP, K.W}, {K.DOWN, K.S}, {K.LEFT, K.A}, {K.RIGHT, K.D}
    SHIFT = {K.LEFT_SHIFT, K.RIGHT_SHIFT}
    held = set()
    state = {"chase": True, "reset": False, "speed": 0.0}

    def on_key(event, *_):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            held.add(event.input)
            if event.input == K.C:
                state["chase"] = not state["chase"]
                print("Camera: " + ("chase" if state["chase"] else "free (orbit with the mouse)"))
            elif event.input == K.R:
                state["reset"] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            held.discard(event.input)
        return True

    input_iface = carb.input.acquire_input_interface()
    keyboard = omni.appwindow.get_default_app_window().get_keyboard()
    key_sub = input_iface.subscribe_to_keyboard_events(keyboard, on_key)
    camera = "/OmniverseKit_Persp"
    ViewportManager.set_camera(camera)

    def reset_robot():
        robot.set_world_poses(positions=[ROBOT_START], orientations=[[1.0, 0.0, 0.0, 0.0]])
        robot.set_velocities(linear_velocities=[[0.0, 0.0, 0.0]], angular_velocities=[[0.0, 0.0, 0.0]])
        robot.set_dof_velocities(np.zeros((1, robot.num_dofs), dtype=np.float32))

    # The Jetbot asset caps its wheels at 12.56 rad/s (~0.4 m/s); raise it so Shift gives ~2 m/s
    robot.set_dof_max_velocities(80.0)
    print("Drive with the arrow keys or WASD (hold Shift for speed), C toggles the camera, R resets")
    start_time, finished = time.monotonic(), False

# Keep running until the user closes the app; every 1800 frames (~30 s) put movable obstacles back
# (in --drive mode they only go back when you press R)
frame = 0
while (not args.headless or args.livestream) and simulation_app.is_running():
    if args.drive:
        boost = 2.0 if held & SHIFT else 1.0
        target = BASE_SPEED * boost * ((1 if held & FORWARD else 0) - (1 if held & BACK else 0))
        step = SPEED_UP if abs(target) > abs(state["speed"]) and target * state["speed"] >= 0 else SLOW_DOWN
        state["speed"] += max(-step, min(step, target - state["speed"]))
        linear = state["speed"]
        max_turn = min(BASE_TURN, MAX_SIDEWAYS_ACCEL / max(abs(linear), 1e-3))
        angular = max_turn * ((1 if held & LEFT else 0) - (1 if held & RIGHT else 0))
        robot.apply_wheel_actions(controller.forward([linear, angular]))

        position, orientation = (a.numpy()[0] for a in robot.get_world_poses())
        if state["chase"]:
            # Camera sits behind and above the robot, looking slightly ahead of it
            w, z = orientation[0], orientation[3]
            heading = math.atan2(2 * w * z, 1 - 2 * z * z)
            forward = np.array([math.cos(heading), math.sin(heading), 0.0])
            ViewportManager.set_camera_view(camera, eye=position - 1.2 * forward + [0, 0, 0.7], target=position + 0.8 * forward)

        if state["reset"]:
            state["reset"], state["speed"], finished = False, 0.0, False
            reset_robot()
            reset_obstacles()
            start_time = time.monotonic()
            print("Reset to the start pad")
        elif not finished and position[0] > L - START_CLEAR:
            finished = True
            print(f"Reached the goal in {time.monotonic() - start_time:.1f} s! Press R to go again")

    simulation_app.update()
    frame += 1
    if not args.drive and frame % 1800 == 0:
        reset_obstacles()

simulation_app.close()
