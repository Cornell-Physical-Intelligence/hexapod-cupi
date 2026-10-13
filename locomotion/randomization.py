"""Table II domain randomization: the frozen ``table2_v1`` declaration and its per-replica draws.

Liu et al. (arXiv 2511.03167, Table II, p. 3) randomize eight physical parameters between
episodes. This module declares the project's ranges for the approved 7.466 kg direct-drive robot
and the explicit motor law of ``env.py``, records each row as kept, adapted or excluded with its
reason, and draws one replica's conditions from that replica's own generator. ``declaration()``
gives the canonical JSON and SHA-256 that run identities, packs and admission bind. The
environments apply the draws; this module imports no simulator. ``none`` is the nominal
declaration: its draw equals the nominal properties, so an environment under ``none`` steps as it
did before this module existed. docs/TRAINING.md, "Step 7", explains each row.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from .env_config import BODY_NAMES, JOINT_NAMES, KD, MASS_KG, MODEL_SHA256

SCHEMA = "hexapod_table2_randomization_v1"
NAMES = ("none", "table2_v1")
KP = 12.0  # The explicit servo gain of env.motor_force.
TORQUE_CAP_NM = 1.6  # The provisional software cap of env.motor_force; the randomization never raises it.
PAPER_MASS_KG = 25.5
PAYLOAD_MAX_KG = MASS_KG * 5. / PAPER_MASS_KG
JOINT_GROUPS = ("coxa_yaw", "femur_pitch", "tibia_pitch")
NOMINAL_POSE_RAD = (0., -.3, .4)  # stance.json nominal_joint_position_rad, one value per joint group.
CONTROL_DT_S = .02
SUBSTEPS = 8
PRIVILEGED_WRENCH = (("force_body_n", 3), ("torque_body_nm", 3))


@dataclass(frozen=True)
class Row:
    """One Table II row: its status, this project's range and how the draw is shared."""
    status: str  # kept, adapted or excluded
    low: float
    high: float
    unit: str
    granularity: str  # shared, per_joint or per_link
    paper: str
    reason: str


@dataclass(frozen=True)
class JointPositionRow:
    """Additive reset-pose offsets, one half-width per joint group."""
    status: str
    half_width_rad: tuple  # (coxa_yaw, femur_pitch, tibia_pitch)
    unit: str
    granularity: str
    paper: str
    reason: str


@dataclass(frozen=True)
class PushSchedule:
    """Force pushes during training: an adaptation, because the paper gives no training schedule."""
    status: str
    first_delay_s: tuple
    interval_s: tuple
    duration_s: float
    magnitude_n: tuple
    direction: str
    application_point: str
    privileged: str
    paper: str
    reason: str


@dataclass(frozen=True)
class Table2:
    """One field per Table II row plus the push schedule. Instances are frozen declarations."""
    name: str
    joint_stiffness: Row
    joint_damping: Row
    joint_position: JointPositionRow
    link_mass: Row
    payload_mass: Row
    payload_position: Row
    foot_friction: Row
    motor_strength: Row
    push: PushSchedule
    draw_order: tuple
    notes: tuple


TABLE2_V1 = Table2(
    name="table2_v1",
    joint_stiffness=Row(
        "kept", .8, 1.2, "factor on the explicit servo Kp 12", "shared",
        "[0.8, 1.2] x 100",
        "One factor per replica scales Kp in the explicit motor law. The base value is this "
        "project's Kp 12, applied in software; the paper's 100 is its own controller gain."),
    joint_damping=Row(
        "kept", .8, 1.2, "factor on the 18 named KD values", "shared",
        "[0.8, 1.2] x 2",
        "One factor per replica scales every named KD value. The implicit PhysX drive and the "
        "MuJoCo joint damping stay at zero; the factor acts through the explicit motor law alone."),
    joint_position=JointPositionRow(
        "adapted", (.12, .12, .16), "rad added to the nominal reset pose", "per_joint",
        "[0.6, 1.4] x nominal value",
        "The paper scales the default joint position at reset. The six coxa yaw joints have a "
        "0.0 rad nominal angle, so a factor never moves them. Each joint draws an additive offset "
        "instead: 0.4 x |nominal| keeps the paper's extent on the femur (0.3 rad) and tibia "
        "(0.4 rad), and the coxa takes the femur half-width. Every offset stays inside the named "
        "travel limits (the tightest coxa limit is 0.8153 rad). The reset root height places the "
        "lowest drawn toe at the nominal 5 mm clearance. The actor's pose reference, the held "
        "target and the executed-action feature keep the nominal pose."),
    link_mass=Row(
        "kept", .9, 1.1, "factor on each link mass and inertia tensor", "per_link",
        "[0.9, 1.1] x nominal value",
        "Each of the 19 links draws its own factor. The inertia tensor scales by the same factor "
        "and the centre of mass stays, so the link keeps its shape. The baseline is the approved "
        "mass ledger, including the nominal motor corrections."),
    payload_mass=Row(
        "adapted", 0., PAYLOAD_MAX_KG, "kg", "shared",
        "[0, 5] kg",
        "The paper's 5 kg is 19.6 percent of its 25.5 kg robot; the same share of this 7.466 kg "
        "robot is 1.464 kg. The payload is a point mass added to the base link: mass, centre of "
        "mass and the parallel-axis inertia change; no body is added, so the 19-body topology stays."),
    payload_position=Row(
        "kept", -.15, .15, "m from the base origin, X and Y", "shared",
        "[-0.15, 0.15] m relative to base position",
        "X and Y draw independently; Z holds 0.04 m above the base origin, a declared deck "
        "assumption above the base centre of mass (0.025 m). A point mass carries no own inertia."),
    foot_friction=Row(
        "kept", .1, 2.5, "effective contact coefficient", "shared",
        "[0.1, 2.5]",
        "One coefficient per replica for all six toes and that replica's floor. MuJoCo takes the "
        "larger of the two geom coefficients, so the floor and toe geoms both carry the draw; PhysX "
        "multiplies the toe material by the floor material 1.0. Each gives the drawn value."),
    motor_strength=Row(
        "kept", .8, 1.2, "factor on the speed-dependent effort ceiling", "per_joint",
        "[0.8, 1.2]",
        "Each joint draws its own factor. The ceiling becomes min(factor x curve(speed), 1.6 N.m); "
        "the cap never rises. The nominal curve exceeds 1.6 N.m below 450 rpm, so a factor above 1 "
        "changes nothing below about 456 rpm and the row is asymmetric: 0.8 lowers the cap to 1.28 "
        "N.m everywhere, 1.2 only lifts the tail above 456 rpm. The 1.6 N.m saturation counter and "
        "the over-rating reward terms keep their nominal reference."),
    push=PushSchedule(
        status="adapted",
        first_delay_s=(1., 10.),
        interval_s=(5., 10.),
        duration_s=.2,
        magnitude_n=(0., 15.),
        direction="uniform heading in the body XY plane; no vertical component",
        application_point="base link centre of mass; the push adds no torque",
        privileged="six values: body-frame XYZ force in N, then XYZ torque in N.m, each the mean over "
                   "the eight physics steps of the control; zero outside a push and after a reset",
        paper="Section V-B applies velocity perturbations at one-second intervals for robustness "
              "evaluation; no training schedule is given",
        reason="A force over 0.2 s is the push that both simulators apply through one call. At 15 N the "
               "velocity change is 0.40 m/s on the 7.466 kg robot, half the paper's 0.803 m/s "
               "tolerance. Each replica draws its own delay, interval, magnitude and heading."),
    draw_order=("joint_stiffness", "joint_damping", "joint_position[18]", "link_mass[19]", "payload_mass",
                "payload_position[2]", "foot_friction", "motor_strength[18]"),
    notes=(
        "Uniform draws. Each replica owns one generator seeded from the run seed, the replica index "
        "and this declaration's hash; a reset redraws that replica alone and advances no task or "
        "learner generator.",
        "Draws apply at reset and hold for the episode. Pushes draw from the same generator after the "
        "reset draw.",
        "No clipping: every range lies inside the named joint limits, the mass ledger stays positive "
        "and the motor cap bounds the ceiling.",
        "The 0.040 rad / 20 ms target limiter, the 0.35 rad action scale and the 1.6 N.m cap stay fixed.",
    ))

TABLES = {"table2_v1": TABLE2_V1}


def base_values():
    """The nominal values each row scales or offsets, bound by hash."""
    return {"model_sha256": MODEL_SHA256, "mass_kg": MASS_KG, "kp": KP, "kd": list(KD),
            "torque_cap_nm": TORQUE_CAP_NM, "payload_max_kg": PAYLOAD_MAX_KG, "paper_mass_kg": PAPER_MASS_KG,
            "joint_names": list(JOINT_NAMES), "body_names": list(BODY_NAMES),
            "joint_groups": list(JOINT_GROUPS), "nominal_pose_rad": list(NOMINAL_POSE_RAD),
            "control_dt_s": CONTROL_DT_S, "substeps": SUBSTEPS,
            "privileged_wrench": [{"name": n, "width": w} for n, w in PRIVILEGED_WRENCH]}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def declaration(name="none"):
    """Canonical JSON of a declaration and its SHA-256. ``none`` declares the nominal properties."""
    if name not in NAMES:
        raise ValueError("Unknown randomization: " + str(name))
    value = {"schema": SCHEMA, "name": name, "base": base_values()}
    if name == "none":
        value["scope"] = "Nominal properties; no draw, no push."
    else:
        value["table"] = asdict(TABLES[name])
    text = canonical(value)
    return {"name": name, "json": text, "sha256": hashlib.sha256(text.encode()).hexdigest()}


def identity(name="none"):
    """The fields a run identity, pack or admission records for a declaration."""
    record = declaration(name)
    return {"name": name, "declaration_sha256": record["sha256"], "schema": SCHEMA}


@dataclass(frozen=True)
class ReplicaDraw:
    """One replica's conditions. Arrays follow JOINT_NAMES and BODY_NAMES order."""
    kp_scale: float
    kd_scale: float
    joint_offset_rad: np.ndarray
    link_mass_scale: np.ndarray
    payload_mass_kg: float
    payload_position_m: np.ndarray
    foot_friction: float
    motor_strength: np.ndarray

    def as_json(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items()}


def nominal_draw():
    return ReplicaDraw(1., 1., np.zeros(18), np.ones(19), 0., np.array([0., 0., .04]), 1., np.ones(18))


@dataclass(frozen=True)
class Push:
    delay_s: float  # From the previous push end, or from the reset for the first push.
    duration_s: float
    force_body_n: np.ndarray


def replica_generator(seed, replica, name="none"):
    """One generator per replica: run seed, replica index and the declaration hash."""
    prefix = int(declaration(name)["sha256"][:8], 16)
    return np.random.default_rng([int(seed), int(replica), prefix])


def draw(name, generator):
    """Draw one replica's conditions in ``draw_order``; ``none`` returns the nominal draw."""
    if name == "none":
        return nominal_draw()
    t = TABLES[name]
    u = generator.uniform
    kp = float(u(t.joint_stiffness.low, t.joint_stiffness.high))
    kd = float(u(t.joint_damping.low, t.joint_damping.high))
    half = np.array([t.joint_position.half_width_rad[JOINT_GROUPS.index(j.split("_", 1)[1])] for j in JOINT_NAMES])
    offsets = u(-half, half)
    masses = u(t.link_mass.low, t.link_mass.high, size=19)
    payload = float(u(t.payload_mass.low, t.payload_mass.high))
    position = np.array([*u(t.payload_position.low, t.payload_position.high, size=2), .04])
    friction = float(u(t.foot_friction.low, t.foot_friction.high))
    strength = u(t.motor_strength.low, t.motor_strength.high, size=18)
    return ReplicaDraw(kp, kd, offsets, masses, payload, position, friction, strength)


def next_push(name, generator, first):
    """The next push of a replica, or None under ``none``."""
    if name == "none":
        return None
    p = TABLES[name].push
    delay = float(generator.uniform(*(p.first_delay_s if first else p.interval_s)))
    magnitude = float(generator.uniform(*p.magnitude_n))
    heading = float(generator.uniform(0., 2. * np.pi))
    return Push(delay, p.duration_s, np.array([magnitude * np.cos(heading), magnitude * np.sin(heading), 0.]))


def reset_pose(neutral, offset, lower, upper):
    """The drawn reset pose; the declaration keeps it inside the limits, and this checks that."""
    pose = np.asarray(neutral, dtype=float) + np.asarray(offset, dtype=float)
    if np.any(pose < np.asarray(lower)) or np.any(pose > np.asarray(upper)):
        raise ValueError("Drawn reset pose leaves the named joint limits")
    return pose


def scaled_link(mass, inertia, scale):
    """A link at ``scale`` times its mass: the tensor scales with it, the centre of mass stays."""
    return float(mass) * float(scale), np.asarray(inertia, dtype=float) * float(scale)


def _shift(d):
    d = np.asarray(d, dtype=float)
    return float(d @ d) * np.eye(3) - np.outer(d, d)


def combined_base_inertial(mass, com, inertia, payload_mass, payload_position):
    """The base link with a point payload: total mass, new centre of mass and the tensor about it.

    ``inertia`` is the base tensor about ``com``; both positions are in the base frame.
    """
    mass, payload_mass = float(mass), float(payload_mass)
    com, position, inertia = (np.asarray(x, dtype=float) for x in (com, payload_position, inertia))
    total = mass + payload_mass
    centre = (mass * com + payload_mass * position) / total
    tensor = inertia + mass * _shift(com - centre) + payload_mass * _shift(position - centre)
    return total, centre, tensor


def strength_ceiling(ceiling, strength):
    """The randomized effort ceiling: the factor scales the curve and the cap stays."""
    return (ceiling * strength).clip(max=TORQUE_CAP_NM)


def wrench_mean(forces_body_n, torques_body_nm):
    """The six privileged values: the mean body-frame force and torque over a control's substeps."""
    forces, torques = np.asarray(forces_body_n, dtype=float), np.asarray(torques_body_nm, dtype=float)
    if forces.shape != (SUBSTEPS, 3) or torques.shape != (SUBSTEPS, 3):
        raise ValueError("A control's wrench holds one force and one torque per physics step")
    return np.concatenate((forces.mean(0), torques.mean(0)))
