"""Model-specific kinematic tripod references for an explicitly adapted AMP prior.

This bounded inverse-kinematics fit is not the paper authors' whole-body dynamic
trajectory optimization. It changes no robot geometry and supplies no poses to
the simulator during policy rollouts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
NAMES = [f"{leg}_{joint}" for leg in LEGS for joint in ("coxa_yaw", "femur_pitch", "tibia_pitch")]
NOMINAL = np.tile([0., -.30, .40], 6)
URDF_SHA = "9492fde54c50170e940b49ccb3037d7b413743a5b77bc1d8707a539d44349e78"
MODEL_SHA = "7126935b35398d2a7f03271335a8a1b8f2059354af2b19d9474f22b8e73e4881"


class Kinematics:
    def __init__(self, model: dict, geometry: dict):
        joints = {x["name"]: x for x in model["joints"]}
        self.joints = [joints[name] for name in NAMES]
        self.lower = np.array([j["lower"] for j in self.joints])
        self.upper = np.array([j["upper"] for j in self.joints])
        self.origins = []
        for j in self.joints:
            t = np.eye(4)
            t[:3, :3] = Rotation.from_quat(j["quaternion_xyzw"]).as_matrix()
            t[:3, 3] = j["xyz"]
            self.origins.append(t)
        shapes = {x["body"]: x for x in geometry["shapes"]}
        self.toe_local = []
        for leg in LEGS:
            s = shapes[leg + "_tibia"]
            center = np.mean(s["cap_bounds_m"], axis=0)
            self.toe_local.append((np.array(s["shape_to_link"]) @ np.r_[center, 1])[:3])
        self.toe_local = np.array(self.toe_local)

    def transform(self, leg: int, q: np.ndarray) -> np.ndarray:
        t = np.eye(4)
        for k in range(3):
            i = leg * 3 + k
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec(np.array(self.joints[i]["axis"]) * q[k]).as_matrix()
            t = t @ self.origins[i] @ turn
        return t

    def leg(self, leg: int, q: np.ndarray) -> np.ndarray:
        return (self.transform(leg,q) @ np.r_[self.toe_local[leg], 1])[:3]

    def feet(self, q: np.ndarray) -> np.ndarray:
        return np.array([self.leg(i, q[3*i:3*i+3]) for i in range(6)])

    def solve(self, leg: int, target: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, float]:
        low, high = self.lower[leg*3:leg*3+3], self.upper[leg*3:leg*3+3]
        # Position fitting dominates the tiny continuity regularizer.
        result = least_squares(lambda q: np.r_[(self.leg(leg, q)-target)*1000, (q-seed)*.002],
                               np.clip(seed, low+1e-7, high-1e-7), bounds=(low+1e-7, high-1e-7),
                               ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=80)
        return result.x, float(np.linalg.norm(self.leg(leg, result.x)-target))


def reference_cycle(kin: Kinematics, command: np.ndarray, *, period=1.2, lift=.018, dt=.02,
                    root_height=.098) -> dict:
    count = round(period/dt)
    if not np.isclose(count*dt, period):
        raise ValueError("Gait period must have an integer number of policy controls")
    nominal = kin.feet(NOMINAL)
    velocity = np.array([command[1], -command[0], 0.])  # navigation -> native body frame
    yaw = np.array([0., 0., command[2]])
    q = np.zeros((count, 18))
    feet = np.zeros((count, 6, 3))
    errors = np.zeros((count, 6))
    moving = np.linalg.norm(command) > 1e-10
    # Two alternating tripods: LF/LR/RM and LM/RF/RR.
    shifts = np.array([0., .5, 0., .5, 0., .5])
    for t in range(count):
        for leg in range(6):
            phase = (t/count + shifts[leg]) % 1.
            foot_velocity = velocity + np.cross(yaw, nominal[leg])
            if phase < .5:
                # Fixed world contact during stance; body advances over the foot.
                displacement = (.25-phase)*period*foot_velocity
                height = 0.
            else:
                u = (phase-.5)*2
                # Smooth swing joins stance with its same endpoint velocity.
                blend = -u + 6*u*u - 4*u*u*u
                displacement = (-.25+.5*blend)*period*foot_velocity
                height = lift * np.sin(np.pi*u)**2 if moving else 0.
            target = nominal[leg] + displacement
            target[2] += height
            seed = q[t-1, leg*3:leg*3+3] if t else NOMINAL[leg*3:leg*3+3]
            fitted, err = kin.solve(leg, target, seed)
            q[t, leg*3:leg*3+3], errors[t, leg] = fitted, err
            feet[t, leg] = kin.leg(leg, fitted)
    dq = (np.roll(q, -1, axis=0)-np.roll(q, 1, axis=0))/(2*dt)
    states = np.concatenate([q, dq, np.tile(velocity, (count, 1)), np.tile(yaw, (count, 1)),
                             np.full((count, 1), root_height), feet.reshape(count,18)], axis=1)
    return {"states": states.astype(np.float32), "next_states": np.roll(states,-1,axis=0).astype(np.float32),
            "commands": np.tile(command,(count,1)).astype(np.float32), "q": q.astype(np.float32),
            "max_position_error_m": float(errors.max()),
            "max_control_delta_rad": float(np.max(np.abs(np.roll(q,-1,axis=0)-q))),
            "max_abs_q_rad": float(np.abs(q).max())}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a=p.parse_args()
    if a.output.exists():
        raise ValueError("Reference output must be fresh")
    if hashlib.sha256(a.model.read_bytes()).hexdigest()!=MODEL_SHA:
        raise ValueError("Wrong canonical model bytes")
    model=json.loads(a.model.read_text()); geometry=json.loads(a.geometry.read_text())
    kin=Kinematics(model,geometry)
    commands=[(0.,0.,0.)]
    for speed in (.025,.05):
        for angle in np.arange(8)*np.pi/4:
            commands.append((float(speed*np.cos(angle)),float(speed*np.sin(angle)),0.))
    commands += [(0.,0.,-.20),(0.,0.,.20),(.04,0.,-.15),(.04,0.,.15)]
    # Exact original shape extrema determine the proposed stance's ground height.
    with np.load(a.geometry.parent/"geometry_extrema.npz") as z:
        minima=[]
        for i,leg in enumerate(LEGS):
            t=kin.transform(i,NOMINAL[i*3:i*3+3]);v=z['all__'+leg+'_tibia']
            minima.append(float((v@t[:3,:3].T+t[:3,3])[:,2].min()))
    root_height=-min(minima)
    cycles=[reference_cycle(kin,np.array(c),root_height=root_height) for c in commands]
    # Infeasible kinematics is a generator error, never labelled an expert.
    if max(c["max_position_error_m"] for c in cycles)>.001:
        raise ValueError("IK reference misses requested endpoint by >1mm")
    if max(c["max_control_delta_rad"] for c in cycles)>.04:
        raise ValueError("Reference exceeds the unchanged 0.04rad/20ms target slew")
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/"tripod_prior.npz", **{k:np.concatenate([c[k] for c in cycles])
        for k in ("states","next_states","commands","q")})
    meta={"schema":"canonical_paper_amp_kinematic_prior_v1", "model_sha256":MODEL_SHA,"urdf_sha256":URDF_SHA,
          "mass_kg":model["total_mass_kg"],"joint_names":NAMES,"toe_names":[x+"_tibia" for x in LEGS],
          "toe_local_points_m":kin.toe_local.tolist(),"nominal_joint_position_rad":NOMINAL.tolist(),
          "neutral_feet_body_m":kin.feet(NOMINAL).tolist(),"root_height_m":root_height,
          "reset_root_height_m":root_height+.005,"dt_s":.02,"period_s":1.2,"lift_m":.018,
          "stance_reason":"The zero inspection pose reaches the tibia lower limit before lifting the foot vertically; this declared walking stance reserves joint travel. It requires its own native validation and does not revise the failed zero-pose standing results.",
          "amp_order":"q18,dq18,native_body_linear3,native_body_angular3,root_height1,native_body_toe_xyz18",
          "command_frame":"forward=-nativeY,left=nativeX,yaw=+Z", "commands":commands,
          "cycles":[{k:v for k,v in c.items() if not isinstance(v,np.ndarray)} for c in cycles],
          "trajectory_optimization_scope":"Bounded exact-model IK with continuity regularization; no whole-body dynamics or contact-force optimization.",
          "physical_admission":False,"pose_forcing_during_ppo":False,
          "prior_sha256":hashlib.sha256((a.output/"tripod_prior.npz").read_bytes()).hexdigest()}
    (a.output/"prior_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({"transitions":sum(len(c["states"]) for c in cycles),
        "max_position_error_m":max(c["max_position_error_m"] for c in cycles),
        "max_control_delta_rad":max(c["max_control_delta_rad"] for c in cycles),
        "max_abs_q_rad":max(c["max_abs_q_rad"] for c in cycles),"output":str(a.output)},indent=2))


if __name__=="__main__":
    main()
