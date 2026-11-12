"""Supply MuJoCo evidence to the common exact capture checks."""

import numpy as np

from locomotion.evaluation.capture import classify_contacts, diagnostic_servo
from locomotion.robots.hexapod import KD
from locomotion.robots.rs05 import RATED_TORQUE_NM, STIFFNESS_NM_PER_RAD


class SurrogateCaptureSource:
    """Snapshot this reset's motor law and decode floor-contact sensor summaries."""

    def __init__(self, env):
        self.env = env
        self.kp = (
            env._kp.numpy().copy()
            if env.randomized
            else np.full((env.num_envs, 1), STIFFNESS_NM_PER_RAD, np.float32)
        )
        self.kd = (
            env._kd.numpy().copy()
            if env.randomized
            else np.broadcast_to(np.asarray(KD, np.float32), (env.num_envs, 18)).copy()
        )
        self.strength = (
            env._strength.numpy().copy()
            if env.randomized
            else np.ones((env.num_envs, 18), np.float32)
        )
        self.metadata = {
            "backend": env.native_readback["surrogate"]["engine"],
            "contacts": "MuJoCo per-geom floor normal force and contact-point sensor summaries; "
            "toe and shaft are separate geoms; no native patch completeness claim",
            "contact_classification": "per_geom_netforce_positions",
            "motor_law": "canonical_rs05_with_reset_gains_and_strength",
            "motor_snapshot": {
                "kp_nm_per_rad": self.kp.tolist(),
                "kd_nm_s_per_rad": self.kd.tolist(),
                "strength": self.strength.tolist(),
            },
            "randomization": env.native_readback["randomization"],
        }

    def counter(self):
        return self.env.physics_steps

    def contacts(self, poses, geometry):
        env = self.env
        data = env._sensor[:, 0].reshape(env.num_envs, 25, 6)
        if not np.isfinite(data).all():
            raise ValueError("Nonfinite MuJoCo contact sensor")
        origins = env._origins64.numpy()
        contacts = []
        # Preserve the body/geom order used by existing surrogate captures.
        geoms = [[i] for i in range(13)] + [[13 + k, 19 + k] for k in range(6)]
        for e in range(env.num_envs):
            for body, indices in zip(env.native_body_names, geoms):
                for g in indices:
                    if data[e, g, 2] > 0.0:
                        contacts.append(
                            (
                                len(contacts),
                                e,
                                body,
                                float(data[e, g, 2]),
                                data[e, g, 3:] + origins[e],
                                np.array([0.0, 0.0, 1.0]),
                                0.0,
                            )
                        )
        return classify_contacts(contacts, poses, geometry, env.num_envs)

    def servo(self, q, dq, target):
        """Independent NumPy oracle: scale the float32 curve, then keep the rated cap."""
        requested, _, nominal_ceiling = diagnostic_servo(q, dq, target, self.kp, self.kd)
        ceiling = np.minimum(nominal_ceiling * self.strength, np.float32(RATED_TORQUE_NM))
        return requested, np.clip(requested, -ceiling, ceiling).astype(np.float32), ceiling
