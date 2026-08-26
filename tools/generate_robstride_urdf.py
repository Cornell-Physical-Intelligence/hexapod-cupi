#!/usr/bin/env python3
"""Generate a physics-ready RobStride RS05 variant of the exported URDF.

The Onshape export used arbitrary low-density inertials. This script preserves
their relative mass distribution while normalizing the assembled body to
1.5 kg and each complete leg to 0.8 kg. One 191 g RS05 actuator is assigned to
the child/leg side of every revolute joint, so all three motors are included in
each leg target. It composes the resulting inertia tensors, adds collision
geometry matching the visuals, and applies the published RS05 output limits.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


BODY_TARGET_MASS_KG = 1.5
LEG_TARGET_MASS_KG = 0.8
LEG_COUNT = 6
MOTORS_PER_LEG = 3
MOTOR_MASS_KG = 0.191
MOTOR_RADIUS_M = 0.025  # Housing approximation; exact housing inertia is unpublished.
MOTOR_LENGTH_M = 0.040
PEAK_TORQUE_NM = 5.5
NO_LOAD_SPEED_RAD_S = 480.0 * 2.0 * math.pi / 60.0
LEG_LINK_PREFIXES = ("coxa", "femur", "tibia")


def is_leg_link(name: str) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}_") for prefix in LEG_LINK_PREFIXES)


def parse_vec(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.fromstring(text, sep=" ", dtype=np.float64)


def rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def parallel_axis(mass: float, offset: np.ndarray) -> np.ndarray:
    return mass * ((offset @ offset) * np.eye(3) - np.outer(offset, offset))


def motor_inertia(axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    i_axis = 0.5 * MOTOR_MASS_KG * MOTOR_RADIUS_M**2
    i_perp = MOTOR_MASS_KG * (3.0 * MOTOR_RADIUS_M**2 + MOTOR_LENGTH_M**2) / 12.0
    return i_perp * np.eye(3) + (i_axis - i_perp) * np.outer(axis, axis)


def fmt(value: float) -> str:
    return f"{value:.10g}"


def generate(source: Path, destination: Path) -> None:
    tree = ET.parse(source)
    robot = tree.getroot()
    robot.set("name", "hexapod_mkii_robstride")

    motors_by_link: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    revolute_count = 0
    for joint in robot.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        revolute_count += 1
        child = joint.find("child")
        axis_elem = joint.find("axis")
        if child is None:
            raise ValueError(f"Joint {joint.get('name')} has no child")
        axis_joint = parse_vec(axis_elem.get("xyz") if axis_elem is not None else None, (1, 0, 0))
        # At q=0 the child frame coincides with the joint frame. Assigning the
        # housing here makes the hip actuator part of the complete leg mass.
        motors_by_link.setdefault(child.get("link", ""), []).append(
            (np.zeros(3, dtype=np.float64), axis_joint)
        )

        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        limit.set("effort", fmt(PEAK_TORQUE_NM))
        limit.set("velocity", fmt(NO_LOAD_SPEED_RAD_S))

        dynamics = joint.find("dynamics")
        if dynamics is None:
            dynamics = ET.SubElement(joint, "dynamics")
        dynamics.set("damping", "0.01")
        dynamics.set("friction", "0.01")

    if revolute_count != 18:
        raise ValueError(f"Expected 18 revolute joints, found {revolute_count}")

    source_masses: dict[str, float] = {}
    for link in robot.findall("link"):
        inertial = link.find("inertial")
        mass_elem = inertial.find("mass") if inertial is not None else None
        if mass_elem is not None:
            source_masses[link.get("name", "")] = float(mass_elem.get("value", "0"))

    source_body_mass = sum(mass for name, mass in source_masses.items() if not is_leg_link(name))
    source_leg_mass = sum(mass for name, mass in source_masses.items() if is_leg_link(name))
    target_leg_structure_mass = LEG_COUNT * (
        LEG_TARGET_MASS_KG - MOTORS_PER_LEG * MOTOR_MASS_KG
    )
    if source_body_mass <= 0.0 or source_leg_mass <= 0.0 or target_leg_structure_mass <= 0.0:
        raise ValueError("Invalid source or target mass distribution")
    body_structure_scale = BODY_TARGET_MASS_KG / source_body_mass
    leg_structure_scale = target_leg_structure_mass / source_leg_mass

    total_mass = 0.0
    for link in robot.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_elem = inertial.find("mass")
        inertia_elem = inertial.find("inertia")
        origin = inertial.find("origin")
        if mass_elem is None or inertia_elem is None:
            raise ValueError(f"Incomplete inertial on {link.get('name')}")

        structure_scale = leg_structure_scale if is_leg_link(link.get("name", "")) else body_structure_scale
        structural_mass = float(mass_elem.get("value", "0")) * structure_scale
        structural_com = parse_vec(origin.get("xyz") if origin is not None else None, (0, 0, 0))
        structural_rpy = parse_vec(origin.get("rpy") if origin is not None else None, (0, 0, 0))
        inertia_local = np.array(
            [
                [float(inertia_elem.get("ixx", "0")), float(inertia_elem.get("ixy", "0")), float(inertia_elem.get("ixz", "0"))],
                [float(inertia_elem.get("ixy", "0")), float(inertia_elem.get("iyy", "0")), float(inertia_elem.get("iyz", "0"))],
                [float(inertia_elem.get("ixz", "0")), float(inertia_elem.get("iyz", "0")), float(inertia_elem.get("izz", "0"))],
            ],
            dtype=np.float64,
        ) * structure_scale
        rotation = rotation_from_rpy(structural_rpy)
        structural_inertia = rotation @ inertia_local @ rotation.T

        components: list[tuple[float, np.ndarray, np.ndarray]] = [
            (structural_mass, structural_com, structural_inertia)
        ]
        for motor_position, motor_axis in motors_by_link.get(link.get("name", ""), []):
            components.append((MOTOR_MASS_KG, motor_position, motor_inertia(motor_axis)))

        combined_mass = sum(component[0] for component in components)
        combined_com = sum(component[0] * component[1] for component in components) / combined_mass
        combined_inertia = np.zeros((3, 3), dtype=np.float64)
        for component_mass, component_com, component_inertia in components:
            combined_inertia += component_inertia + parallel_axis(component_mass, component_com - combined_com)
        combined_inertia = 0.5 * (combined_inertia + combined_inertia.T)
        if np.min(np.linalg.eigvalsh(combined_inertia)) <= 0.0:
            raise ValueError(f"Non-positive inertia generated for {link.get('name')}")

        mass_elem.set("value", fmt(combined_mass))
        if origin is None:
            origin = ET.SubElement(inertial, "origin")
        origin.set("xyz", " ".join(fmt(value) for value in combined_com))
        origin.set("rpy", "0 0 0")
        inertia_elem.attrib.clear()
        inertia_elem.set("ixx", fmt(combined_inertia[0, 0]))
        inertia_elem.set("ixy", fmt(combined_inertia[0, 1]))
        inertia_elem.set("ixz", fmt(combined_inertia[0, 2]))
        inertia_elem.set("iyy", fmt(combined_inertia[1, 1]))
        inertia_elem.set("iyz", fmt(combined_inertia[1, 2]))
        inertia_elem.set("izz", fmt(combined_inertia[2, 2]))
        total_mass += combined_mass

        if link.find("collision") is None:
            for visual in link.findall("visual"):
                collision = ET.Element("collision", {"name": "collision_from_visual"})
                visual_origin = visual.find("origin")
                visual_geometry = visual.find("geometry")
                if visual_origin is not None:
                    collision.append(copy.deepcopy(visual_origin))
                if visual_geometry is not None:
                    collision.append(copy.deepcopy(visual_geometry))
                link.append(collision)

    robot.insert(
        0,
        ET.Comment(
            " Physics variant generated from the Onshape export. The assembled body is normalized "
            "to 1.5 kg and each complete leg to 0.8 kg, including three 0.191 kg RobStride RS05 "
            "actuators assigned to child-side joint origins. Housing dimensions are an explicit "
            "inertia approximation; "
            "the Isaac Lab actuator separately applies the official 0.0007 kg*m^2 output armature. "
        ),
    )
    ET.indent(tree, space="    ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    print(f"Generated {destination}")
    print(f"Revolute joints: {revolute_count}")
    print(f"Total modeled rigid-body mass: {total_mass:.6f} kg")
    print(f"Body target: {BODY_TARGET_MASS_KG:.6f} kg")
    print(f"Per-leg target: {LEG_TARGET_MASS_KG:.6f} kg")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "robot/hexapod_mkii_mock_assy/urdf/hexapod_mkii_mock_assy.urdf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "robot/hexapod_mkii_mock_assy/urdf/hexapod_mkii_robstride.urdf",
    )
    args = parser.parse_args()
    generate(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
