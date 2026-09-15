"""Two explicit single-robot placements; no physics, servo or scoring changes."""
from pathlib import Path
import json
import math
import struct

PLACEMENTS = {"origin": (0.0, 0.0), "xy14_4": (14.0, 4.0)}
SCHEMA = "canonical_single_placement_diagnostic_v1"
BASE_MATRIX = [[1., 0., 0., 0.], [0., 1., 0., 0.],
               [0., 0., 1., 0.], [0., 0., 0.08161109101311204, 1.]]
TRANSLATE_OP = "xformOp:translate:diagnosticPlacement"


def declaration(placement):
    if placement not in PLACEMENTS:
        raise ValueError("Explicit origin or xy14_4 placement required")
    x, y = PLACEMENTS[placement]
    geometry = json.loads((Path(__file__).parent / "geometry/geometry.json").read_text())
    return {"schema": SCHEMA, "placement": placement, "root": "/Robot",
            "xy_m": [x, y], "reset_xyz_m": [x, y, geometry["reset_plate_m"]],
            "scope": "Single-robot placement diagnosis only; no batch or training admission"}


def requested_root(placement, plate_z):
    expected = declaration(placement)
    if plate_z != expected["reset_xyz_m"][2]:
        raise ValueError("Reset height differs from unchanged geometry")
    # Preserve source005's actual float32 root write, including its rounded Z.
    return [[struct.unpack('<f', struct.pack('<f', value))[0]
             for value in expected["reset_xyz_m"] + [0., 0., 0., 1.]]]


def read_authored(stage, placement, UsdGeom, Usd):
    expected = declaration(placement)
    prim = stage.GetPrimAtPath(expected["root"])
    matrix = prim.GetAttribute("xformOp:transform").Get()
    record = {"base_matrix": [[float(matrix[i][j]) for j in range(4)] for i in range(4)],
              "xform_op_order": list(prim.GetAttribute("xformOpOrder").Get()),
              "translation_m": [float(v) for v in prim.GetAttribute(TRANSLATE_OP).Get()]}
    world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    record["computed_world_matrix"] = [[float(world[i][j]) for j in range(4)] for i in range(4)]
    validate_authored(record, expected)
    return record


def author_and_read(stage, roots, placement, UsdGeom, Gf, Usd):
    if roots != ["/Robot"]:
        raise ValueError("Placement diagnostic permits exactly /Robot")
    # Verify the inherited asset transform before authoring the only intervention.
    prim = stage.GetPrimAtPath("/Robot")
    base = prim.GetAttribute("xformOp:transform").Get()
    if ([[float(base[i][j]) for j in range(4)] for i in range(4)] != BASE_MATRIX
            or list(prim.GetAttribute("xformOpOrder").Get()) != ["xformOp:transform"]):
        raise ValueError("Unexpected inherited root transform")
    if placement not in PLACEMENTS:
        raise ValueError("Unknown placement")
    # Both arms use the same transform operation order, including a zero offset.
    x, y = PLACEMENTS[placement]
    UsdGeom.Xformable(prim).AddTranslateOp(
        opSuffix="diagnosticPlacement").Set(Gf.Vec3d(x, y, 0.))
    return read_authored(stage, placement, UsdGeom, Usd)


def validate_authored(record, expected):
    if expected != declaration(expected.get("placement")):
        raise ValueError("Placement declaration changed")
    if record.get("base_matrix") != BASE_MATRIX:
        raise ValueError("Inherited root transform changed")
    order = ["xformOp:transform", TRANSLATE_OP]
    vector = expected["xy_m"] + [0.]
    if record.get("xform_op_order") != order or record.get("translation_m") != vector:
        raise ValueError("Authored placement differs from declared intervention")
    world = [row[:] for row in BASE_MATRIX]
    world[3][:2] = expected["xy_m"]
    if record.get("computed_world_matrix") != world:
        raise ValueError("Computed world transform differs from declared placement")


def validate_reset(record, expected):
    want = requested_root(expected["placement"], expected["reset_xyz_m"][2])
    if record.get("placement") != expected or record.get("requested_root") != want:
        raise ValueError("Requested reset placement differs from source identity")
    if record.get("placement_readback_source") != "physics_articulation_view.get_root_transforms":
        raise ValueError("Reset placement requires actual physics-view readback")
    actual = record.get("post_reset", {}).get("root_pose_xyzw")
    if not isinstance(actual, list) or len(actual) != 1 or len(actual[0]) != 7:
        raise ValueError("Missing single native reset pose")
    for actual_value, wanted in zip(actual[0], want[0]):
        if (type(actual_value) not in (int, float) or not math.isfinite(actual_value)
                or abs(actual_value - wanted) > 2e-6):
            raise ValueError("Actual native reset placement differs")


def validate_evidence(directory, identity):
    directory = Path(directory)
    expected = identity.get("placement")
    if not isinstance(expected, dict) or expected != declaration(expected.get("placement")):
        raise ValueError("Missing exact placement in identity")
    authored = json.loads((directory / "placement_readback.json").read_text())
    if authored.get("declaration") != expected:
        raise ValueError("Authored placement identity mismatch")
    validate_authored(authored["before_warmup"], expected)
    validate_reset(json.loads((directory / "initial_reset.json").read_text()), expected)
