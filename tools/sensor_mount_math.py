"""CPU geometry for provisional sensor-mount screening (metres, radians).

Camera coordinates follow the RealSense depth optical frame: x right, y down,
z forward. Depth limits refer to z, not Euclidean distance. No simulator needed.
"""
from itertools import product
import math
import numpy as np


def rpy_matrix(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                             math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    out = np.eye(4)
    out[:3, :3] = rpy_matrix(*rpy)
    out[:3, 3] = xyz
    return out


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    out = np.eye(4)
    out[:3, :3] = np.eye(3) + math.sin(angle)*skew + (1-math.cos(angle))*(skew@skew)
    return out


def apply_transform(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def box_corners(bounds):
    return np.array(list(product(*np.asarray(bounds).T)))


def segments_hit_boxes(origin, targets, boxes, endpoint_margin_m=0.001):
    """Return ray-segment hits, excluding the last 1 mm at the terrain target.

    Boxes are (B,2,3), points (N,3). Parallel rays and origins inside a box are
    handled explicitly. Starting inside any robot bound counts as obstructed.
    """
    origin, targets, boxes = map(np.asarray, (origin, targets, boxes))
    if len(boxes) == 0 or len(targets) == 0:
        return np.zeros(len(targets), dtype=bool)
    direction = targets - origin
    parallel = np.abs(direction) < 1e-12
    safe_direction = np.where(parallel, 1., direction)
    t1 = (boxes[None, :, 0] - origin) / safe_direction[:, None]
    t2 = (boxes[None, :, 1] - origin) / safe_direction[:, None]
    low, high = np.minimum(t1, t2), np.maximum(t1, t2)
    low = np.where(parallel[:, None], -np.inf, low)
    high = np.where(parallel[:, None], np.inf, high)
    outside = ((origin < boxes[:, 0]) | (origin > boxes[:, 1]))
    impossible = (parallel[:, None] & outside[None]).any(axis=2)
    enter = np.maximum(low.max(axis=2), 0.)
    leave = np.minimum(high.min(axis=2),
                       (1-endpoint_margin_m/np.maximum(np.linalg.norm(direction, axis=1), 1e-12))[:, None])
    return ((enter <= leave) & ~impossible).any(axis=1)


def nominal_profile(model, width, height, min_z):
    """Nominal rectified surrogate; replace with selected-device intrinsics.

    Datasheet depth FoV includes stereo overlap at a stated reference distance.
    Invert its section 4.4 expression to estimate the individual imager angle,
    then explicitly project into BOTH imagers at the target depth. This keeps
    the left invalid stereo band instead of treating nominal FoV as constant.
    """
    baseline, depth_hfov, reference_z, mass = {
        'D405': (.018, 84., .2, .058),
        'D435': (.050, 87., 2., .075),
        'D455': (.095, 87., 2., .116),
    }[model]
    lo, hi = 0., math.pi/2-1e-6
    for _ in range(70):
        a = (lo+hi)/2
        if a + math.atan(math.tan(a)-baseline/reference_z) < math.radians(depth_hfov):
            lo = a
        else:
            hi = a
    fx = width/(2*math.tan((lo+hi)/2))
    return dict(id=f'{model}_{width}x{height}', model=model, width=width, height=height,
                fx=fx, fy=height/(2*math.tan(math.radians(58/2))),
                ppx=width/2, ppy=height/2, baseline_m=baseline, min_z_m=min_z,
                max_z_m=.5 if model == 'D405' else 2., mass_kg=mass,
                nominal_depth_hfov_deg=depth_hfov, nominal_depth_vfov_deg=58.,
                fov_reference_z_m=reference_z,
                calibrated=False, source='D400 datasheet rev020 tables3-49,4-11; section4.4; inferred pinhole intrinsics')


def optical_visibility(optical_points, profile, roi_fraction=1.):
    p = np.asarray(optical_points)
    z = p[:, 2]
    safe_z = np.maximum(z, 1e-12)
    u = profile['fx'] * p[:, 0]/safe_z + profile['ppx']
    # Right optical centre is +baseline along left camera +X.
    ur = profile['fx'] * (p[:, 0]-profile['baseline_m'])/safe_z + profile['ppx']
    v = profile['fy'] * p[:, 1]/safe_z + profile['ppy']
    xmargin = profile['width']*(1-roi_fraction)/2
    ymargin = profile['height']*(1-roi_fraction)/2
    return ((z >= profile['min_z_m']) & (z <= profile['max_z_m']) &
            (u >= xmargin) & (ur >= xmargin) &
            (u < profile['width']-xmargin) & (ur < profile['width']-xmargin) &
            (v >= ymargin) & (v < profile['height']-ymargin))


def look_outward(position, pitch_deg):
    horizontal = np.array([position[0], position[1], 0.], dtype=float)
    horizontal /= np.linalg.norm(horizontal)
    z = math.cos(math.radians(pitch_deg))*horizontal + [0, 0, -math.sin(math.radians(pitch_deg))]
    x = np.cross(z, [0, 0, 1.])
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def replay_visibility_contract(visible, sector_count=6):
    """Synthetic observation-mask fixture through the existing terrain contract.

    The matrix is sector × sample index, NOT a spatial elevation raster. Zero
    heights are an explicit flat-fixture input, never inferred from absent depth.
    This verifies missing visibility and age handling only, not support strength.
    """
    from terrain_readiness import terrain_channels, support_region_observed
    visible=np.asarray(visible,dtype=bool).reshape(sector_count,-1)
    height=np.zeros(visible.shape);variance=np.full(visible.shape,.005**2)
    captured=np.where(visible,10.,np.nan)
    results={}
    for name,now,observed,times in [
            ('fresh',10.10,visible,captured),
            ('all_streams_stale',10.30,visible,captured),
            ('no_observation',10.10,np.zeros_like(visible),captured),
            ('future_timestamp',9.90,visible,captured)]:
        channels=terrain_channels(height,variance,observed,times,now_s=now)
        sector_ok=[]
        for sector in range(sector_count):
            required=np.zeros_like(visible);required[sector]=True
            sector_ok.append(support_region_observed(channels,required))
        results[name]=dict(usable_count=int(channels[...,1].sum()),
            required_sector_all_samples_observed=sector_ok,
            unseen_marked_usable=bool(np.any(channels[...,1][~visible])),
            padding_height_is_support_evidence=False)
    return results
