"""Exact full-robot contact mapping and native integer coverage checks."""
from __future__ import annotations
import math
from pathlib import PurePosixPath

CAPACITY = 2048
LAYOUT = "coincident_flat_origin_v1"


def expected_bindings(body_paths, environments):
    if len(body_paths) != 31 or len(environments) != 2 or len(set(environments)) != 2:
        raise ValueError("Need31 uniquely named bodies and two distinct environment namespaces")
    paths = {}
    for name, relative in body_paths.items():
        path = PurePosixPath(relative)
        if (not isinstance(name, str) or not isinstance(relative, str) or path.is_absolute()
                or ".." in path.parts or not relative.startswith("Geometry/body") or path.name != name):
            raise ValueError("Invalid named body path in exact kinematic mapping")
        paths[name] = relative
    if len(set(paths.values())) != 31:
        raise ValueError("Duplicate mapped body paths")
    return [{"source_environment": index, "source_name": name,
             "source_body": environment+"/Robot/"+relative,
             "target_environment": 1-index, "target_names": list(paths),
             "target_filters": [environments[1-index]+"/Robot/"+target for target in paths.values()]}
            for index, environment in enumerate(environments) for name, relative in paths.items()]


def native_integer(native):
    import torch
    try: limits = torch.iinfo(native.dtype)
    except TypeError as error: raise ValueError("Native contact indices/counts must be integer") from error
    if limits.max > torch.iinfo(torch.int64).max:
        raise ValueError("Native integer cannot be losslessly represented as int64")
    value = native.to(dtype=torch.int64)
    if bool(((value < 0) | (value > 2**32-1)).any()):
        raise ValueError("Native contact integer outside uint32 range")
    return value


def read_pair_sample(view, to_torch, dt):
    """One exact-source/31-filter view; validate live patch ranges, not max pair alone."""
    import torch
    if view.sensor_count != 1 or view.filter_count != 31 or view.max_contact_data_count != CAPACITY or not view.check():
        raise ValueError("Unsupported or invalid exact-source/31-target native contact view")
    matrix = to_torch(view.get_contact_force_matrix(dt=dt))
    data = view.get_contact_data(dt=dt)
    if len(data) != 6: raise ValueError("Unsupported detailed native contact tuple")
    normal_force, point, normal, separation = [to_torch(value) for value in data[:4]]
    native_count, native_start = to_torch(data[4]), to_torch(data[5])
    count, start = native_integer(native_count), native_integer(native_start)
    if (matrix.shape != (1,31,3) or count.shape != (1,31) or start.shape != (1,31)
            or normal_force.numel() != CAPACITY or point.shape != (CAPACITY,3)
            or normal.shape != (CAPACITY,3) or separation.numel() != CAPACITY):
        raise ValueError("Native full-body contact tensor layout differs")
    loaded = count > 0
    end = start + count
    # Capacity is shared across all31 targets of this view. Equality is ambiguous saturation.
    capacity_reached = bool((count.sum() >= CAPACITY) | ((end >= CAPACITY) & loaded).any())
    if bool(((end > CAPACITY) & loaded).any()): raise ValueError("Native patch range exceeds allocated capacity")
    offsets = torch.arange(CAPACITY, device=count.device)
    valid = ((offsets[:,None] >= start.flatten()) & (offsets[:,None] < end.flatten()) & loaded.flatten()).any(-1)
    membership = ((offsets[:,None] >= start.flatten()) & (offsets[:,None] < end.flatten()) & loaded.flatten()).sum(-1)
    if bool((membership > 1).any()): raise ValueError("Native target contact patch ranges overlap")
    finite = torch.isfinite(matrix).all()
    for value in (normal_force.reshape(CAPACITY)[valid], point[valid], normal[valid], separation.reshape(CAPACITY)[valid]):
        finite = finite & torch.isfinite(value).all()
    if not bool(finite): raise ValueError("Nonfinite force or loaded contact patch data")
    return matrix[0].clone(), count[0].clone(), start[0].clone(), capacity_reached, {
        "counts": str(native_count.dtype), "start_indices": str(native_start.dtype)}


def grade(case, metrics, steps):
    errors = []
    if metrics.get("samples") != steps or type(metrics.get("samples")) is not int or metrics.get("finite") is not True:
        errors.append("Incomplete or nonfinite physical samples")
    if metrics.get("contact_buffer_capacity_reached") is not False:
        errors.append("Contact data capacity reached or not verified")
    counts, forces = metrics.get("max_pair_contact_count"), metrics.get("max_pair_force_n")
    if type(counts) is not int or counts < 0 or type(forces) not in (int,float) or not math.isfinite(forces) or forces < 0:
        return errors+["Invalid full-body count/force metrics"]
    coverage = metrics.get("source_body_excitation")
    if (not isinstance(coverage,list) or len(coverage) != 62
            or any(not isinstance(row,dict) or type(row.get("contact_count_observed")) is not bool
                   or type(row.get("force_above_1mN_observed")) is not bool for row in coverage)):
        errors.append("Missing62-source negative-excitation coverage")
    if case == "filtered":
        if counts != 0 or forces > 1e-6: errors.append("Filtered full-body view observed foreign contact")
        if metrics.get("ground_support_observed_per_robot") != [True,True]: errors.append("Missing independent foot-ground support")
    elif case == "unfiltered_negative":
        if counts < 1 or forces <= 1e-3: errors.append("Negative control did not excite actual foreign contact")
        if metrics.get("foreign_contact_observed_per_direction") != [True,True]: errors.append("Negative contact was not observed in both directions")
    else: errors.append("Unknown full-body fixture case")
    return errors
