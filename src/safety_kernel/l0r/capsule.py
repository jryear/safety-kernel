"""Capsule-capsule surface distance and FK helpers for self-collision."""

from __future__ import annotations

import numpy as np


def capsule_segment_endpoints(pos: np.ndarray, mat: np.ndarray,
                              half_length: float) -> tuple[np.ndarray, np.ndarray]:
    """Get world-frame segment endpoints of a capsule geom.

    MuJoCo capsule: centered at `pos`, aligned along the local Z-axis,
    with segment half-length = (total_length / 2). The `mat` is a 3x3
    rotation matrix (geom frame in world).

    Returns:
        (p0, p1) — the two endpoints of the capsule's central segment.
    """
    axis = mat[:, 2]  # local Z in world frame
    p0 = pos - axis * half_length
    p1 = pos + axis * half_length
    return p0, p1


def segment_segment_distance(p0: np.ndarray, p1: np.ndarray,
                             q0: np.ndarray, q1: np.ndarray) -> float:
    """Minimum distance between two line segments in 3D.

    Uses the standard closest-point parametric approach.
    Returns the Euclidean distance.
    """
    d1 = p1 - p0  # direction of segment 1
    d2 = q1 - q0  # direction of segment 2
    r = p0 - q0

    a = np.dot(d1, d1)  # |d1|^2
    e = np.dot(d2, d2)  # |d2|^2
    f = np.dot(d2, r)

    EPS = 1e-12

    if a <= EPS and e <= EPS:
        # Both segments degenerate to points
        return float(np.linalg.norm(r))

    if a <= EPS:
        # First segment degenerates to a point
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        if e <= EPS:
            # Second segment degenerates to a point
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b

            if denom > EPS:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0

            t = (b * s + f) / e

            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)

    closest_p = p0 + d1 * s
    closest_q = q0 + d2 * t
    return float(np.linalg.norm(closest_p - closest_q))


def capsule_capsule_surface_distance(
    pos_a: np.ndarray, mat_a: np.ndarray, size_a: np.ndarray,
    pos_b: np.ndarray, mat_b: np.ndarray, size_b: np.ndarray,
) -> float:
    """Surface distance between two MuJoCo capsule geoms.

    MuJoCo capsule `size` = [radius, half_length].

    Returns:
        Surface distance (positive = separated, 0 = contact, negative = penetrating).
    """
    radius_a = size_a[0]
    half_len_a = size_a[1]
    radius_b = size_b[0]
    half_len_b = size_b[1]

    p0, p1 = capsule_segment_endpoints(pos_a, mat_a, half_len_a)
    q0, q1 = capsule_segment_endpoints(pos_b, mat_b, half_len_b)

    center_dist = segment_segment_distance(p0, p1, q0, q1)
    surface_dist = center_dist - radius_a - radius_b
    return surface_dist


def compute_collision_distances(
    model, data, collision_pairs: list[tuple[int, int]]
) -> list[float]:
    """Compute surface distances for all collision pairs.

    Assumes mj_fwdPosition (or mj_step/mj_forward) has been called,
    so geom_xpos and geom_xmat are up to date.

    Args:
        model: mujoco.MjModel
        data: mujoco.MjData
        collision_pairs: list of (geom_id_a, geom_id_b)

    Returns:
        List of surface distances, one per pair.
    """
    distances = []
    for id_a, id_b in collision_pairs:
        pos_a = data.geom_xpos[id_a]
        mat_a = data.geom_xmat[id_a].reshape(3, 3)
        size_a = model.geom_size[id_a]

        pos_b = data.geom_xpos[id_b]
        mat_b = data.geom_xmat[id_b].reshape(3, 3)
        size_b = model.geom_size[id_b]

        dist = capsule_capsule_surface_distance(
            pos_a, mat_a, size_a, pos_b, mat_b, size_b
        )
        distances.append(dist)

    return distances


def compute_collision_distances_at_q(
    model, data, q_next: np.ndarray, collision_pairs: list[tuple[int, int]]
) -> list[float]:
    """Compute collision distances at a hypothetical joint configuration.

    Temporarily sets qpos to q_next, runs FK, computes distances,
    then restores the original qpos. Does NOT change qvel or time.
    """
    import mujoco

    q_save = data.qpos.copy()
    data.qpos[:] = q_next
    mujoco.mj_fwdPosition(model, data)
    distances = compute_collision_distances(model, data, collision_pairs)
    data.qpos[:] = q_save
    mujoco.mj_fwdPosition(model, data)
    return distances
