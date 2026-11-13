from __future__ import annotations
import numpy as np

def _as_np3(v):
    a = np.asarray(v, dtype=float).reshape(3)
    return a

def normalize(v, eps: float = 1e-12):
    v = _as_np3(v)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros(3, dtype=float)
    return v / n

def orthogonal_unit(z, eps: float = 1e-12):
    """Stably pick a unit vector perpendicular to z."""
    z = normalize(z, eps)
    # pick a reference axis that is not nearly parallel to z
    bases = [np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,1.])]
    ref = max(bases, key=lambda e: 1. - abs(float(np.dot(z, e))))
    u = ref - np.dot(ref, z) * z
    return normalize(u, eps)

def signed_angle_around(axis, u, v, eps: float = 1e-12) -> float:
    """Signed angle (rad) from u to v about axis. Only the components of u, v perpendicular to axis are used."""
    axis = normalize(axis, eps)
    u = _as_np3(u); v = _as_np3(v)
    u_p = u - np.dot(u, axis) * axis
    v_p = v - np.dot(v, axis) * axis
    nu = np.linalg.norm(u_p); nv = np.linalg.norm(v_p)
    if nu < eps or nv < eps:
        return 0.0
    u_p /= nu; v_p /= nv
    x = float(np.clip(np.dot(u_p, v_p), -1.0, 1.0))
    y = float(np.dot(axis, np.cross(u_p, v_p)))
    return float(np.arctan2(y, x))

def common_normal(q0, w0, q1, w1, eps: float = 1e-9):
    """
    Compute the common normal of the two axes L0:(q0 + t w0), L1:(q1 + s w1).
    Returns: r0 (foot on axis 0), r1 (foot on axis 1), x (unit vector along the common normal), a (distance), parallel (bool)
    """
    q0 = _as_np3(q0); q1 = _as_np3(q1)
    w0 = normalize(w0, eps); w1 = normalize(w1, eps)
    r = q1 - q0
    b = float(np.dot(w0, w1))
    d = float(np.dot(w0, r))
    e = float(np.dot(w1, r))
    denom = 1.0 - b*b
    if abs(denom) > eps:
        # general (skew/intersecting) case
        t = (d - b*e) / denom
        s = (b*d - e) / denom
        r0 = q0 + t * w0
        r1 = q1 + s * w1
        n = r1 - r0
        a = float(np.linalg.norm(n))
        if a > eps:
            x = n / a
        else:
            # intersecting, but the perpendicular length is 0
            c = np.cross(w0, w1)
            if np.linalg.norm(c) > eps:
                x = c / np.linalg.norm(c)
            else:
                x = orthogonal_unit(w0)
        if np.dot(x, q1 - q0) < 0:
            x = -x
        return r0, r1, x, a, False
    else:
        # parallel/antiparallel case
        n = r - np.dot(r, w0) * w0
        a = float(np.linalg.norm(n))
        x = n / a if a > eps else orthogonal_unit(w0)
        if np.dot(x, q1 - q0) < 0:
            x = -x
        r0 = q0
        r1 = q1 - np.dot(q1 - r0, w1) * w1  # (r1 - r0) ⟂ w1
        return r0, r1, x, a, True
