from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Protocol, Optional, List, Tuple, Any
import numpy as np
from models import RobotModel, Joint
from geom import normalize, common_normal, signed_angle_around, orthogonal_unit
        
@dataclass(frozen=True)
class ScrewAxis:
    """(w, q) screw-theoretic axis representation for a single joint."""
    w: np.ndarray  # shape (3,)
    q: np.ndarray  # shape (3,)

@dataclass(frozen=True)
class DHParam:
    """Standard DH parameters for a single link: theta, d, a, alpha."""
    theta: float
    d: float
    a: float
    alpha: float
    axis_offset: float

class KinematicsExtractor(Protocol):
    """Interface for extracting a kinematic representation from RobotModel."""
    kind: str
    def extract(self, robot: RobotModel) -> Dict[str, object]:
        ...

class ScrewKinematicsExtractor:
    """Extracts (w, q) from RobotModel.joints (behavior-preserving)."""
    kind = "screw"
    def extract(self, robot: RobotModel) -> Dict[str, ScrewAxis]:
        result: Dict[str, ScrewAxis] = {}
        for name, j in robot.joints.items():
            # j.axis_world / j.position are stored in RobotModel by PR-1
            w = np.asarray(j.axis_world, dtype=float)
            q = np.asarray(j.position, dtype=float)
            result[name] = ScrewAxis(w=w, q=q)
        return result

class SDHKinematicsExtractor:
    """
    Standard DH extractor.
    - uses (z_{i-1}, z_i) axis pairs per chain.
    - z0, q0 must be provided by the caller (recommended: z0 = normalize(neck - base), q0 = base).
    Usage:
      chains = {"left_arm": ["l_shoulder_roll", "l_shoulder_pitch", ...], ...}
      z0q0_by_chain = {"left_arm": (z0_vec, q0_point), ...}
      extract(robot) -> {chain_name: [(joint_name_i, DHParam_i), ...]}
    """
    kind = "dh_standard"
    def __init__(
        self,
        chains: Dict[str, List[str]],
        base_name: str = "base_link",
        base_origin: np.ndarray | None = None,
        base_z: np.ndarray | None = None,
        prefer_x: np.ndarray | None = None,
        eps: float = 1e-9,
    ):
        self.chains = chains
        self.base_name = base_name
        self.base_origin = np.zeros(3) if base_origin is None else np.asarray(base_origin, float)
        self.base_z = np.array([0.0, 0.0, 1.0]) if base_z is None else self._norm(np.asarray(base_z, float), eps)
        self.prefer_x = np.array([1.0, 0.0, 0.0]) if prefer_x is None else np.asarray(prefer_x, float)
        self.eps = eps

    @staticmethod
    def _norm(v: np.ndarray, eps: float) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v if n < eps else v / n
    
    def _call_first_available(self, obj: Any, names: List[str], *args, **kwargs):
        for nm in names:
            func = getattr(obj, nm, None)
            if callable(func):
                return func(*args, **kwargs)
        raise AttributeError(f"None of {names} found on {type(obj).__name__}")

    def _joint_axis(self, robot: Any, joint: str) -> np.ndarray:
        return robot.joints[joint].axis_world

    def _joint_point(self, robot: Any, joint: str) -> np.ndarray:
        return robot.joints[joint].position

    def _is_joint(self, robot: Any, name: str) -> bool:
        # use the explicit method if available
        if name in robot.joints.keys():
            return True
        else:
            return False
        
    def _any_unit_perp_to(self, z: np.ndarray, prefer: np.ndarray) -> np.ndarray:
        z = self._norm(z, self.eps)
        v = prefer - np.dot(prefer, z) * z
        if np.linalg.norm(v) < 0.05:
            alt = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
            v = alt - np.dot(alt, z) * z
        return self._norm(v, self.eps)

    def _closest_points_on_lines(self, p1, u, p2, v, eps=1e-12):
        p1 = np.asarray(p1, float); u = np.asarray(u, float)
        p2 = np.asarray(p2, float); v = np.asarray(v, float)

        u = u / (np.linalg.norm(u) + eps)
        v = v / (np.linalg.norm(v) + eps)
        w0 = p1 - p2
        b = float(np.dot(u, v))
        delta = 1.0 - b*b

        # general (skew/intersecting)
        if abs(delta) > 1e-10:
            d = float(np.dot(u, w0))
            e = float(np.dot(v, w0))
            t = (b*e - d) / delta
            s = (e - b*d) / delta
            P = p1 + t*u
            Q = p2 + s*v
            return P, Q, t, s

        # parallel/collinear
        # collinear (coincident): keep the reference point of the current axis
        if np.linalg.norm(np.cross(w0, u)) <= 1e-10:
            Q = p2.copy(); P = Q.copy()
            t = float(np.dot(Q - p1, u)); s = 0.0
            return P, Q, t, s

        # distinct parallel lines: closest point at the same axis parameter 'height' as p2
        t = float(np.dot(p2 - p1, u)); s = 0.0
        P = p1 + t*u; Q = p2.copy()
        return P, Q, t, s

    def _compute_sdh_step(
        self,
        p_prev: np.ndarray, z_prev: np.ndarray,
        p_cur:  np.ndarray,  z_cur:  np.ndarray,
        x_prev: Optional[np.ndarray] = None,
        o_prev: Optional[np.ndarray] = None,
        debug_label: Optional[str] = None,
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, float]:
        # --- normalize inputs ---
        eps = self.eps
        p_prev = np.asarray(p_prev, float); p_cur  = np.asarray(p_cur, float)
        z_prev = self._norm(np.asarray(z_prev, float), eps)
        z_cur  = self._norm(np.asarray(z_cur , float), eps)
        o_prev = np.zeros(3) if o_prev is None else np.asarray(o_prev, float)

        # x_prev: make it ⟂ z_prev (fallback if degenerate)
        if x_prev is None:
            x_prev = self._any_unit_perp_to(z_prev, prefer=self.prefer_x)
        else:
            x_prev = np.asarray(x_prev, float)
            x_prev_proj = x_prev - np.dot(x_prev, z_prev) * z_prev
            x_prev = ( self._any_unit_perp_to(z_prev, prefer=self.prefer_x)
                    if np.linalg.norm(x_prev_proj) < eps else
                    self._norm(x_prev_proj, eps) )

        # --- robust tolerances (scale-aware) ---
        L = max(1.0, np.linalg.norm(p_prev), np.linalg.norm(p_cur), np.linalg.norm(p_cur - p_prev))
        abs_tol  = 5e-6          # 5 μm (recommended if the unit is m; can be raised to 1e-5~1e-4 if needed)
        rel_tol  = 1e-5          # tolerance proportional to length
        line_tol = max(abs_tol, rel_tol * L)

        # --- incidence / parallelity tests ---
        v01 = p_cur - p_prev
        same_anchor = np.linalg.norm(v01) <= line_tol

        # decide by 'orthogonal distance to the line' instead of the old cross-product test (more robust)
        dist_prev_line = np.linalg.norm(v01 - np.dot(v01, z_prev) * z_prev)
        dist_cur_line  = np.linalg.norm(-v01 - np.dot(-v01, z_cur ) * z_cur)
        on_prev = dist_prev_line <= line_tol   # p_cur ∈ L_prev ?
        on_cur  = dist_cur_line  <= line_tol   # p_prev ∈ L_cur ?

        # also relax the angle tolerance (about 0.001 rad difference in the logs)
        cos_tol = float(np.cos(1e-2))          # ≈ 0.057°
        parallel = abs(float(np.dot(z_prev, z_cur))) >= cos_tol
        colinear = parallel and on_prev

        # --- choose P,Q ---
        if same_anchor or on_prev:
            P = p_cur.copy(); Q = p_cur.copy()
        elif on_cur:
            P = p_prev.copy(); Q = p_prev.copy()
        elif colinear:
            P = p_cur.copy(); Q = p_cur.copy()
        elif parallel:
            P = p_prev + np.dot(p_cur - p_prev, z_prev) * z_prev
            Q = p_cur.copy()
        else:
            # general (skew/intersecting)
            u, v = z_prev, z_cur
            w0 = p_prev - p_cur
            b = float(np.dot(u, v))
            d = float(np.dot(u, w0))
            e = float(np.dot(v, w0))
            dsc = 1.0 - b*b
            t = (b*e - d) / dsc
            s = (e - b*d) / dsc
            P = p_prev + t * u
            Q = p_cur  + s * v

        # --- SDH params ---
        n = Q - P
        dist = float(np.linalg.norm(n))
        a = 0.0 if dist < line_tol else dist

        # x_i
        if a > 0.0:
            x_i = n / a
        else:
            k = np.cross(z_prev, z_cur)
            x_i = ( self._norm(k, eps) if np.linalg.norm(k) > eps
                    else self._any_unit_perp_to(z_cur, prefer=x_prev) )

        # alpha, d, theta
        sin_alpha = float(np.dot(x_i, np.cross(z_prev, z_cur)))
        cos_alpha = float(np.dot(z_prev, z_cur))
        alpha = float(np.arctan2(sin_alpha, cos_alpha))
        d = float(np.dot(P - o_prev, z_prev))
        if abs(d) < 1e-12:
            d = 0.0

        sin_theta = float(np.dot(z_prev, np.cross(x_prev, x_i)))
        cos_theta = float(np.dot(x_prev, x_i))
        theta = float(np.arctan2(sin_theta, cos_theta))

        # axis_offset: always included (0 if missing)
        o_i = Q
        x_i = self._norm(x_i, eps)

        # === axis_offset: distance of p_cur from Q(=o_i) along the current axis z_cur ===
        axis_offset = float(np.dot(p_cur - o_i, z_cur))
        # denoise (optional): 0 if below the line tolerance
        if abs(axis_offset) <= line_tol:
            axis_offset = 0.0

        if debug_label:
            print(f"[{debug_label}] same_anchor={same_anchor}, on_prev={on_prev}, on_cur={on_cur}, parallel={parallel}, colinear={colinear}")
            print(f"[{debug_label}] P={P}, Q={Q}, ||Q-P||={a}, d={d}, alpha={alpha}, theta={theta}, axis_offset={axis_offset}")

        return theta, d, a, alpha, o_i, x_i, axis_offset

    # ... (omitted part is the same) ...

    def extract(self, robot: RobotModel) -> Dict[str, List[Tuple[str, DHParam]]]:
        """
        Returns a list of (joint_name, DHParam) for each chain.
        - compute 'spine' first (if present); the other chains then start from the frame of the last spine joint.
        - if the first element is the base (link name), use the world frame; if it is a joint name, find that joint's (o,x) frame or use a reasonable default.
        """
        results: Dict[str, List[Tuple[str, DHParam]]] = {}
        frames_after_joint: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        order = list(self.chains.keys())
        if "spine" in order:  # always spine first
            order = ["spine"] + [k for k in order if k != "spine"]

        for chain_name in order:
            names = self.chains[chain_name]
            if len(names) < 2:
                results[chain_name] = []
                continue

            # initial frame/axis setup
            prev_id = names[0]
            if self._is_joint(robot, prev_id):
                # previous joint axis/point
                p_prev = self._joint_point(robot, prev_id)
                z_prev = self._joint_axis(robot, prev_id)
                # origin/axis of the previous frame: use the one computed at the spine if available
                if prev_id in frames_after_joint:
                    o_prev, x_prev = frames_after_joint[prev_id]
                else:
                    o_prev = p_prev.copy()
                    x_prev = self._any_unit_perp_to(z_prev, prefer=self.prefer_x)
            else:
                # start from the base (world)
                p_prev = self.base_origin.copy()
                z_prev = self.base_z.copy()
                o_prev = self.base_origin.copy()
                x_prev = self._any_unit_perp_to(z_prev, prefer=self.prefer_x)

            chain_out: List[Tuple[str, DHParam]] = []

            # one SDH step for each joint of the chain
            for i in range(1, len(names)):
                jname = names[i]
                p_cur = self._joint_point(robot, jname)
                z_cur = self._joint_axis(robot, jname)
                theta, d, a, alpha, o_i, x_i, axis_offset = self._compute_sdh_step(
                    p_prev, z_prev, p_cur, z_cur, x_prev=x_prev, o_prev=o_prev, debug_label=None
                )
                # 1) "effective origin" (tip) of this joint = Q_i + tau_i z_i
                o_i_eff = o_i + axis_offset * z_cur

                # 2) store the result (4 DH params + tau)
                chain_out.append((jname, DHParam(theta=theta, d=d, a=a, alpha=alpha, axis_offset=axis_offset)))

                # 3) pass the 'effective origin' on as the input of the next step
                frames_after_joint[jname] = (o_i_eff, x_i)  # from now on, chains anchored at this joint start at the tip
                p_prev, z_prev = p_cur, z_cur               # (p_prev is originally p_i = Q_i + τ_i z_i, so it matches o_i_eff)
                o_prev, x_prev = o_i_eff, x_i               # key patch: update o_prev to the 'tip'

            results[chain_name] = chain_out

        return results