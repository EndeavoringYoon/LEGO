from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
# define chains (arm-centric; leg/torso chains can be added if needed)
from collections import OrderedDict, defaultdict
# Default builder for (w,q) only. DH will be added as an adapter in a later PR.
from representations import SDHKinematicsExtractor

@dataclass
class RobotKinematicData:
    robot_name: str
    group_arrays: Dict[str, np.ndarray]   # part_name -> (N,6) array

@dataclass
class KinematicsDataset:
    arrays: np.ndarray                    # (R, N_total, 6)
    labels: List[str]                     # (R,)
    max_joints_per_group: List[int]       # len == #groups
    group_order: List[str]                # e.g. ["torso","neck",...]
    selected_indices: np.ndarray          # (num_poe, len(group_order))
    feature_dim: int = 6

class KinematicsDatasetBuilder:
    """
    Builder that modularizes the original create_data_to_classify() logic.
    - default representation: screw (w,q) = 6D
    - applies the pos/axis fill, normalize and collapse rules as is
    """
    def __init__(self,
                 classify_joints: str = "all",
                 repr_kind: str = "screw",         # "screw" | "dh"
                 axis_fill: str = "zero",          # "zero" | "local_average"
                 pos_fill: str = "local_average",  # "zero" | "local_average"
                 normalize: str = "each",          # "none" | "all" | "each"
                 use_shoulder_pos: bool = True,
                 # --- DH-only options ---
                 dh_length_fill: str = "zero",  # a,d
                 dh_angle_fill: str = "zero",            # theta,alpha
                 dh_variant: str = "standard",           # "standard" (default)
                 dh_wrist_groups: Tuple[str,...] = ("left_wrist","right_wrist"),  # wrist group names
                 ):
        self.classify_joints = classify_joints
        self.repr_kind = repr_kind
        self.axis_fill = axis_fill
        self.pos_fill = pos_fill
        self.normalize = normalize
        self.use_shoulder_pos = use_shoulder_pos
        self.dh_length_fill = dh_length_fill
        self.dh_angle_fill = dh_angle_fill
        self.dh_variant = dh_variant
        self.dh_wrist_groups = dh_wrist_groups
    def _tcp_group_for(self,part_name: str) -> str:
        # take a wrist chain name (e.g. 'left_wrist', 'right_wrist', 'wrist') and return the TCP group name
        if part_name.startswith("left_"):
            return "left_tcp"
        if part_name.startswith("right_"):
            return "right_tcp"
        return "tcp"
    def _compute_max_joints(self, rc, classification_groups):
        max_j = [0] * len(classification_groups)
        for robot_name in rc.classifying_joint_info.keys():
            for i, group in enumerate(classification_groups):
                cnt = 0
                for part in group:
                    cnt += len(rc.classify_data.get(robot_name, {}).get(part, []))
                if part.endswith("_tcp"):
                    cnt+=1
                max_j[i] = max(max_j[i], cnt)
        return max_j

    def _compute_poe_indices(self, group_order):
        # 7) poe indices (original logic kept)
        poe_left_elbow  = ['torso','neck','left_shoulder_girdle','left_shoulder','left_upperarm','left_elbow']
        poe_right_elbow = ['torso','neck','right_shoulder_girdle','right_shoulder','right_upperarm','right_elbow']
        poe_left_wrist  = ['torso','neck','left_shoulder_girdle','left_shoulder','left_upperarm','left_elbow','left_forearm','left_wrist']
        poe_right_wrist = ['torso','neck','right_shoulder_girdle','right_shoulder','right_upperarm','right_elbow','right_forearm','right_wrist']
        poe_list = [poe_left_elbow, poe_right_elbow, poe_left_wrist, poe_right_wrist]
        return np.array([[1 if g in poe else 0 for g in group_order] for poe in poe_list], dtype=int)

    def build(self,
              rc,                          # RobotCharacteristics instance (uses its existing methods)
              verbose: bool = False) -> KinematicsDataset:
        """
        uses rc.classifying_joint_info / rc.classify_data / rc.classifying_* / rc.get_avg_neck_and_base.
        Return values and side effects are the same as the original create_data_to_classify():
          - fills rc.classifying_joint_info[robot]['data_to_classify']
          - updates rc.max_joints_per_group, rc.classify_data_shape
        """
        # 1) select group
        if self.classify_joints == "all":
            if self.repr_kind=="screw":
                classification_groups = rc.classifying_all_joints
            elif self.repr_kind=="dh":
                classification_groups = rc.classifying_all_joints_dh
        elif self.classify_joints == "except_torso_and_neck":
            classification_groups = rc.classifying_joints_except_torso_and_neck
        else:
            raise ValueError(f"Invalid classify_joints: {self.classify_joints}")
        group_order = [g[0] for g in classification_groups]

        # 2) max number of joints per group
        max_joints_per_group = self._compute_max_joints(rc,classification_groups)
        selected_indices = self._compute_poe_indices(group_order)
        if verbose:
            print("max_joints_per_group:", max_joints_per_group)
        if self.repr_kind == "screw":
            return self._build_screw(rc, classification_groups, group_order, max_joints_per_group, selected_indices, verbose)
        elif self.repr_kind == "dh":
            if self.dh_variant != "standard":
                raise NotImplementedError("Only standard DH is wired in PR-5.")
            return self._build_dh_sdh(rc, classification_groups, group_order, max_joints_per_group, selected_indices, verbose)
        else:
            raise ValueError(f"Invalid repr_kind: {self.repr_kind}")


    def _build_screw(self, rc, classification_groups, group_order, max_joints_per_group, selected_indices, verbose):
        # 3) build per-robot (part -> (N,6)) data
        for robot_name, info in rc.classifying_joint_info.items():
            joint_info = info  # original dict: joint_name -> {'axis_world','position',...}
            robot_data: Dict[str, np.ndarray] = {}
            for g_idx, group in enumerate(classification_groups):
                part = group[0]
                arr = np.zeros((max_joints_per_group[g_idx], 6), dtype=float)
                if part not in rc.classify_data.get(robot_name, {}):
                    raise ValueError(f"robot {robot_name}'s {part} missing in classify_data")
                jnames = rc.classify_data[robot_name][part]
                cur = 0
                for jn in jnames:
                    if jn in joint_info:
                        w = np.asarray(joint_info[jn]['axis_world'], dtype=float)
                        q = np.asarray(joint_info[jn]['position'], dtype=float)
                        arr[cur, 0:3] = w
                        arr[cur, 3:6] = q
                    cur += 1
                robot_data[part] = arr
            rc.classifying_joint_info[robot_name]['data_to_classify'] = robot_data

        # 4) fill: axis/pos
        for robot_name, info in rc.classifying_joint_info.items():
            data = info["data_to_classify"]
            for g_idx, group in enumerate(classification_groups):
                part = group[0]
                arr = data[part]
                # axis
                zero_axis = np.all(arr[:, 0:3] == 0, axis=-1)
                valid_axis = arr[~zero_axis, 0:3]
                if self.axis_fill == "local_average" and len(valid_axis) > 0:
                    arr[zero_axis, 0:3] = np.mean(valid_axis, axis=0)
                elif self.axis_fill == "zero":
                    pass
                else:
                    if self.axis_fill not in ("zero", "local_average"):
                        raise ValueError(f"Invalid axis_fill: {self.axis_fill}")
                # pos
                zero_pos = np.all(arr[:, 3:6] == 0, axis=-1)
                fill_mask = zero_axis & zero_pos
                valid_pos = arr[~fill_mask, 3:6]
                pos_mean = None
                if self.pos_fill == "local_average":
                    if len(valid_pos) > 0:
                        pos_mean = np.mean(valid_pos, axis=0)
                    # collapse rule when a whole group is empty
                    if pos_mean is None and fill_mask.sum() == max_joints_per_group[g_idx]:
                        if part.startswith(("left_", "right_")):
                            side = part.split('_')[0]
                            if any(k in part for k in ("girdle", "upperarm")):
                                ref_part = f"{side}_shoulder"
                            elif any(k in part for k in ("forearm", "wrist")):
                                ref_part = f"{side}_elbow"
                            else:
                                raise ValueError(f"Unhandled part for collapse: {robot_name}:{part}")
                            ref = data[ref_part]
                            c0 = np.all(ref[:, 0:3] == 0, axis=-1)
                            c1 = np.all(ref[:, 3:6] == 0, axis=-1)
                            cvalid = (~c0) & (~c1)
                            if np.any(cvalid):
                                pos_mean = np.mean(ref[cvalid, 3:6], axis=0)
                        elif "torso" in part or "neck" in part:
                            avg_base, avg_neck = rc.get_avg_neck_and_base(robot_name, True, False)
                            pos_mean = avg_base if "torso" in part else avg_neck
                elif self.pos_fill == "zero":
                    pass
                else:
                    raise ValueError(f"Invalid pos_fill: {self.pos_fill}")
                if pos_mean is not None:
                    arr[fill_mask, 3:6] = pos_mean
                data[part] = arr
            rc.classifying_joint_info[robot_name]["data_to_classify"] = data

        # 5) normalize: by the neck-base distance
        diffs = {}
        bases = {}
        for robot_name, info in rc.classifying_joint_info.items():
            avg_base, avg_neck = rc.get_avg_neck_and_base(robot_name, self.use_shoulder_pos, False)
            diffs[robot_name] = float(np.linalg.norm(avg_neck - avg_base))
            bases[robot_name] = avg_base
        if self.normalize == "none":
            for robot_name, info in rc.classifying_joint_info.items():
                base = bases[robot_name]
                data = info["data_to_classify"]
                for g_idx, group in enumerate(classification_groups):
                    part = group[0]
                    data[part][:, 3:6] -= base
                rc.classifying_joint_info[robot_name]["data_to_classify"] = data
        elif self.normalize == "all":
            gmax = max(diffs.values()) if diffs else 0.0
            if gmax > 1e-9:
                for robot_name, info in rc.classifying_joint_info.items():
                    base = bases[robot_name]
                    scale = 1.0 / gmax
                    data = info["data_to_classify"]
                    for g_idx, group in enumerate(classification_groups):
                        part = group[0]
                        data[part][:, 3:6] = (data[part][:, 3:6] - base) * scale
                    rc.classifying_joint_info[robot_name]["data_to_classify"] = data
        elif self.normalize == "each":
            for robot_name, info in rc.classifying_joint_info.items():
                base = bases[robot_name]
                diff = diffs[robot_name]
                if diff < 1e-9:
                    diff = 1.0
                scale = 1.0 / diff
                data = info["data_to_classify"]
                for g_idx, group in enumerate(classification_groups):
                    part = group[0]
                    data[part][:, 3:6] = (data[part][:, 3:6] - base) * scale
                rc.classifying_joint_info[robot_name]["data_to_classify"] = data
        else:
            raise ValueError(f"Invalid normalize: {self.normalize}")

        # 6) per-robot stack & full stack
        data_list: List[np.ndarray] = []
        labels: List[str] = []
        for robot_name, robot_dict in rc.classifying_joint_info.items():
            data_dict = robot_dict.get("data_to_classify")
            if data_dict is None:
                continue
            part_arrays = [data_dict[p] for p in group_order]
            if len(part_arrays) == 0:
                continue
            stacked = np.vstack(part_arrays)
            data_list.append(stacked)
            labels.append(robot_name)
        if not data_list:
            raise RuntimeError("No data_to_classify found after building dataset.")
        all_data = np.stack(data_list, axis=0)  # (R, N_total, 6)

        # 7) update RC state (same fields as before)
        rc.max_joints_per_group = max_joints_per_group
        rc.classify_data_shape  = all_data.shape

        return KinematicsDataset(
            arrays=all_data,
            labels=labels,
            max_joints_per_group=max_joints_per_group,
            group_order=group_order,
            selected_indices=selected_indices,
            feature_dim=6,
        )
    
    # ------------------ SDH path ------------------
    def _build_dh_sdh(self, rc, classification_groups, group_order, max_joints_per_group, selected_indices, verbose):
        """
        Extracts Standard DH parameters directly from (axis_world, position).
        - chains: spine (torso+neck), left_arm, right_arm
        - anchors:
            * spine: world frame (self.spine_anchor_axis: 'x'|'z'|'auto'), q0 = origin
            * arms : z0 = axis of the last torso joint, q0 = foot of the common normal on that axis (P_parent),
                    x0 = direction of the common normal between parent z and the first child z  (absorbs the attachment into the first DH)
        - returns: stores (N,4)[θ,d,a,α] in rc.classifying_joint_info[robot]['data_to_classify_dh'], then normalizes/stacks
        """
        # ---------------------------
        # 0) define chains
        # ---------------------------
        chain_defs = OrderedDict([
            ("spine",["torso"]),
            ("neck", ["neck"]),
            ("left_arm",  ["left_shoulder_girdle","left_shoulder","left_upperarm","left_elbow","left_forearm","left_wrist"]),
            ("right_arm", ["right_shoulder_girdle","right_shoulder","right_upperarm","right_elbow","right_forearm","right_wrist"]),
        ])

        # ---------------------------
        # shared helpers
        # ---------------------------
        WORLD_AXES = np.eye(3, dtype=float)
        ORIGIN     = np.zeros(3, dtype=float)

        def _first_valid_row(arr: np.ndarray, atol=1e-12):
            """Return the index of the first non-zero row (not all zeros), or None."""
            if not (isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 2):
                return None
            mask = ~np.all(np.isclose(arr, 0.0, atol=atol), axis=1)
            return int(np.argmax(mask)) if np.any(mask) else None

        def rebase_first_d_from_base_inplace(
            data_dh: dict,
            base_pos: np.ndarray,
            base_origin=np.zeros(3),
            base_z=np.array([0.,0.,1.]),
            spine_parts=('torso','neck'),                 # spine priority
            arm_roots=('left_shoulder_girdle','right_shoulder_girdle'),
            arm_fallback={'left_shoulder_girdle':'left_shoulder',
                        'right_shoulder_girdle':'right_shoulder'},
            atol=1e-12,
        ):
            """
            the base height d0_offset = (base_pos - base_origin)·base_z_unit
            subtract it from d of the 'first root joints'.

            Rules:
            - if the torso has a valid row: subtract only from d of the torso's 'first valid row'.
            - if the torso is empty: also subtract from the first valid row of the neck and of each shoulder_girdle (shoulder if empty).
            Returns: (data_dh, info)  // info records where it was applied
            """
            bz = np.asarray(base_z, float)
            bz /= (np.linalg.norm(bz) + 1e-12)
            d0_offset = float(np.dot(np.asarray(base_pos, float) - np.asarray(base_origin, float), bz))

            applied = []

            # 1) if there is a torso, correct only the torso
            torso_idx = _first_valid_row(data_dh.get('torso'))
            if torso_idx is not None:
                data_dh['torso'][torso_idx, 1] -= d0_offset
                applied.append(('torso', torso_idx, d0_offset))
                return data_dh, {'d0_offset': d0_offset, 'applied': applied}

            # 2) if the torso is empty: also correct the neck + each arm root (or fallback)
            # neck
            neck_idx = _first_valid_row(data_dh.get('neck'))
            if neck_idx is not None:
                data_dh['neck'][neck_idx, 1] -= d0_offset
                applied.append(('neck', neck_idx, d0_offset))

            # arms (girdle first, otherwise shoulder)
            for root in arm_roots:
                part = root
                idx  = _first_valid_row(data_dh.get(part))
                if idx is None and root in arm_fallback:
                    part_f = arm_fallback[root]
                    idx    = _first_valid_row(data_dh.get(part_f))
                    if idx is not None:
                        part = part_f
                if idx is not None:
                    data_dh[part][idx, 1] -= d0_offset
                    applied.append((part, idx, d0_offset))

            return data_dh, {'d0_offset': d0_offset, 'applied': applied}

        # =========================================================
        # 1) extract SDH per robot -> map into per-group (N,5) arrays
        # =========================================================
        for robot_name, joint_info in rc.classifying_joint_info.items():
            info = joint_info  # convenience alias (contains axis_world/position of each joint)
            # (a) allocate empty arrays per group
            robot_data: Dict[str, np.ndarray] = {}
            for g_idx, group in enumerate(classification_groups):
                part = group[0]
                robot_data[part] = np.zeros((max_joints_per_group[g_idx], 5), dtype=float)  # θ,d,a,α

            # (b) build chains: concatenate the actual joint order of the defined parts
            chains: Dict[str, List[str]] = {}
            for cname, parts in chain_defs.items():
                seq: List[str] = []
                if cname == 'spine': 
                    seq+=['base_link']
                else: 
                    seq+=[chains['spine'][-1]]
                for p in parts:
                    seq += rc.classify_data.get(robot_name, {}).get(p, [])
                if len(seq) > 0:
                    chains[cname] = seq

            if not chains:
                # if empty, store the zero matrix as is and move to the next robot
                rc.classifying_joint_info[robot_name]['data_to_classify_dh'] = robot_data
                continue

            # (d) extract SDH
            extractor = SDHKinematicsExtractor(chains=chains)
            dh_map = extractor.extract(rc.robot_models[robot_name])  # {chain: [(joint_name, DHParam), ...]}

            # (e) chain result -> index per joint
            dh_by_joint: Dict[str, Tuple[float, float, float, float, float]] = {}
            for _c, items in dh_map.items():
                for jn, dh in items:
                    dh_by_joint[jn] = (float(dh.theta), float(dh.d), float(dh.a), float(dh.alpha),float(dh.axis_offset))

            # (f) write into per-group arrays (keep classification order)
            write_cursor: Dict[str, int] = {group[0]: 0 for group in classification_groups}
            group_to_jnames: Dict[str, List[str]] = {
                group[0]: rc.classify_data[robot_name].get(group[0], []) for group in classification_groups
            }
   
            for part, jnames in group_to_jnames.items():
                # array/cursor of the original part
                for jn in jnames:
                    # decide the routing target
                    target_part = self._tcp_group_for(part) if (isinstance(jn, str) and jn.startswith("virtual_")) else part

                    # array/cursor of this part
                    arr = robot_data[target_part]
                    k   = write_cursor[target_part]

                    if jn in dh_by_joint:
                        theta, d, a, alpha, tau = dh_by_joint[jn]
                        row5 = np.array([theta, d, a, alpha, tau], dtype=float)

                        # capacity check (should not overflow normally; can grow defensively)
                        if k >= arr.shape[0]:
                            # defensive growth (capacity is normally sized in the pre-scan, so ideally this path is never taken)
                            arr = np.vstack([arr, np.zeros((1, arr.shape[1]), dtype=float)])
                            robot_data[target_part] = arr
                            # max_joints_per_group affects the global dataset shape -> resolve it in the pre-scan if possible

                        arr[k, :5] = row5
                        write_cursor[target_part] = k + 1
            # (g) store in rc (used later by the fill/normalize/stack steps)
            rc.classifying_joint_info[robot_name]['data_to_classify_dh'] = robot_data

        # =========================================================
        # 2) === fill: lengths (a,d), angles (theta,alpha) ===   (original kept as requested)
        # =========================================================
        for robot_name, info in rc.classifying_joint_info.items():
            data = info["data_to_classify_dh"]
            for g_idx, group in enumerate(classification_groups):
                part = group[0]
                arr = data[part]
                # decide whether missing
                zero_all = np.all(arr == 0, axis=-1)
                # fill lengths (a, d)
                if self.dh_length_fill == "local_average":
                    valid = arr[~zero_all, 1:3]  # d,a
                    if len(valid) > 0:
                        mean_da = np.mean(valid, axis=0)
                        arr[zero_all, 1:3] = mean_da
                elif self.dh_length_fill != "zero":
                    raise ValueError(f"Invalid dh_length_fill: {self.dh_length_fill}")
                # fill angles (theta, alpha): keep the default 0
                if self.dh_angle_fill == "local_average":
                    valid = arr[~zero_all, [0,3]]  # theta, alpha
                    if len(valid) > 0:
                        mean_ta = np.mean(valid, axis=0)
                        arr[zero_all, 0] = mean_ta[0]
                        arr[zero_all, 3] = mean_ta[1]
                elif self.dh_angle_fill != "zero":
                    raise ValueError(f"Invalid dh_angle_fill: {self.dh_angle_fill}")
                data[part] = arr
            rc.classifying_joint_info[robot_name]["data_to_classify_dh"] = data

        # =========================================================
        # 3) === normalize: scale only a,d ===        (original kept as requested)
        # =========================================================
        diffs = {}
        bases = {}
        for robot_name, info in rc.classifying_joint_info.items():
            base, neck = rc.get_avg_neck_and_base(robot_name, self.use_shoulder_pos, False)
            diffs[robot_name] = float(np.linalg.norm(neck - base))
            bases[robot_name] = base  # no q normalization for SDH

        def _scale_da(arr, scale):
            arr[:, [1,2,4]] *= scale   # d, a,tau
            return arr

        if self.normalize == "all":
            gmax = max(diffs.values()) if diffs else 0.0
            if gmax > 1e-9:
                for robot_name, info in rc.classifying_joint_info.items():
                    data = info["data_to_classify_dh"]
                    data,_ = rebase_first_d_from_base_inplace(data, bases[robot_name],
                                   base_origin=np.zeros(3),
                                   base_z=np.array([0.,0.,1.]))  # use a per-robot z0 if needed
                    for g_idx, group in enumerate(classification_groups):
                        part = group[0]
                        data[part] = _scale_da(data[part], 1.0/gmax)
                    rc.classifying_joint_info[robot_name]["data_to_classify_dh"] = data
        elif self.normalize == "each":
            for robot_name, info in rc.classifying_joint_info.items():
                diff = diffs[robot_name]
                scale = 1.0/diff if diff > 1e-9 else 1.0
                data = info["data_to_classify_dh"]
                # data = _rebase_spine_d(data, bases[robot_name],
                #                    base_z=np.array([0.,0.,1.]))  # use a per-robot z0 if needed
                data, _ = rebase_first_d_from_base_inplace(
                    data,
                    bases[robot_name],
                    base_origin=np.zeros(3),
                    base_z=np.array([0.,0.,1.]),
                    spine_parts=('torso','neck'),
                    arm_roots=('left_shoulder_girdle','right_shoulder_girdle'),
                    arm_fallback={'left_shoulder_girdle':'left_shoulder',
                                'right_shoulder_girdle':'right_shoulder'},
                )
                for g_idx, group in enumerate(classification_groups):
                    part = group[0]
                    data[part] = _scale_da(data[part], scale)
                rc.classifying_joint_info[robot_name]["data_to_classify_dh"] = data
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Invalid normalize: {self.normalize}")

        # =========================================================
        # 4) === stack ===                              (original kept as requested)
        # =========================================================
        data_list: List[np.ndarray] = []
        labels: List[str] = []
        for robot_name, robot_dict in rc.classifying_joint_info.items():
            data_dict = robot_dict.get("data_to_classify_dh")
            if data_dict is None:
                continue
            part_arrays = [data_dict[p] for p in group_order]
            if len(part_arrays) == 0:
                continue
            stacked = np.vstack(part_arrays)   # (N_total, 5)
            data_list.append(stacked)
            labels.append(robot_name)
        if not data_list:
            raise RuntimeError("No DH data_to_classify found.")
        all_data = np.stack(data_list, axis=0)  # (R, N_total, 5)

        # update rc state
        rc.max_joints_per_group = max_joints_per_group
        rc.classify_data_shape  = all_data.shape
        rc.group_order = group_order

        return KinematicsDataset(
            arrays=all_data,
            labels=labels,
            max_joints_per_group=max_joints_per_group,
            group_order=group_order,
            selected_indices=selected_indices,
            feature_dim=5,
        )