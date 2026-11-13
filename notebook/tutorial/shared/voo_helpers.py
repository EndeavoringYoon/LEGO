"""LEGO VOO helpers: extracted from notebook/ae_test/experiment1_real.ipynb.

This module consolidates the helper functions that previously lived inline in
multiple cells of `experiment1_real.ipynb`. Source cell map:

  cell 3   → _similarity_transform_umeyama, _sequence_pa_mpjpe, _sequence_mpjpe
  cell 17  → sum_without_negatives, get_smpl_pos, get_cmu_pos, r_sq, retarget_and_procrustes
  cell 18  → override_joi, evaluate, evaluate_all_motions
  cell 19  → DEFAULT_VOO_PARAMS, get_rngs
  cell 26  → run_once
  cell 22  → run_once_all
  cell 32  → evaluate_raw

Usage from a tutorial notebook:
    import sys, os
    for p in ['../../package', '../../package/yoon', '../../package/mujoco_helper',
              '../../package/kinematics_helper', '../../package/utility',
              '../../package/bvh_helper', '../../package/smpl',
              '../../package/motion_retarget', '../../package/irvae_tools',
              '../../package/kinematics', '../../asset']:
        sys.path.insert(0, os.path.abspath(p))
    from shared.voo_helpers import (
        DEFAULT_VOO_PARAMS, get_rngs,
        override_joi, evaluate, evaluate_all_motions, evaluate_raw,
        run_once, run_once_all,
        retarget_and_procrustes,
    )

Conventions:
  - All functions are pure: they take all needed state as arguments (no module-level
    globals beyond DEFAULT_VOO_PARAMS).
  - VOO loops use a `params` dict (e.g., DEFAULT_VOO_PARAMS) so tutorials can override.
  - Random number generation goes through `rng` (numpy Generator), not the global state.
"""

# ---- Standard ----
import os
import numpy as np
import torch as th
from copy import deepcopy
from scipy.spatial import distance, procrustes

# ---- Project package imports (assumed on sys.path by caller) ----
from mujoco_parser import MuJoCoParserClass
from data_utils import load_saved_model
from motion_retarget import directional_preretarget_postrigging, get_joi_body_name_of_irae
from bvh_parser import get_chains_from_bvh_cmu_mocap
from smpl_parser import get_chains_from_smpl
from smpl_model import SMPLModelClass
from utils import animate_chains_slider


# =============================================================================
# Default VOO hyperparameters (paper arXiv:2604.08636v1 §V-A + experiment1_real.ipynb cell 19)
# =============================================================================

DEFAULT_VOO_PARAMS = {
    'p_global': 0.55,         # global exploration probability
    'length_scale': 3.5,
    'invlen': 1.0 / (3.5 ** 2),  # ≈ 0.0816
    'n_loop_switch': 20,      # local sampling switch threshold
    'max_loop': 500,
    'sigma_coef': 0.6,        # local proposal scale
    'x_minval': -15.0,
    'x_maxval': 15.0,
    'n_init': 16,
    'max_iter': 30,
    'eval': 'PA-MPJPE',       # objective metric
    'jnt_penalty': True,
    'jnt_coeff': 3.5,         # λ_joint
    'animate': False,
    'master_seed': 123456789,
    'n_runs': 10,
}


def get_rngs(master_seed=123456789, n_runs=10):
    """Spawn n_runs independent numpy Generators from a master seed."""
    ss = np.random.SeedSequence(master_seed)
    return [np.random.default_rng(cs) for cs in ss.spawn(n_runs)]


# =============================================================================
# Section A: PA-MPJPE helpers (from cell 3)
# =============================================================================

def _similarity_transform_umeyama(P, G, with_scale=True):
    """Umeyama (1991) similarity transform that aligns P to G.
    P, G: (J, 3). Returns aligned P_hat (J, 3) + (scale, R, t)."""
    assert P.shape == G.shape and P.shape[1] == 3
    S1, S2 = P.T, G.T
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = np.sum(X1 ** 2)
    K = X1 @ X2.T
    U, s, Vt = np.linalg.svd(K)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    if with_scale:
        scale = np.trace(R @ K) / var1
    else:
        scale = 1.0
    t = mu2 - scale * R @ mu1
    S1_hat = scale * R @ S1 + t
    return S1_hat.T, scale, R, t


def _sequence_pa_mpjpe(gt_seq, pred_seq, with_scale=True):
    """Per-frame similarity alignment then mean L2 error over all joints,
    averaged over all frames. gt_seq, pred_seq: (T, J, 3)."""
    assert gt_seq.shape == pred_seq.shape and gt_seq.shape[2] == 3
    T = gt_seq.shape[0]
    per_frame = np.empty(T)
    for t in range(T):
        P_hat, _, _, _ = _similarity_transform_umeyama(pred_seq[t], gt_seq[t], with_scale=with_scale)
        per_frame[t] = float(np.mean(np.linalg.norm(P_hat - gt_seq[t], axis=1)))
    return float(np.mean(per_frame))


def _sequence_mpjpe(gt_seq, pred_seq, root_strategy='centroid'):
    """MPJPE with root removed per frame. root_strategy ∈ {'centroid', 'first'}."""
    assert gt_seq.shape == pred_seq.shape
    if root_strategy == 'centroid':
        gt_root = gt_seq.mean(axis=1, keepdims=True)
        pr_root = pred_seq.mean(axis=1, keepdims=True)
    elif root_strategy == 'first':
        gt_root = gt_seq[:, :1, :]
        pr_root = pred_seq[:, :1, :]
    else:
        raise ValueError(f'unknown root_strategy {root_strategy}')
    gt_c = gt_seq - gt_root
    pr_c = pred_seq - pr_root
    return float(np.mean(np.linalg.norm(pr_c - gt_c, axis=2)))


# =============================================================================
# Section B: Retargeting helpers (from cell 17)
# =============================================================================

def sum_without_negatives(disparity):
    """Sum entries excluding -999 sentinels (used as failure marker for Procrustes)."""
    return sum(v for v in disparity if v != -999)


def get_smpl_pos(chains, joi_body_name, base_between_pelvis=True):
    """Extract SMPL joint positions matching joi_body_name.keys() over a chain sequence.
    Returns (T, J, 3)."""
    arrays = []
    joi_keys = list(joi_body_name.keys())
    joi_keys = [k.replace('spine', 'spine1') for k in joi_keys]
    for chain in chains:
        small = []
        for j, key in enumerate(joi_keys):
            if key in chain.joint_names:
                if base_between_pelvis and key == 'base':
                    p_rp, _ = chain.get_joint_pR(joint_name='rp')
                    p_lp, _ = chain.get_joint_pR(joint_name='lp')
                    p = 0.5 * (p_rp + p_lp)
                else:
                    p, _ = chain.get_joint_pR(joint_name=key)
                small.append(p)
        arrays.append(np.array(small)[np.newaxis, :, :])
    return np.vstack(arrays)


def get_cmu_pos(chains, joi_body_name, base_between_pelvis=True, fix_root=False):
    """Extract CMU mocap joint positions matching joi_body_name.keys() over a chain sequence.
    Returns (T, J, 3)."""
    pairs = {
        'base': 'Hips', 'spine': 'Spine',
        'rs': 'RightArm', 're': 'RightForeArm', 'rw': 'RightHand', 'rh': 'RightHand',
        'ls': 'LeftArm',  'le': 'LeftForeArm',  'lw': 'LeftHand',  'lh': 'LeftHand',
        'neck': 'Neck', 'head': 'Head',
        'rp': 'RightUpLeg', 'rk': 'RightLeg', 'ra': 'RightFoot', 'rf': 'RightFoot',
        'lp': 'LeftUpLeg',  'lk': 'LeftLeg',  'la': 'LeftFoot',  'lf': 'LeftFoot',
        'rtoe': 'RightToeBase', 'ltoe': 'LeftToeBase',
    }
    arrays = []
    joi_keys = list(joi_body_name.keys())
    for chain in chains:
        small = []
        for key in joi_keys:
            if key == 'root':
                # Compute root but DON'T append (matches original cell 17 behavior)
                continue
            if pairs.get(key) not in chain.joint_names:
                continue
            if base_between_pelvis and key == 'base':
                p_rp, _ = chain.get_joint_pR(joint_name=pairs['rp'])
                p_lp, _ = chain.get_joint_pR(joint_name=pairs['lp'])
                p = 0.5 * (p_rp + p_lp)
            elif key == 'neck':
                # Use (Neck.x, Neck.y, mid_shoulder.z): original behavior
                p_neck, _ = chain.get_joint_pR(joint_name=pairs[key])
                p_rs, _ = chain.get_joint_pR(joint_name=pairs['rs'])
                p_ls, _ = chain.get_joint_pR(joint_name=pairs['ls'])
                p_neck_z = 0.5 * (p_rs + p_ls)
                p = np.array([p_neck[0], p_neck[1], p_neck_z[2]])
            else:
                p, _ = chain.get_joint_pR(joint_name=pairs[key])
            small.append(p)
        arrays.append(np.array(small)[np.newaxis, :, :])
    return np.vstack(arrays)


def r_sq(x1, x2, x_range=1.0, invlen=5.0):
    """Scaled squared-Euclidean pairwise distances. Used by VOO Voronoi-radius computation."""
    x1s = invlen * x1 / x_range
    x2s = invlen * x2 / x_range
    return distance.cdist(x1s, x2s, 'sqeuclidean')


def retarget_and_procrustes(
        env, pkl_path, file_path, motion_type='smpl',
        ik_p_spine=False, ik_p_neck=True, leg_fixed=False, ik_R_hand=False,
        animate_chains=False, animate_ik=False, verbose=False,
        override=None, eval='procrustes', jnt_penalty=True, jnt_coeff=1,
        unit_scale=1000, p2_root_strategy='centroid'):
    """Retarget a source motion onto a robot env and return discrepancy.

    Returns:
      - eval='procrustes': sum of joint-wise Procrustes disparities (sentinel −999 excluded)
      - eval='PA-MPJPE'  : mean PA-MPJPE (Protocol #1) × unit_scale
    Joint penalty (+ jnt_coeff * env.n_joint) is added if jnt_penalty=True.
    """
    joi_body_name = get_joi_body_name_of_irae(override=override)
    joi_smpl_name = list(joi_body_name.keys())
    joi_robot_name = list(joi_body_name.values())

    if motion_type == 'smpl':
        smpl = SMPLModelClass(model_path=pkl_path)
        secs, chains = get_chains_from_smpl(
            file_path=file_path, smpl=smpl, intp_traj=False, plot_chain_graph=False)
        motion_pos = get_smpl_pos(chains, joi_body_name)
    elif motion_type == 'cmu':
        secs, chains = get_chains_from_bvh_cmu_mocap(
            bvh_path=file_path, p_rate=0.056444, zup=True,
            plot_chain_graph=False, verbose=False)
        motion_pos = get_cmu_pos(chains, joi_body_name)
    else:
        raise ValueError(f'unknown motion_type {motion_type!r}')

    L = len(chains)
    dt = secs[1] - secs[0]
    if animate_chains:
        animate_chains_slider(env, secs, chains)
    if verbose:
        print(f'[{file_path}] L={L} dt={dt:.4f}s HZ={int(1/dt)}')

    qpos_dir_list = directional_preretarget_postrigging(
        env, secs, chains,
        ik_p_spine=ik_p_spine, ik_p_neck=ik_p_neck, leg_fixed=leg_fixed,
        src_name=motion_type, trgt_name='irae',
        joint_names_for_ik=None,
        max_ik_tick=100, ik_stepsize=1, ik_err_th=1e-3,
        print_ik_failure=False,
        ik_R_hand=ik_R_hand, ik_R_foot=not leg_fixed,
        reset_env=True, animate=animate_ik,
        plot_joi_src=False, plot_joi_trgt=False,
        render_every=1, verbose=verbose, print_every=30,
        pause_each_tick=False, post_rigging=True, override=override)

    env.reset(step=True)
    if animate_ik:
        env.init_viewer(title='Pre-Retarget', transparent=True)

    robot_joi_pos = np.zeros((L, len(joi_robot_name), 3), dtype=np.float32)
    for tick in range(L):
        env.forward(q=qpos_dir_list[tick])
        for body_idx in range(1, env.n_body):
            body_name = env.body_names[body_idx]
            if body_name in joi_robot_name:
                for j in [i for i, v in enumerate(joi_robot_name) if v == body_name]:
                    robot_joi_pos[tick, j, :] = env.data.body(body_name).xpos

    robot_joi_pos_wo_root = np.delete(robot_joi_pos, 0, axis=1)
    joi_smpl_name_wo_root = joi_smpl_name[1:]
    assert motion_pos.shape[:2] == robot_joi_pos_wo_root.shape[:2], (
        f'Shape mismatch: GT {motion_pos.shape} vs Pred {robot_joi_pos_wo_root.shape}')

    eval_key = str(eval).lower()
    if eval_key == 'procrustes':
        per_joint = []
        for j in range(motion_pos.shape[1]):
            try:
                _, _, disparity = procrustes(motion_pos[:, j, :], robot_joi_pos_wo_root[:, j, :])
                per_joint.append(disparity)
            except Exception as e:
                print(f'[Procrustes] joint {j} ({joi_smpl_name_wo_root[j]}) failed: {e}')
                per_joint.append(-999.0)
        if verbose:
            for j, d in enumerate(per_joint):
                print(f'  joint {j} ({joi_smpl_name_wo_root[j]}): disparity={d:.6f}')
        result = float(sum_without_negatives(per_joint))
    elif eval_key in ('pa-mpjpe', 'p-mpjpe', 'p1'):
        pa = _sequence_pa_mpjpe(motion_pos, robot_joi_pos_wo_root, with_scale=True) * unit_scale
        if verbose:
            p2 = _sequence_mpjpe(motion_pos, robot_joi_pos_wo_root, root_strategy=p2_root_strategy) * unit_scale
            print(f'PA-MPJPE: {pa:.3f} | MPJPE(root={p2_root_strategy}): {p2:.3f}')
        result = float(pa)
    else:
        if animate_ik: env.viewer.close()
        raise ValueError("eval must be one of {'procrustes', 'PA-MPJPE'}")

    if animate_ik:
        env.viewer.close()
    if jnt_penalty:
        result += jnt_coeff * env.n_joint
    return result


# =============================================================================
# Section C: Objective functions used by VOO (from cells 18, 32)
# =============================================================================

def override_joi(env, repr_kind):
    """Build JOI body-name mapping from an env, choosing the deepest matching body per group.
    repr_kind controls the fallback chain (different for screw vs dh due to TCP groups)."""
    body_names = env.body_names
    group_map = {k: [] for k in [
        'base_link','torso','neck',
        'right_shoulder_girdle','right_shoulder','right_upperarm',
        'right_elbow','right_forearm','right_wrist',
        'left_shoulder_girdle','left_shoulder','left_upperarm',
        'left_elbow','left_forearm','left_wrist',
        'right_tcp','left_tcp',
    ]}
    for name in body_names:
        for group in group_map:
            if group in name:
                group_map[group].append(name)

    fallback_screw = {
        'right_elbow': ['right_upperarm','right_shoulder','right_shoulder_girdle','torso','base_link'],
        'right_wrist': ['right_forearm','right_elbow','right_upperarm','right_shoulder','right_shoulder_girdle','torso','base_link'],
        'left_elbow':  ['left_upperarm','left_shoulder','left_shoulder_girdle','torso','base_link'],
        'left_wrist':  ['left_forearm','left_elbow','left_upperarm','left_shoulder','left_shoulder_girdle','torso','base_link'],
        'torso': ['base_link'], 'neck': ['torso','base_link'],
        'right_shoulder': ['right_shoulder_girdle','torso','base_link'],
        'left_shoulder':  ['left_shoulder_girdle','torso','base_link'],
    }
    fallback_dh = {
        'right_elbow': ['right_upperarm','right_shoulder','right_shoulder_girdle','torso','base_link'],
        'right_wrist': ['right_tcp','right_wrist','right_forearm','right_elbow','right_upperarm','right_shoulder','right_shoulder_girdle','torso','base_link'],
        'left_elbow':  ['left_upperarm','left_shoulder','left_shoulder_girdle','torso','base_link'],
        'left_wrist':  ['left_tcp','left_wrist','left_forearm','left_elbow','left_upperarm','left_shoulder','left_shoulder_girdle','torso','base_link'],
        'torso': ['base_link'], 'neck': ['torso','base_link'],
        'right_shoulder': ['right_shoulder_girdle','torso','base_link'],
        'left_shoulder':  ['left_shoulder_girdle','torso','base_link'],
    }
    fallback = fallback_screw if repr_kind == 'screw' else fallback_dh

    targets = ['torso','neck',
               'right_shoulder','right_elbow','right_wrist',
               'left_shoulder','left_elbow','left_wrist']
    aliases = ['spine','neck','rs','re','rw','ls','le','lw']

    results = {}
    for group, key in zip(targets, aliases):
        matches = group_map.get(group, [])
        if matches:
            sel = matches[-1]
            results[key] = sel
            if key == 'rw': results['rh'] = sel
            if key == 'lw': results['lh'] = sel
            continue
        for fb in fallback.get(group, []):
            cands = group_map.get(fb, [])
            if cands:
                sel = cands[-1]
                results[key] = sel
                if key == 'rw': results['rh'] = sel
                if key == 'lw': results['lh'] = sel
                break
    return results


def _build_envs_from_decoded(robot_chars, decoded_120_or_180, repr_kind, threshold=0.5):
    """Given decoded data (either symmetric 120 or full 180), call build/convert/visualize_data
    to produce robot_chars.xml dict (one MJCF per decoded sample)."""
    full = robot_chars.build_full_interp_from_symmetric(
        interp_data_sym=decoded_120_or_180, threshold=threshold, repr_kind=repr_kind)
    data_to_vis = robot_chars.convert_flat_interp_to_dict(
        flat_interp_data=full, repr_kind=repr_kind)
    robot_chars.visualize_data_to_classify(
        data_to_vis, save_video=False, vis=False, xml=True, repr_kind=repr_kind)
    return robot_chars.xml


def evaluate(x, model, robot_chars, motion_type, pkl_path, file_path,
             animate=False, repr_kind='screw', eval='PA-MPJPE',
             jnt_penalty=True, jnt_coeff=3.5, verbose=False):
    """Latent-space objective. x: (N, z_dim). Returns (N, 1)."""
    results = np.zeros((x.shape[0], 1))
    device = next(model.parameters()).device
    decode = model.decode(th.tensor(x, device=device, dtype=th.float32)).detach().numpy()
    xml_dict = _build_envs_from_decoded(robot_chars, decode, repr_kind)
    for i, (name, xml) in enumerate(xml_dict.items()):
        env = MuJoCoParserClass(name=name, xml_string=xml, verbose=False)
        if len(env.get_idxs_jac(joint_names=env.rev_joint_names)) <= 5:
            results[i] = 300.0 if str(eval).lower() in ('pa-mpjpe', 'p-mpjpe', 'p1') else 10.0
            continue
        override = override_joi(env, repr_kind)
        results[i] = retarget_and_procrustes(
            env, pkl_path=pkl_path, file_path=file_path, motion_type=motion_type,
            ik_p_spine=True, ik_p_neck=False, leg_fixed=True, ik_R_hand=False,
            animate_chains=False, animate_ik=animate, verbose=False,
            override=override, eval=eval, jnt_penalty=jnt_penalty, jnt_coeff=jnt_coeff)
        if verbose:
            print(f'  {x[i]} → {float(results[i]):.4f}')
    return results


def evaluate_all_motions(x, model, robot_chars, motion_type, pkl_path, file_path_list,
                         animate=False, repr_kind='screw', eval='PA-MPJPE',
                         jnt_penalty=True, jnt_coeff=3.5, verbose=False):
    """Motion-set objective: sum of per-motion PA-MPJPE across file_path_list, joint penalty added once.

    Used to reproduce paper Table I Motion Set column. model=None falls back to raw-x mode
    (matches evaluate_all_motions with model=None branch in cell 18)."""
    results = np.zeros((x.shape[0], 1), dtype=np.float64)
    if model is not None:
        device = next(model.parameters()).device
        decode = model.decode(th.tensor(x, device=device, dtype=th.float32)).detach().numpy()
    else:
        decode = x
    xml_dict = _build_envs_from_decoded(robot_chars, decode, repr_kind)
    for i, (name, xml) in enumerate(xml_dict.items()):
        env = MuJoCoParserClass(name=name, xml_string=xml, verbose=False)
        override = override_joi(env, repr_kind)
        if len(env.get_idxs_jac(joint_names=env.rev_joint_names)) <= 5:
            fail = 300.0 if str(eval).lower() in ('pa-mpjpe', 'p-mpjpe', 'p1') else 10.0
            results[i] += fail * len(file_path_list)
            continue
        for fp in file_path_list:
            r = retarget_and_procrustes(
                env, pkl_path=pkl_path, file_path=fp, motion_type=motion_type,
                ik_p_spine=True, ik_p_neck=False, leg_fixed=True, ik_R_hand=False,
                animate_chains=False, animate_ik=animate, verbose=False,
                override=override, eval=eval, jnt_penalty=False, jnt_coeff=jnt_coeff)
            results[i] += r
            if verbose:
                print(f'  {x[i]} / {os.path.basename(fp)} → {r:.4f}')
        if jnt_penalty:
            results[i] += env.n_joint * jnt_coeff
    return results


def evaluate_raw(x, robot_chars, motion_type, pkl_path, file_path,
                 animate=False, repr_kind='screw', eval='PA-MPJPE',
                 jnt_penalty=True, jnt_coeff=3.5, verbose=False):
    """Raw 120D-x objective. Same as evaluate() but skips the AE decode step."""
    results = np.zeros((x.shape[0], 1))
    xml_dict = _build_envs_from_decoded(robot_chars, x, repr_kind)
    for i, (name, xml) in enumerate(xml_dict.items()):
        env = MuJoCoParserClass(name=name, xml_string=xml, verbose=False)
        if len(env.body_names) <= 2:
            results[i] = 10.0
            continue
        override = override_joi(env, repr_kind)
        results[i] = retarget_and_procrustes(
            env, pkl_path=pkl_path, file_path=file_path, motion_type=motion_type,
            ik_p_spine=True, ik_p_neck=False, leg_fixed=True, ik_R_hand=False,
            animate_chains=False, animate_ik=animate, verbose=False,
            override=override, eval=eval, jnt_penalty=jnt_penalty, jnt_coeff=jnt_coeff)
        if verbose:
            print(f'  {x[i]} → {float(results[i]):.4f}')
    return results


# =============================================================================
# Section D: VOO loops (from cells 22, 26)
# =============================================================================

def _voo_step(rng, x_voo, y_voo, x_dim, params):
    """Propose next x_sel given current (x_voo, y_voo) and rng."""
    rand = rng.random
    randn = rng.standard_normal
    if rand(1)[0] < params['p_global']:
        return params['x_minval'] + (params['x_maxval'] - params['x_minval']) * rand((1, x_dim))
    idx_min = int(np.argmin(y_voo))
    x_min_pt = x_voo[idx_min, :].reshape((1, -1))
    dists = r_sq(x_min_pt, np.delete(x_voo, [idx_min], axis=0), invlen=params['invlen'])
    dist_min = float(np.min(dists))
    n_loop = 0
    while True:
        n_loop += 1
        if n_loop < params['n_loop_switch']:
            x_sel = params['x_minval'] + (params['x_maxval'] - params['x_minval']) * rand((1, x_dim))
        else:
            sigma = params['sigma_coef'] * np.sqrt(max(dist_min, 1e-12) / float(x_dim))
            x_sel = x_min_pt + sigma * randn((1, x_dim))
            x_sel = np.clip(x_sel, params['x_minval'], params['x_maxval'])
        dist_sel = r_sq(x_min_pt, x_sel, invlen=params['invlen'])
        if dist_sel < dist_min or n_loop >= params['max_loop']:
            if n_loop >= params['max_loop']:
                x_sel = params['x_minval'] + (params['x_maxval'] - params['x_minval']) * rand((1, x_dim))
            return x_sel


def run_once(model_spec, rng, file_path, repr_kind, robot_chars,
             motion_type='cmu', pkl_path=None, params=None):
    """Per-motion VOO run (paper Table I per-motion). Returns (X, Y) trajectory.

    model_spec: str (model name to load via load_saved_model) or None (raw-x baseline).
    """
    if params is None:
        params = DEFAULT_VOO_PARAMS
    if model_spec is not None:
        ae, _ = load_saved_model(model_name=model_spec, device='cpu')
        x_dim = ae.z_dim
    else:
        ae = None
        x_dim = 120 if repr_kind == 'screw' else 105

    rand = rng.random
    x_voo = params['x_minval'] + (params['x_maxval'] - params['x_minval']) * rand((params['n_init'], x_dim))
    if ae is not None:
        y_voo = evaluate(x_voo, ae, robot_chars, motion_type, pkl_path, file_path,
                         animate=params['animate'], repr_kind=repr_kind,
                         eval=params['eval'], jnt_penalty=params['jnt_penalty'],
                         jnt_coeff=params['jnt_coeff'])
    else:
        y_voo = evaluate_raw(x_voo, robot_chars, motion_type, pkl_path, file_path,
                             animate=params['animate'], repr_kind=repr_kind,
                             eval=params['eval'], jnt_penalty=params['jnt_penalty'],
                             jnt_coeff=params['jnt_coeff'])

    for _ in range(params['max_iter']):
        x_sel = _voo_step(rng, x_voo, y_voo, x_dim, params)
        if ae is not None:
            y_sel = evaluate(x_sel, ae, robot_chars, motion_type, pkl_path, file_path,
                             animate=params['animate'], repr_kind=repr_kind,
                             eval=params['eval'], jnt_penalty=params['jnt_penalty'],
                             jnt_coeff=params['jnt_coeff'])
        else:
            y_sel = evaluate_raw(x_sel, robot_chars, motion_type, pkl_path, file_path,
                                 animate=params['animate'], repr_kind=repr_kind,
                                 eval=params['eval'], jnt_penalty=params['jnt_penalty'],
                                 jnt_coeff=params['jnt_coeff'])
        x_voo = np.concatenate((x_voo, x_sel), axis=0)
        y_voo = np.concatenate((y_voo, y_sel), axis=0)

    X = np.asarray(x_voo, float)
    Y = np.asarray(y_voo, float).reshape(-1)
    if X.shape[0] != Y.size:
        X = X[:Y.size]
    return X.copy(), Y.copy()


def run_once_all(model_spec, rng, file_path_list, repr_kind, robot_chars,
                 motion_type='cmu', pkl_path=None, params=None):
    """Motion-set VOO run (paper Table I Motion Set column). Returns (X, Y) trajectory."""
    if params is None:
        params = DEFAULT_VOO_PARAMS
    if model_spec is not None:
        ae, _ = load_saved_model(model_name=model_spec, device='cpu')
        x_dim = ae.z_dim
    else:
        ae = None
        x_dim = 120 if repr_kind == 'screw' else 105

    rand = rng.random
    x_voo = params['x_minval'] + (params['x_maxval'] - params['x_minval']) * rand((params['n_init'], x_dim))
    y_voo = evaluate_all_motions(
        x_voo, ae, robot_chars, motion_type, pkl_path, file_path_list,
        animate=params['animate'], repr_kind=repr_kind, eval=params['eval'],
        jnt_penalty=params['jnt_penalty'], jnt_coeff=params['jnt_coeff'])

    for _ in range(params['max_iter']):
        x_sel = _voo_step(rng, x_voo, y_voo, x_dim, params)
        y_sel = evaluate_all_motions(
            x_sel, ae, robot_chars, motion_type, pkl_path, file_path_list,
            animate=params['animate'], repr_kind=repr_kind, eval=params['eval'],
            jnt_penalty=params['jnt_penalty'], jnt_coeff=params['jnt_coeff'])
        x_voo = np.concatenate((x_voo, x_sel), axis=0)
        y_voo = np.concatenate((y_voo, y_sel), axis=0)

    X = np.asarray(x_voo, float)
    Y = np.asarray(y_voo, float).reshape(-1)
    if X.shape[0] != Y.size:
        X = X[:Y.size]
    return X.copy(), Y.copy()


# =============================================================================
# Public API summary
# =============================================================================

__all__ = [
    # constants / factories
    'DEFAULT_VOO_PARAMS', 'get_rngs',
    # PA-MPJPE
    '_similarity_transform_umeyama', '_sequence_pa_mpjpe', '_sequence_mpjpe',
    # retargeting
    'sum_without_negatives', 'get_smpl_pos', 'get_cmu_pos',
    'r_sq', 'retarget_and_procrustes',
    # objectives
    'override_joi', 'evaluate', 'evaluate_all_motions', 'evaluate_raw',
    # VOO loops
    'run_once', 'run_once_all',
]
