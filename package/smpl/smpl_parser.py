from math import radians, cos, sin
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy

"""
    Call the following line on the main.py
    sys.path.append("../helper")
"""
from kinematic_chain import KinematicChainClass
from transforms import (
    r2rpy,
    rpy2r,
    rpy2r_order,
    t2p,
    t2r,
    pr2t,
    t2pr,
    T_yuzf2zuxf,
    R_yuzf2zuxf,
)
from scipy.spatial.transform import Rotation as R
from grp_ import *

def rs2rpys(Rs,unit='rad'):
    rpys = []
    for R in Rs:
        rpys.append(r2rpy(R))
    rpys = np.array(rpys)
    return rpys

def sixd_to_matrix(d6):
    """ 
        6D rotation (two rows, Gram-Schmidt; same as pytorch3d rotation_6d_to_matrix) to rotation matrix
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)

def get_thetas_3d(data):
    """ 
        Joint rotation matrices (24, L, 3, 3). MDM's *_smpl_params.npy only has the 6D 'thetas' (24, 6, L)
    """
    if 'thetas_3d' in data:
        return np.asarray(data['thetas_3d'])
    return sixd_to_matrix(np.asarray(data['thetas']).transpose(0, 2, 1))

def get_chains_from_smpl(
        file_path,
        smpl,
        p_rate           = 1.0, # positional scale
        HZ               = 20,
        intp_traj        = True,
        intp_HZ          = 120,
        plot_chain_graph = True,
        verbose          = True,
):
    """ 
        Get chains from SMPL
    """
    data = np.load(file_path, allow_pickle=True).item()
    beta = np.zeros((10,)) # (10,)

    if intp_traj:
        # Load original data
        thetas_org = get_thetas_3d(data)
        thetas_org = R.from_matrix(thetas_org.reshape(-1,3,3)).as_euler('XYZ').reshape(thetas_org.shape[0], thetas_org.shape[1], 3) # rs2rpys(thetas.reshape(-1,3,3)).reshape(thetas.shape[0], thetas.shape[1], 3)
        thetas_org = thetas_org.transpose(1,0,2)
        thetas_org = thetas_org.reshape(thetas_org.shape[0], thetas_org.shape[1]*thetas_org.shape[2])
        trans_org = data['root_translation'].transpose(1,0) # (L, 3)
        L_org=trans_org.shape[0] # (L)
        dt_org = 1/HZ
        dt_intp = 1/intp_HZ

        # GRP Interpolate
        GRP_posterior = GaussianRandomPathClass(name='GRP Posterior')
        t_anchor = np.arange(0,dt_org*L_org,dt_org)
        t_test = np.arange(0,dt_org*L_org,dt_intp)
        ss_range = 40*np.pi/180*np.concatenate((-np.ones((thetas_org.shape[1],1)),np.ones((thetas_org.shape[1],1))),axis=1)
        ss_range = np.concatenate((ss_range, np.inf*np.concatenate((-np.ones((trans_org.shape[1],1)),np.ones((trans_org.shape[1],1))),axis=1)))
        x_anchor = np.concatenate((thetas_org[:120,:], trans_org),axis=1)
        traj_joints_intp = GRP_posterior.interpolate(
            t_anchor=t_anchor.reshape(-1,1),x_anchor=x_anchor,t_test=t_test.reshape(-1,1),hyp={'g':1,'l':1/4,'w':1e-8},
            APPLY_EPSRU=True,t_eps=0.01,
            x_min=ss_range[:,0],x_max=ss_range[:,1],margin=3.0*np.pi/180.0)
        
        trans = traj_joints_intp[:,-3:]
        thetas = traj_joints_intp[:,:-3]
        pose = thetas
        dt = 1/intp_HZ
        L = trans.shape[0]
        secs = np.arange(0,dt*L,dt)

    else:
        thetas = get_thetas_3d(data) # Rotation Matrix
        thetas = R.from_matrix(thetas.reshape(-1,3,3)).as_euler('XYZ').reshape(thetas.shape[0], thetas.shape[1], 3) # rs2rpys(thetas.reshape(-1,3,3)).reshape(thetas.shape[0], thetas.shape[1], 3)
        thetas = thetas.transpose(1,0,2)
        thetas = thetas.reshape(thetas.shape[0], thetas.shape[1]*thetas.shape[2])
        pose = thetas # (L, 72)

        trans = data['root_translation'].transpose(1,0) # (L, 3)
        dt = 1/HZ
        L=trans.shape[0] # (L)
        secs = np.arange(0,dt*L,dt)

    chain = KinematicChainClass(name='SMPL')
    
    # Get skeleton offset by assigning zeros parameter
    smpl.set_params(
        beta     = beta,
        pose     = np.zeros(72,),
        trans    = np.zeros(3,),
        T_offset = pr2t(p=np.array([0,0,1.25]),R=rpy2r(np.radians([0,0,0]))))
    
    for idx in range(smpl.n_joint):
        global_R = t2r(smpl.T_joints[idx,:])
        local_R = global_R if idx == 0 else t2r(smpl.T_joints[smpl.parent[idx],:].T @ smpl.T_joints[idx,:])
        chain.add_joint(
            name = smpl.joint_names[idx], 
            parent_name = None if idx == 0 else smpl.joint_names[smpl.parent[idx]],
            p_offset = (t2p(smpl.T_joints[idx]) - (np.array([0,0,0]) if idx == 0 else (t2p(smpl.T_joints[smpl.parent[idx]]))))*p_rate,
            R_offset = local_R # np.eye(3,3)
        )


    # Plot chain graph
    if plot_chain_graph:
        chain.plot_chain_graph(
            align='vertical',
            figsize=(5,2),
            node_size=200,
            font_size_node=8,
            node_colors=None,
            font_size_title=10,
            ROOT_ON_TOP=True,
        )
        
    chains = []
    
    for tick in range(L):
        smpl.set_params(
        beta     = beta,
        pose     = pose[tick],
        trans    = trans[tick],
        T_offset = pr2t(p=np.array([0,0,1.25]),R=rpy2r(np.radians([0,0,0]))))

        p_root = p_rate*trans[tick]
        R_offset_list = []
        for joint_idx in range(smpl.n_joint):
            if joint_idx == 0: # root 
                rpy_deg = np.array([0,0,0])
                rpy_deg_root = smpl.pose.reshape(-1,3)[joint_idx] #r2rpy(t2r(smpl.T_joints[joint_idx,:])) # backup root rot
            else:
                rpy_deg = smpl.pose.reshape(-1,3)[joint_idx]#[[2,0,1]]
                R_offset = R.from_euler('YZX',rpy_deg).as_matrix()
                R_offset_list.append(
                    R_offset
                )
        R_root = R.from_euler('YZX',rpy_deg_root).as_matrix()# rpy2r_order(np.radians(rpy_deg_root),order=rpy_order)
        chain.set_root_joint_R(R=R_root)
        chain.set_root_joint_p(p=p_root[[2,0,1]])
        chain.set_joint_R_offset_list(chain.joint_names[1:],R_offset_list)
        chain.forward_kinematics()
        # Append
        chains.append(deepcopy(chain)) # we have to 'deepcopy'


    return secs, chains