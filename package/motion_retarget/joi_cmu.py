import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/kinematics_helper/') # for 'transforms'
"""
from transforms import *

def get_T_joi_of_chain_cmu(
        chain,
        base_between_pelvis = True,
    ):
    """ 
        Get joints of interest of CMU mocap chain
    """
    p_base,R_base = chain.get_joint_pR(joint_name='Hips')
    p_spine,R_spine = chain.get_joint_pR(joint_name='Spine')
    p_rs,R_rs = chain.get_joint_pR(joint_name='RightArm')
    p_re,R_re = chain.get_joint_pR(joint_name='RightForeArm')
    p_rw,R_rw = chain.get_joint_pR(joint_name='RightHand')
    p_rh,R_rh = chain.get_joint_pR(joint_name='RightHand')
    p_ls,R_ls = chain.get_joint_pR(joint_name='LeftArm')
    p_le,R_le = chain.get_joint_pR(joint_name='LeftForeArm')
    p_lw,R_lw = chain.get_joint_pR(joint_name='LeftHand')
    p_lh,R_lh = chain.get_joint_pR(joint_name='LeftHand')
    p_neck,R_neck = chain.get_joint_pR(joint_name='Neck')
    # z neck position to be the center of two shoulder positions
    p_neck_z = 0.5*(p_rs+p_ls) 
    p_neck = np.array([p_neck[0],p_neck[1],p_neck_z[2]])
    p_head,R_head = chain.get_joint_pR(joint_name='Head')
    p_rp,R_rp = chain.get_joint_pR(joint_name='RightUpLeg')
    p_rk,R_rk = chain.get_joint_pR(joint_name='RightLeg')
    p_ra,R_ra = chain.get_joint_pR(joint_name='RightFoot')
    p_rf,R_rf = chain.get_joint_pR(joint_name='RightFoot')
    p_lp,R_lp = chain.get_joint_pR(joint_name='LeftUpLeg')
    p_lk,R_lk = chain.get_joint_pR(joint_name='LeftLeg')
    p_la,R_la = chain.get_joint_pR(joint_name='LeftFoot')
    p_lf,R_lf = chain.get_joint_pR(joint_name='LeftFoot')
    p_r_toe,R_r_toe = chain.get_joint_pR(joint_name='RightToeBase')
    p_l_toe,R_l_toe = chain.get_joint_pR(joint_name='LeftToeBase')
    # Modify base position
    if base_between_pelvis: p_base = 0.5*(p_rp+p_lp)
    # Return
    T_joi = {
        'base': pr2t(p_base,R_base),
        'spine': pr2t(p_spine,R_spine),
        'rs': pr2t(p_rs,R_rs),
        're': pr2t(p_re,R_re),
        'rw': pr2t(p_rw,R_rw),
        'rh': pr2t(p_rh,R_rh),
        'ls': pr2t(p_ls,R_ls),
        'le': pr2t(p_le,R_le),
        'lw': pr2t(p_lw,R_lw),
        'lh': pr2t(p_lh,R_lh),
        'neck':pr2t(p_neck,R_neck),
        'head':pr2t(p_head,R_head),
        'rp': pr2t(p_rp,R_rp),
        'rk': pr2t(p_rk,R_rk),
        'ra': pr2t(p_ra,R_ra),
        'rf': pr2t(p_rf,R_rf),
        'rtoe': pr2t(p_r_toe,R_r_toe),
        'lp': pr2t(p_lp,R_lp),
        'lk': pr2t(p_lk,R_lk),
        'la': pr2t(p_la,R_la),
        'lf': pr2t(p_lf,R_lf),
        'ltoe': pr2t(p_l_toe,R_l_toe),
    }
    return T_joi