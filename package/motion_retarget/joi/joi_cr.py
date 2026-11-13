import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/kinematics_helper/') # for 'transforms'
"""
from transforms import *

def get_joi_body_name_of_cr():
    """
        Get the body name of JOI of Unitree G1
    """
    joi_body_name = {
        'root':'base',
        'base':'base',
        'spine':'spine',
        'neck':'neck',
        'rs':'right_shoulder',
        're':'right_elbow',
        'rw':'right_wrist',
        'rh':'rmiddle_l0',
        'ls':'left_shoulder',
        'le':'left_elbow',
        'lw':'left_wrist',
        'lh':'lmiddle_l0',
        'rp':'right_pelvis',
        'rk':'right_knee',
        'ra':'right_ankle',
        'rf':'right_foot',
        'lp':'left_pelvis',
        'lk':'left_knee',
        'la':'left_ankle',
        'lf':'left_foot',
    }
    return joi_body_name

def get_T_joi_of_cr(
        env,
        neck_between_shoulder = True,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of G1
    """
    joi_body_name = get_joi_body_name_of_cr()
    T_joi = {}
    for key in joi_body_name.keys():
        T_joi[key] = env.get_T_body(body_name=joi_body_name[key])
    # Neck 
    if neck_between_shoulder:
        T_joi['neck'] = pr2t(0.5*(t2p(T_joi['rs'])+t2p(T_joi['ls'])),np.eye(3))
    # Base
    if base_between_pelvis:
        T_joi['base'] = pr2t(0.5*(t2p(T_joi['rp'])+t2p(T_joi['lp'])),t2r(T_joi['base']))
    return T_joi