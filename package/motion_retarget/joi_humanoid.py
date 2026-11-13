import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/kinematics_helper/') # for 'transforms'
"""
from transforms import *

def get_joi_body_name_of_humanoid():
    """
        Get the body name of JOI of Humanoid
    """
    joi_body_name = {
        'root':'torso',
        'base':'pelvis',
        'spine':'waist_lower',
        'neck':'torso',
        'rs':'upper_arm_right',
        're':'lower_arm_right',
        'rw':'hand_right',
        'rh':'hand_right',
        'ls':'upper_arm_left',
        'le':'lower_arm_left',
        'lw':'hand_left',
        'lh':'hand_left',
        'rp':'thigh_right',
        'rk':'shin_right',
        'ra':'foot_right',
        'rf':'foot_right',
        'lp':'thigh_left',
        'lk':'shin_left',
        'la':'foot_left',
        'lf':'foot_left',
    }
    return joi_body_name

def get_T_joi_of_humanoid(
        env,
        neck_between_shoulder = True,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of G1
    """
    joi_body_name = get_joi_body_name_of_humanoid()
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