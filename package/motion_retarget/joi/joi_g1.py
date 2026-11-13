import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/kinematics_helper/') # for 'transforms'
"""
from transforms import *

def get_joi_body_name_of_g1():
    """
        Get the body name of JOI of Unitree G1
    """
    joi_body_name = {
        'root':'pelvis',
        'base':'pelvis',
        'spine':'pelvis',
        'rs':'right_shoulder_roll_link',
        're':'right_elbow_pitch_link',
        'rw':'virtual_right_wrist_link',
        'rh':'virtual_right_wrist_link',
        'ls':'left_shoulder_roll_link',
        'le':'left_elbow_pitch_link',
        'lw':'virtual_left_wrist_link',
        'lh':'virtual_left_wrist_link',
        'rp':'right_hip_roll_link',
        'rk':'right_knee_link',
        'ra':'right_ankle_pitch_link',
        'rf':'right_ankle_roll_link',
        'lp':'left_hip_roll_link',
        'lk':'left_knee_link',
        'la':'left_ankle_pitch_link',
        'lf':'left_ankle_roll_link'
    }
    return joi_body_name

def get_T_joi_of_g1(
        env,
        neck_between_shoulder = True,
        base_between_pelvis   = True,
    ):
    """
        Get T of JOI of G1
    """
    joi_body_name = get_joi_body_name_of_g1()
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
