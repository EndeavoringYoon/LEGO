import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_x1():
    """
        Get the body name of JOI of x1
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'lumber_pitch',
        'neck':'left_shoulder_pitch',
        'rs':'right_shoulder_roll',
        're':'right_elbow_pitch',
        'rw':'right_wrist_pitch',
        'rh':'right_wrist_pitch',
        'ls':'left_shoulder_roll',
        'le':'left_elbow_pitch',
        'lw':'left_wrist_pitch',
        'lh':'left_wrist_pitch',
        'rp':'right_hip_roll',
        'rk':'right_knee_pitch',
        'ra':'right_ankle_pitch',
        'rf':'right_ankle_roll',
        'lp':'left_hip_roll',
        'lk':'left_knee_pitch',
        'la':'left_ankle_pitch',
        'lf':'left_ankle_roll',
    }
    return joi_body_name

def get_T_joi_of_x1(
        env,
        neck_between_shoulder = True,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of x1
    """
    joi_body_name = get_joi_body_name_of_x1()
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
