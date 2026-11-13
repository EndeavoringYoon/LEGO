import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_lego_hard():
    """
        Get the body name of JOI of Lego V2
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'torso_link',
        'rs':'right_shoulder_link',
        're':'right_elbow_pitch_link',
        'rw':'right_wrist_yaw_link',
        'rh':'right_wrist_yaw_link',
        'ls':'left_shoulder_link',
        'le':'left_elbow_pitch_link',
        'lw':'left_wrist_yaw_link',
        'lh':'left_wrist_yaw_link',
        'rp':'right_thigh_link',
        'rk':'right_calf_link',
        'ra':'right_foot_link',
        'rf':'right_foot_link',
        'lp':'left_thigh_link',
        'lk':'left_calf_link',
        'la':'left_foot_link',
        'lf':'left_foot_link'
    }
    return joi_body_name

def get_T_joi_of_lego_hard(
        env,
        neck_between_shoulder = True,
        base_between_pelvis   = True,
    ):
    """
        Get T of JOI of Lego V2
    """
    joi_body_name = get_joi_body_name_of_lego_hard()
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
