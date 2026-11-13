import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_apollo():
    """
        Get the body name of JOI of apollo
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'torso_pitch_link',
        'neck':'neck_roll_link',
        'rs':'r_shoulder_fe_link',
        're':'r_elbow_fe_link',
        'rw':'r_wrist_yaw_link',
        'rh':'r_wrist_pitch_link',
        'ls':'l_shoulder_fe_link',
        'le':'l_elbow_fe_link',
        'lw':'l_wrist_yaw_link',
        'lh':'l_wrist_pitch_link',
        'rp':'r_hip_fe_link',
        'rk':'r_knee_fe_link',
        'ra':'r_ankle_ie_link',
        'rf':'r_foot_link',
        'lp':'l_hip_fe_link',
        'lk':'l_knee_fe_link',
        'la':'l_ankle_ie_link',
        'lf':'l_foot_link',
    }
    return joi_body_name

def get_T_joi_of_apollo(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = True,
    ):
    """
        Get T of JOI of apollo
    """
    joi_body_name = get_joi_body_name_of_apollo()
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
