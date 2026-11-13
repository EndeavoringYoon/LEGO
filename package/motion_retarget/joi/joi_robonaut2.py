import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_robonaut2():
    """
        Get the body name of JOI of robonaut2
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'waist_center',
        'neck':'neck_lower',
        'rs':'right_shoulder_pitch',
        're':'right_elbow',
        'rw':'right_wrist_pitch',
        'rh':'right_wrist_yaw',
        'ls':'left_shoulder_pitch',
        'le':'left_elbow',
        'lw':'left_wrist_pitch',
        'lh':'left_wrist_yaw',
        'rp':'right_hip_pitch',
        'rk':'right_knee',
        'ra':'right_ankle_roll',
        'rf':'right_ankle_roll',
        'lp':'left_hip_pitch',
        'lk':'left_knee',
        'la':'left_ankle_roll',
        'lf':'left_ankle_roll',
    }
    return joi_body_name

def get_T_joi_of_robonaut2(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of robonaut2
    """
    joi_body_name = get_joi_body_name_of_robonaut2()
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
