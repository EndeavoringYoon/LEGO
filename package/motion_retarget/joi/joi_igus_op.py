import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_igus_op():
    """
        Get the body name of JOI of igus_op
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'base_link',
        'neck':'head_link',
        'rs':'right_upper_arm_link',
        're':'right_lower_arm_link',
        'rw':'virtual_right_wrist',
        'rh':'virtual_right_wrist',
        'ls':'left_upper_arm_link',
        'le':'left_lower_arm_link',
        'lw':'virtual_left_wrist_link',
        'lh':'virtual_left_wrist_link',
        'rp':'right_hip_roll_link',
        'rk':'right_shank_link',
        'ra':'right_ankle_link',
        'rf':'right_foot_link',
        'lp':'left_hip_roll_link',
        'lk':'left_shank_link',
        'la':'left_ankle_link',
        'lf':'left_foot_link',
    }
    return joi_body_name

def get_T_joi_of_igus_op(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = True,
    ):
    """
        Get T of JOI of igus_op
    """
    joi_body_name = get_joi_body_name_of_igus_op()
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
