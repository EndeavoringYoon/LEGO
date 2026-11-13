import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_talos():
    """
        Get the body name of JOI of talos
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'torso_2_link',
        'neck':'head_2_link',
        'rs':'arm_right_3_link',
        're':'arm_right_4_link',
        'rw':'arm_right_7_link',
        'rh':'gripper_right_motor_double_link',
        'ls':'arm_left_3_link',
        'le':'arm_left_4_link',
        'lw':'arm_left_7_link',
        'lh':'gripper_left_motor_double_link',
        'rp':'leg_right_3_link',
        'rk':'leg_right_4_link',
        'ra':'leg_right_6_link',
        'rf':'leg_right_6_link',
        'lp':'leg_left_3_link',
        'lk':'leg_left_4_link',
        'la':'leg_left_6_link',
        'lf':'leg_left_6_link',
    }
    return joi_body_name

def get_T_joi_of_talos(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of talos
    """
    joi_body_name = get_joi_body_name_of_talos()
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
