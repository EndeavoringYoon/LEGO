import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_rby1():
    """
        Get the body name of JOI of rby1
    """
    joi_body_name = {
        'root':'base',
        'base':'base',
        'spine':'link_torso_3',
        'neck':'link_head_1',
        'rs':'link_right_arm_1',
        're':'link_right_arm_3',
        'rw':'link_right_arm_5',
        'rh':'EE_BODY_R',
        'ls':'link_left_arm_1',
        'le':'link_left_arm_3',
        'lw':'link_left_arm_5',
        'lh':'EE_BODY_L',
    }
    return joi_body_name

def get_T_joi_of_rby1(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of rby1
    """
    joi_body_name = get_joi_body_name_of_rby1()
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
