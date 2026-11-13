import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_baymax():
    """
        Get the body name of JOI of Baymax
    """
    joi_body_name = {
        'root':'base_link',
        'base':'waist_yaw',
        'spine':'waist_pitch',
        'neck':'neck_yaw_1',
        'rs':'right_shoulder_roll_1',
        're':'right_elbow_1',
        'rw':'right_hand_1',
        'rh':'right_hand_1',
        'ls':'left_shoulder_roll_1',
        'le':'left_elbow_1',
        'lw':'left_hand_1',
        'lh':'left_hand_1',
    }
    return joi_body_name

def get_T_joi_of_baymax(
        env,
        neck_between_shoulder = True,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of G1
    """
    joi_body_name = get_joi_body_name_of_baymax()
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
