import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_ergocub():
    """
        Get the body name of JOI of ergocub
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'torso_2',
        'neck':'neck_2',
        'rs':'r_shoulder_2',
        're':'r_forearm',
        'rw':'r_wrist_1',
        'rh':'r_hand_index_1',
        'ls':'l_shoulder_2',
        'le':'l_forearm',
        'lw':'l_wrist_1',
        'lh':'l_hand_index_1',
        'rp':'r_hip_2',
        'rk':'r_lower_leg',
        'ra':'r_ankle_1',
        'rf':'r_ankle_2',
        'lp':'l_hip_2',
        'lk':'l_lower_leg',
        'la':'l_ankle_1',
        'lf':'l_ankle_2',
    }
    return joi_body_name

def get_T_joi_of_ergocub(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of ergocub
    """
    joi_body_name = get_joi_body_name_of_ergocub()
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
