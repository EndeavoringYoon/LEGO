import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_iCub():
    """
        Get the body name of JOI of iCub
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'chest',
        'neck':'head',
        'rs':'r_shoulder_3',
        're':'r_elbow_1',
        'rw':'r_wrist_1',
        'rh':'r_hand',
        'ls':'l_shoulder_3',
        'le':'l_elbow_1',
        'lw':'l_wrist_1',
        'lh':'l_hand',
        'rp':'r_hip_2',
        'rk':'r_lower_leg',
        'ra':'r_ankle_2',
        'rf':'r_ankle_2',
        'lp':'l_hip_2',
        'lk':'l_lower_leg',
        'la':'l_ankle_2',
        'lf':'l_ankle_2',
    }
    return joi_body_name

def get_T_joi_of_iCub(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = True
    ):
    """
        Get T of JOI of iCub
    """
    joi_body_name = get_joi_body_name_of_iCub()
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
