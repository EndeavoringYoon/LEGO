import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_poppy():
    """
        Get the body name of JOI of poppy
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'spine',
        'neck':'neck',
        'rs':'r_shoulder_motor',
        're':'r_forearm',
        'rw':'r_virtual_wrist',
        'rh':'r_virtual_wrist',
        'ls':'l_shoulder_motor',
        'le':'l_forearm',
        'lw':'l_virtual_wrist',
        'lh':'l_virtual_wrist',
        'rp':'r_thigh',
        'rk':'r_shin',
        'ra':'r_foot',
        'rf':'r_foot',
        'lp':'l_thigh',
        'lk':'l_shin',
        'la':'l_foot',
        'lf':'l_foot',
    }
    return joi_body_name

def get_T_joi_of_poppy(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of poppy
    """
    joi_body_name = get_joi_body_name_of_poppy()
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
