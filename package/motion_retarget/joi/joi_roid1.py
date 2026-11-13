import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_roid1():
    """
        Get the body name of JOI of roid1
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'c_chest',
        'neck':'c_head',
        'rs':'r_upperarm',
        're':'r_elbow',
        'rw':'r_virtual_wrist',
        'rh':'r_virtual_wrist',
        'ls':'l_upperarm',
        'le':'l_elbow',
        'lw':'l_virtual_wrist',
        'lh':'l_virtual_wrist',
        'rp':'r_hipjoint_lower',
        'rk':'r_leg_lower',
        'ra':'r_ankle',
        'rf':'r_foot_large',
        'lp':'l_hipjoint_lower',
        'lk':'l_leg_lower',
        'la':'l_ankle',
        'lf':'l_foot_large',
    }
    return joi_body_name

def get_T_joi_of_roid1(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of roid1
    """
    joi_body_name = get_joi_body_name_of_roid1()
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
