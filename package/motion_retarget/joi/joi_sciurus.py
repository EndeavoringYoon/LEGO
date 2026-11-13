import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_sciurus():
    """
        Get the body name of JOI of sciurus
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'body_link',
        'neck':'neck_pitch_link',
        'rs':'r_link2',
        're':'r_link4',
        'rw':'r_link5',
        'rh':'r_link7',
        'ls':'l_link2',
        'le':'l_link4',
        'lw':'l_link5',
        'lh':'l_link7',
    }
    return joi_body_name

def get_T_joi_of_sciurus(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of sciurus
    """
    joi_body_name = get_joi_body_name_of_sciurus()
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
