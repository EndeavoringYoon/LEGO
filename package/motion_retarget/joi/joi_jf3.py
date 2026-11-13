import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_jf3():
    """
        Get the body name of JOI of JF3
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'base_link',
        'head':'head',
        'neck':'head',
        'rs':'link2_1',
        're':'link4_1',
        'rw':'hand_1',
        'rh':'hand_1',
        'ls':'link2_2',
        'le':'link4_2',
        'lw':'hand_2',
        'lh':'hand_2',
    }
    return joi_body_name

def get_T_joi_of_jf3(
        env,
        neck_between_shoulder = True,
        base_between_pelvis   = False,
        spine_between_neck     = True,
    ):
    """
        Get T of JOI of JF3
    """
    joi_body_name = get_joi_body_name_of_jf3()
    T_joi = {}
    for key in joi_body_name.keys():
        T_joi[key] = env.get_T_body(body_name=joi_body_name[key])
    # Neck 
    if neck_between_shoulder:
        T_joi['neck'] = pr2t(0.5*(t2p(T_joi['rs'])+t2p(T_joi['ls'])),np.eye(3))
    # Base
    if base_between_pelvis:
        T_joi['base'] = pr2t(0.5*(t2p(T_joi['rp'])+t2p(T_joi['lp'])),t2r(T_joi['base']))
    # Spine
    if spine_between_neck:
        T_joi['spine'] = pr2t(0.5*t2p(T_joi['neck']),t2r(T_joi['base']))
    return T_joi
