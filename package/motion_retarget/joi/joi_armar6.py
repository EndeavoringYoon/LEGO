import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_armar6():
    """
        Get the body name of JOI of armar6
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'torso',
        'neck':'middle_neck',
        'rs':'arm_t34_r1',
        're':'arm_t56_r1',
        'rw':'arm_t78_r1',
        'rh':'hand_fixed_r1',
        'ls':'arm_t34_r0',
        'le':'arm_t56_r0',
        'lw':'arm_t78_r0',
        'lh':'hand_fixed_r0',
    }
    return joi_body_name

def get_T_joi_of_armar6(
        env,
        neck_between_shoulder = True,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of armar6
    """
    joi_body_name = get_joi_body_name_of_armar6()
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
