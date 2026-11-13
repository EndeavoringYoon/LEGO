import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_pepper():
    """
        Get the body name of JOI of pepper
    """
    joi_body_name = {
        'root':'Tibia',
        'base':'Pelvis',
        'spine':'Hip',
        'neck':'Neck',
        'rs':'RBicep',
        're':'RForeArm',
        'rw':'r_wrist',
        'rh':'r_wrist',
        'ls':'LBicep',
        'le':'LForeArm',
        'lw':'l_wrist',
        'lh':'l_wrist',
    }
    return joi_body_name

def get_T_joi_of_pepper(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of pepper
    """
    joi_body_name = get_joi_body_name_of_pepper()
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
