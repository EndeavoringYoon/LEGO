import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_auto_gen():
    """
        Get the body name of JOI of auto_gen
    """
    joi_body_name = {
        'root':'torso_1',
        'base':'torso_1',
        'spine':'torso_3',
        'neck':'torso_4',
        'rs':'R_shoulder_1',
        're':'R_elbow_1',
        'rw':'R_wrist_1',
        'rh':'R_wrist_2',
        'ls':'L_shoulder_1',
        'le':'L_elbow_1',
        'lw':'L_wrist_1',
        'lh':'L_wrist_2',
    }
    return joi_body_name

def get_T_joi_of_auto_gen(
        env,
        neck_between_shoulder = True,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of auto_gen
    """
    joi_body_name = get_joi_body_name_of_auto_gen()
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
