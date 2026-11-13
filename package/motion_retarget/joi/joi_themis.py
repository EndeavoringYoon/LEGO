import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_themis():
    """
        Get the body name of JOI of themis
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'base_link',
        'neck':'UPPERSHOULDER_L',
        'rs':'UPPERSHOULDER_R',
        're':'FOREARM_R',
        'rw':'VIRTUALWRIST_R',
        'rh':'VIRTUALWRIST_R',
        'ls':'UPPERSHOULDER_L',
        'le':'FOREARM_L',
        'lw':'VIRTUALWRIST_L',
        'lh':'VIRTUALWRIST_L',
        'rp':'FEMUR_R',
        'rk':'TIBIA_R',
        'ra':'ANKLE_R',
        'rf':'FOOT_R',
        'lp':'FEMUR_L',
        'lk':'TIBIA_L',
        'la':'ANKLE_L',
        'lf':'FOOT_L',
    }
    return joi_body_name

def get_T_joi_of_themis(
        env,
        neck_between_shoulder = True,
        base_between_pelvis   = True,
    ):
    """
        Get T of JOI of themis
    """
    joi_body_name = get_joi_body_name_of_themis()
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
