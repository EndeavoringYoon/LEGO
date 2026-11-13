import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_jvrc():
    """
        Get the body name of JOI of jvrc
    """
    joi_body_name = {
        'root':'PELVIS_S',
        'base':'PELVIS_S',
        'spine':'WAIST_R_S',
        'neck':'NECK_P_S',
        'rs':'R_SHOULDER_Y_S',
        're':'R_ELBOW_P_S',
        'rw':'R_WRIST_Y_S',
        'rh':'R_WRIST_Y_S',
        'ls':'L_SHOULDER_Y_S',
        'le':'L_ELBOW_P_S',
        'lw':'L_WRIST_Y_S',
        'lh':'L_WRIST_Y_S',
        'rp':'R_HIP_Y_S',
        'rk':'R_KNEE_S',
        'ra':'R_ANKLE_P_S',
        'rf':'R_ANKLE_P_S',
        'lp':'L_HIP_Y_S',
        'lk':'L_KNEE_S',
        'la':'L_ANKLE_P_S',
        'lf':'L_ANKLE_P_S',
    }
    return joi_body_name

def get_T_joi_of_jvrc(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of jvrc
    """
    joi_body_name = get_joi_body_name_of_jvrc()
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
