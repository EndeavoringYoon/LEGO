import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_hrp4():
    """
        Get the body name of JOI of hrp4
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'CHEST_P_LINK',
        'neck':'NECK_Y_LINK',
        'rs':'R_SHOULDER_R_LINK',
        're':'R_ELBOW_P_LINK',
        'rw':'R_WRIST_P_LINK',
        'rh':'r_wrist',
        'ls':'L_SHOULDER_R_LINK',
        'le':'L_ELBOW_P_LINK',
        'lw':'L_WRIST_P_LINK',
        'lh':'l_wrist',
        'rp':'R_HIP_P_LINK',
        'rk':'R_KNEE_P_LINK',
        'ra':'R_ANKLE_R_LINK',
        'rf':'R_ANKLE_R_LINK',
        'lp':'L_HIP_P_LINK',
        'lk':'L_KNEE_P_LINK',
        'la':'L_ANKLE_R_LINK',
        'lf':'L_ANKLE_R_LINK',
    }
    return joi_body_name

def get_T_joi_of_hrp4(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of hrp4
    """
    joi_body_name = get_joi_body_name_of_hrp4()
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
