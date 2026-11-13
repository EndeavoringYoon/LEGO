import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_bruce():
    """
        Get the body name of JOI of bruce
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'base_link',
        'neck':'shoulder_pitch_link_l',
        'rs':'shoulder_pitch_link_r',
        're':'elbow_pitch_link_r',
        'rw':'virtual_wrist_link_r',
        'rh':'virtual_wrist_link_r',
        'ls':'shoulder_pitch_link_l',
        'le':'elbow_pitch_link_l',
        'lw':'virtual_wrist_link_l',
        'lh':'virtual_wrist_link_l',
        'rp':'hip_pitch_link_r',
        'rk':'knee_pitch_link_r',
        'ra':'ankle_pitch_link_r',
        'rf':'ankle_pitch_link_r',
        'lp':'hip_pitch_link_l',
        'lk':'knee_pitch_link_l',
        'la':'ankle_pitch_link_l',
        'lf':'ankle_pitch_link_l',
    }
    return joi_body_name

def get_T_joi_of_bruce(
        env,
        neck_between_shoulder = True,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of bruce
    """
    joi_body_name = get_joi_body_name_of_bruce()
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
