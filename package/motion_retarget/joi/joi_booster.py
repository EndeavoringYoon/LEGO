import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_booster():
    """
        Get the body name of JOI of booster
    """
    joi_body_name = {
        'root':'Trunk',
        'base':'Trunk',
        'spine':'Waist',
        'neck':'H1',
        'rs':'AR2',
        're':'AR3',
        'rw':'virtual_wrist_link_r',
        'rh':'virtual_wrist_link_r',
        'ls':'AL2',
        'le':'AL3',
        'lw':'virtual_wrist_link_l',
        'lh':'virtual_wrist_link_l',
        'rp':'Hip_Roll_Right',
        'rk':'Shank_Right',
        'ra':'Ankle_Cross_Right',
        'rf':'right_foot_link',
        'lp':'Hip_Roll_Left',
        'lk':'Shank_Left',
        'la':'Ankle_Cross_Left',
        'lf':'left_foot_link',
    }
    return joi_body_name

def get_T_joi_of_booster(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of booster
    """
    joi_body_name = get_joi_body_name_of_booster()
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
