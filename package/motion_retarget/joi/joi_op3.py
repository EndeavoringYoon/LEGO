import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_op3():
    """
        Get the body name of JOI of op3
    """
    joi_body_name = {
        'root':'body_link',
        'base':'body_link',
        'spine':'body_link',
        'neck':'head_pan_link',
        'rs':'r_sho_roll_link',
        're':'r_el_link',
        'rw':'virtual_r_wr_link',
        'rh':'virtual_r_wr_link',
        'ls':'l_sho_roll_link',
        'le':'l_el_link',
        'lw':'virtual_l_wr_link',
        'lh':'virtual_l_wr_link',
        'rp':'r_hip_pitch_link',
        'rk':'r_knee_link',
        'ra':'r_ank_pitch_link',
        'rf':'r_ank_pitch_link',
        'lp':'l_hip_pitch_link',
        'lk':'l_knee_link',
        'la':'l_ank_pitch_link',
        'lf':'l_ank_pitch_link',
    }
    return joi_body_name

def get_T_joi_of_op3(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of op3
    """
    joi_body_name = get_joi_body_name_of_op3()
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
