import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_armar4():
    """
        Get the body name of JOI of armar4
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'torso_yaw',
        'neck':'neck_1',
        'rs':'armr_sho3',
        're':'armr_elb1',
        'rw':'armr_wri1',
        'rh':'armr_wri2',
        'ls':'arml_sho3',
        'le':'arml_elb1',
        'lw':'arml_wri1',
        'lh':'arml_wri2',
        'rp':'legr_hip2',
        'rk':'legr_knee',
        'ra':'legr_ank1',
        'rf':'legr_heel',
        'lp':'legl_hip2',
        'lk':'legl_knee',
        'la':'legl_ank1',
        'lf':'legl_heel',
    }
    return joi_body_name

def get_T_joi_of_armar4(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of armar4
    """
    joi_body_name = get_joi_body_name_of_armar4()
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
