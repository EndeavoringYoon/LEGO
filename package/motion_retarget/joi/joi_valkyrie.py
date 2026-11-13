import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_valkyrie():
    """
        Get the body name of JOI of valkyrie
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'torso',
        'neck':'lowerNeckPitchLink',
        'rs':'rightShoulderYawLink',
        're':'rightForearmLink',
        'rw':'rightWristRollLink',
        'rh':'rightPalm',
        'ls':'leftShoulderYawLink',
        'le':'leftForearmLink',
        'lw':'leftWristRollLink',
        'lh':'leftPalm',
        'rp':'rightHipPitchLink',
        'rk':'rightKneePitchLink',
        'ra':'rightAnklePitchLink',
        'rf':'rightFoot',
        'lp':'leftHipPitchLink',
        'lk':'leftKneePitchLink',
        'la':'leftAnklePitchLink',
        'lf':'leftFoot',
    }
    return joi_body_name

def get_T_joi_of_valkyrie(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of valkyrie
    """
    joi_body_name = get_joi_body_name_of_valkyrie()
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
