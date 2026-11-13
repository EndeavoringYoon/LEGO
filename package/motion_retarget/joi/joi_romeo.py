import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_romeo():
    """
        Get the body name of JOI of romeo
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'torso',
        'neck':'NeckYawLink',
        'rs':'RShoulderYawLink',
        're':'RElbowYawLink',
        'rw':'RWristYawLink',
        'rh':'r_wrist',
        'ls':'LShoulderYawLink',
        'le':'LElbowYawLink',
        'lw':'LWristYawLink',
        'lh':'l_wrist',
        'rp':'RHipPitchLink',
        'rk':'RKneePitchLink',
        'ra':'RAnklePitchLink',
        'rf':'r_ankle',
        'lp':'LHipPitchLink',
        'lk':'LKneePitchLink',
        'la':'LAnklePitchLink',
        'lf':'l_ankle',
    }
    return joi_body_name

def get_T_joi_of_romeo(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = True
    ):
    """
        Get T of JOI of romeo
    """
    joi_body_name = get_joi_body_name_of_romeo()
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
