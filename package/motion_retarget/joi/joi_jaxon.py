import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_jaxon():
    """
        Get the body name of JOI of jaxon
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'CHEST_LINK1',
        'neck':'HEAD_LINK1',
        'rs':'RARM_LINK2',
        're':'RARM_LINK4',
        'rw':'RARM_LINK7',
        'rh':'RARM_LINK7',
        'ls':'LARM_LINK2',
        'le':'LARM_LINK4',
        'lw':'LARM_LINK7',
        'lh':'LARM_LINK7',
        'rp':'RLEG_LINK2',
        'rk':'RLEG_LINK3',
        'ra':'RLEG_LINK5',
        'rf':'RLEG_LINK5',
        'lp':'LLEG_LINK2',
        'lk':'LLEG_LINK3',
        'la':'LLEG_LINK5',
        'lf':'LLEG_LINK5',
    }
    return joi_body_name

def get_T_joi_of_jaxon(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = True
    ):
    """
        Get T of JOI of jaxon
    """
    joi_body_name = get_joi_body_name_of_jaxon()
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
