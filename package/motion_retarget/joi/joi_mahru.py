import numpy as np
"""
    Following should be called:
        sys.path.append('../../package/helper/')
"""
from transforms import pr2t,t2p,t2r

def get_joi_body_name_of_mahru():
    """
        Get the body name of JOI of KIST Mahru
    """
    joi_body_name = {
        'root':'base_link',
        'base':'base_link',
        'spine':'TOP',
        'rs':'RA2',
        're':'RA4',
        'rw':'RH',
        'rh':'RH',
        'ls':'LA2',
        'le':'LA4',
        'lw':'LH',
        'lh':'LH',
        'rp':'RL2',
        'rk':'RL3',
        'ra':'RW',
        'rf':'RW',
        'lp':'LL2',
        'lk':'LL3',
        'la':'LW',
        'lf':'LW'
    }
    return joi_body_name

def get_T_joi_of_mahru(
        env,
        neck_between_shoulder = True,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of KIST Mahru
    """
    joi_body_name = get_joi_body_name_of_mahru()
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
