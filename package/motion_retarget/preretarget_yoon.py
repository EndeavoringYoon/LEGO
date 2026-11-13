import mujoco
import numpy as np
import copy
"""
    Following should be called:
        sys.path.append('../../package/helper/') # for 'utility' and 'transformation'
        sys.path.append('../../package/motion_retarget/') # for 'joi_cmu' and 'joi_g1'
        sys.path.append('../../package/mujoco_usage/') # for 'mujoco_parser'
"""
from utils import len_T_joi,uv_T_joi
from transforms import T_yuzf2zuxf,t2p,t2r,rpy2r,r2rpy
from .joi import *
from ik import init_ik_info,add_ik_info,get_dq_from_ik_info,plot_ik_info

def directional_preretarget(
        env,
        secs,
        chains,
        src_name           = 'cmu',
        trgt_name          = 'g1',
        joint_names_for_ik = None,
        max_ik_tick        = 100, # maximum IK iteration
        ik_stepsize        = 1, # IK stepsize
        ik_err_th          = 1e-3, # IK error threshold
        print_ik_failure   = False,
        ik_R_hand          = False,
        reset_env          = True,
        animate            = True,
        plot_joi_src       = False,
        plot_joi_trgt      = False,
        render_every       = 1,
        verbose            = True,
        print_every        = 10,
        pause_each_tick    = False,
    ):
    """ 
        Directional pre-retargeting (aka unit-vector method)
        Return: (qpos_dir_list)
    """
    
    # Reset env
    if reset_env:
        env.reset(step=True)
        
    # Initialize viewer
    if animate:
        env.init_viewer(
            title       = 'Directional Pre-Retargeting Process',
            transparent = True,
        )
    
    # Initialize IK configuration
    if joint_names_for_ik is None: 
        joint_names_for_ik = env.rev_joint_names 
    joint_idxs_jac = env.get_idxs_jac(joint_names=joint_names_for_ik)
    joint_with_base_idxs_jac = np.concatenate(([0,1,2,3,4,5],joint_idxs_jac))
    
    # Target JOI body name
    """
        Target-specific
    """
    if trgt_name == 'g1':
        joi_body_name_trgt = get_joi_body_name_of_g1()
        T_joi_trgt         = get_T_joi_of_g1(env)
        ik_p_spine         = False 
        ik_p_neck          = False
    elif trgt_name == 'cr':
        joi_body_name_trgt = get_joi_body_name_of_cr()
        T_joi_trgt         = get_T_joi_of_cr(env)
        ik_p_spine         = True 
        ik_p_neck          = True
    elif trgt_name == 'humanoid':
        joi_body_name_trgt = get_joi_body_name_of_humanoid()
        T_joi_trgt         = get_T_joi_of_humanoid(env)
        ik_p_spine         = False 
        ik_p_neck          = False
    elif trgt_name == 'baymax':
        joi_body_name_trgt = get_joi_body_name_of_baymax()
        T_joi_trgt         = get_T_joi_of_baymax(env)
        ik_p_spine         = False 
        ik_p_neck          = False
    elif trgt_name == 'jf3':
        joi_body_name_trgt = get_joi_body_name_of_jf3()
        T_joi_trgt         = get_T_joi_of_jf3(env)
        ik_p_spine         = False 
        ik_p_neck          = False
    elif trgt_name == 'mahru':
        joi_body_name_trgt = get_joi_body_name_of_mahru()
        T_joi_trgt         = get_T_joi_of_mahru(env)
        ik_p_spine         = False 
        ik_p_neck          = False
    
    if trgt_name == 'baymax' or trgt_name == 'jf3':
        # Get target lengths
        len_base2spine = len_T_joi(T_joi_trgt,'base','spine')
        len_spine2neck = len_T_joi(T_joi_trgt,'spine','neck')
        len_neck2rs    = len_T_joi(T_joi_trgt,'neck','rs')
        len_rs2re      = len_T_joi(T_joi_trgt,'rs','re')
        len_re2rw      = len_T_joi(T_joi_trgt,'re','rw')
        len_neck2ls    = len_T_joi(T_joi_trgt,'neck','ls')
        len_ls2le      = len_T_joi(T_joi_trgt,'ls','le')
        len_le2lw      = len_T_joi(T_joi_trgt,'le','lw')
    else:
        # Get target lengths
        len_base2rp    = len_T_joi(T_joi_trgt,'base','rp')
        len_rp2rk      = len_T_joi(T_joi_trgt,'rp','rk')
        len_rk2ra      = len_T_joi(T_joi_trgt,'rk','ra')
        len_base2lp    = len_T_joi(T_joi_trgt,'base','lp')
        len_lp2lk      = len_T_joi(T_joi_trgt,'lp','lk')
        len_lk2la      = len_T_joi(T_joi_trgt,'lk','la')
        len_base2spine = len_T_joi(T_joi_trgt,'base','spine')
        len_spine2neck = len_T_joi(T_joi_trgt,'spine','neck')
        len_neck2rs    = len_T_joi(T_joi_trgt,'neck','rs')
        len_rs2re      = len_T_joi(T_joi_trgt,'rs','re')
        len_re2rw      = len_T_joi(T_joi_trgt,'re','rw')
        len_neck2ls    = len_T_joi(T_joi_trgt,'neck','ls')
        len_ls2le      = len_T_joi(T_joi_trgt,'ls','le')
        len_le2lw      = len_T_joi(T_joi_trgt,'le','lw')
        
    # Loop
    qpos_dir_list = []
    L = len(secs)
    for tick in range(L): # for each tick
        # Pause each tick (for debugging)
        if pause_each_tick:
            env.viewer_pause()
        
        # Source chain
        sec = secs[tick]
        chain_src = chains[tick]
        """
            Source-specific
        """
        if src_name == 'cmu':
            T_joi_src = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
        elif src_name == 'smpl':
            T_joi_src = get_T_joi_of_chain_smpl(chain_src,base_between_pelvis=True)
        
        # Move target base (once)
        if tick == 0:
            T_base_trgt  = T_joi_src['base']
            """
                Source-specific
            """
            if src_name == 'cmu': # for CMU-MoCap (y-up z-front), make it z-up x-front
                T_base_trgt = T_yuzf2zuxf(T_base_trgt)
            if trgt_name == 'humanoid' or trgt_name == 'baymax' or trgt_name == 'jf3':
                pass
            else:
                env.set_T_base_body(body_name=joi_body_name_trgt['root'],T=T_base_trgt)
            env.forward()

        # Get unit vectors of the source rig
        uv_base2spine = uv_T_joi(T_joi_src,'base','spine')
        uv_spine2neck = uv_T_joi(T_joi_src,'spine','neck')
        uv_neck2rs    = uv_T_joi(T_joi_src,'neck','rs')
        uv_rs2re      = uv_T_joi(T_joi_src,'rs','re')
        uv_re2rw      = uv_T_joi(T_joi_src,'re','rw')
        uv_neck2ls    = uv_T_joi(T_joi_src,'neck','ls')
        uv_ls2le      = uv_T_joi(T_joi_src,'ls','le')
        uv_le2lw      = uv_T_joi(T_joi_src,'le','lw')
        uv_base2rp    = uv_T_joi(T_joi_src,'base','rp')
        uv_rp2rk      = uv_T_joi(T_joi_src,'rp','rk')
        uv_rk2ra      = uv_T_joi(T_joi_src,'rk','ra')
        uv_base2lp    = uv_T_joi(T_joi_src,'base','lp')
        uv_lp2lk      = uv_T_joi(T_joi_src,'lp','lk')
        uv_lk2la      = uv_T_joi(T_joi_src,'lk','la')

        # Set positional targets
        """
            Target-specific
        """
        if trgt_name == 'g1':
            T_joi_trgt   = get_T_joi_of_g1(env)
        elif trgt_name == 'cr':
            T_joi_trgt   = get_T_joi_of_cr(env)
        elif trgt_name == 'humanoid':
            T_joi_trgt   = get_T_joi_of_humanoid(env)
        elif trgt_name == 'baymax':
            T_joi_trgt   = get_T_joi_of_baymax(env)
        elif trgt_name == 'jf3':
            T_joi_trgt   = get_T_joi_of_jf3(env)
        elif trgt_name == 'mahru':
            T_joi_trgt   = get_T_joi_of_mahru(env)
        
        # Set IK targets 
        ik_info_whole_body = init_ik_info()

        if trgt_name == 'baymax' or trgt_name == 'jf3':
            # Upper body position targets
            p_base_trgt = t2p(T_joi_trgt['base'])
            p_spine_trgt = p_base_trgt + len_base2spine*uv_base2spine
            p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
            p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
            p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
            p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
            p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
            p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
            p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
            if ik_p_spine: # spine target is optional
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
            if ik_p_neck: # neck target is optional
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
        else:
            # Lower body position targets
            p_base_trgt = t2p(T_joi_src['base'])
            p_rp_trgt   = p_base_trgt + len_base2rp*uv_base2rp
            p_rk_trgt   = p_rp_trgt + len_rp2rk*uv_rp2rk
            p_ra_trgt   = p_rk_trgt + len_rk2ra*uv_rk2ra
            p_lp_trgt   = p_base_trgt + len_base2lp*uv_base2lp
            p_lk_trgt   = p_lp_trgt + len_lp2lk*uv_lp2lk
            p_la_trgt   = p_lk_trgt + len_lk2la*uv_lk2la
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rp'],p_trgt=p_rp_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lp'],p_trgt=p_lp_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rk'],p_trgt=p_rk_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lk'],p_trgt=p_lk_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ra'],p_trgt=p_ra_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['la'],p_trgt=p_la_trgt)
            
            # Upper body position targets
            p_spine_trgt = p_base_trgt + len_base2spine*uv_base2spine
            p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
            p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
            p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
            p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
            p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
            p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
            p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
            if ik_p_spine: # spine target is optional
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
            if ik_p_neck: # neck target is optional
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
        
        # Hand rotation targets
        if ik_R_hand:
            R_rh_trgt = t2r(T_joi_src['rh'])
            R_lh_trgt = t2r(T_joi_src['lh'])
            """
                Source and Target-specific
            """
            if (src_name=='cmu') and (trgt_name=='g1'): # CMU to G1
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([180,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,0]))
            elif (src_name=='cmu') and (trgt_name=='cr'): # CMU to Common-Rig
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([-90,0,0]))@rpy2r(np.deg2rad([0,0,-90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([-90,0,0]))@rpy2r(np.deg2rad([0,0,-90]))
            elif (src_name=='cmu') and (trgt_name=='humanoid'): # CMU to Humanoid
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,0]))@rpy2r(np.deg2rad([0,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,0]))@rpy2r(np.deg2rad([0,0,0]))

            elif (src_name=='smpl') and (trgt_name=='jf3'): # SMPL to JF3
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,180]))@rpy2r(np.deg2rad([90,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,0,0]))
            elif (src_name=='smpl') and (trgt_name=='baymax'): # SMPL to Baymax
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,0]))

            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rh'],R_trgt=R_rh_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lh'],R_trgt=R_lh_trgt)
        
        # Solve IK for the Lower Bodies
        ik_err_norm_list = []
        for ik_tick in range(max_ik_tick): # ik loop
            dq,ik_err_stack = get_dq_from_ik_info(
                env            = env,
                ik_info        = ik_info_whole_body,
                stepsize       = ik_stepsize,
                eps            = 1e-3,
                th             = np.radians(10.0),
                joint_idxs_jac = joint_with_base_idxs_jac,
            ) # dq:[43]
            qpos = env.get_qpos() # get current joint position  [44]
            mujoco.mj_integratePos(env.model,qpos,dq,1)
            env.forward(q=qpos)
            ik_err_norm = np.linalg.norm(ik_err_stack)
            ik_err_norm_list.append(ik_err_norm)
            if ik_err_norm < ik_err_th: break
        if print_ik_failure and (ik_err_norm>=ik_err_th):
            print ("Lower Body IK Error Too High: tick:[%d] ik_err_norm:[%.2e] ik_err_th:[%.2e]"%
                   (tick,ik_err_norm,ik_err_th))
            
        # Append joint position
        qpos_dir_list.append(env.get_qpos())
        
        # Render
        if animate and (tick%render_every==0 or tick==0):
            chain_src.plot_chain_mujoco(
                env             = env,
                plot_joint      = False,
                plot_joint_name = False,
                r_link          = 0.0025,
                rgba_link       = (1,0,0,0.5),
            ) # source link with red lines
            env.plot_links_between_bodies(
                r    = 0.0025,
                rgba = (0,0,1,0.5),
            ) # plot target with blue
            env.plot_T()
            env.plot_text(p=np.array([0,0,1]),label='tick:[%d/%d time:[%.3f]sec'%(tick,L,sec))
            env.plot_joint_axis(axis_len=0.01,axis_r=0.001)
            env.plot_contact_info()
            plot_ik_info(env=env,ik_info=ik_info_whole_body)
            if plot_joi_src:
                """
                    Source-specific
                """
                if src_name == 'cmu':
                    T_joi_src = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
                for key in T_joi_src.keys():
                    env.plot_T(T=T_joi_src[key],plot_axis=True,axis_len=0.05,axis_width=0.005)
            if plot_joi_trgt:
                """
                    Target-specific
                """
                if trgt_name == 'g1':
                    T_joi_trgt   = get_T_joi_of_g1(env)
                elif trgt_name == 'cr':
                    T_joi_trgt   = get_T_joi_of_cr(env)
                elif trgt_name == 'humanoid':
                    T_joi_trgt   = get_T_joi_of_humanoid(env)
                elif trgt_name == 'baymax':
                    T_joi_trgt   = get_T_joi_of_baymax(env)
                elif trgt_name == 'jf3':
                    T_joi_trgt   = get_T_joi_of_jf3(env)
                elif trgt_name == 'mahru':
                    T_joi_trgt   = get_T_joi_of_mahru(env)
                for key in T_joi_trgt.keys():
                    env.plot_T(T=T_joi_trgt[key],plot_axis=True,axis_len=0.05,axis_width=0.005)
            env.render()
            
        # Print process
        if verbose:
            if ((tick%print_every)==0) or (tick==(L-1)):
                print ("[%d/%d] [%.1f]%% ik_err_norm:[%.2e] ik_err_th:[%.2e]"%
                       (tick,L,100*tick/L,ik_err_norm,ik_err_th))
                
    # Close viewer
    if animate: env.close_viewer()  
    
    # Return
    return qpos_dir_list


def directional_preretarget_v2(
        env,
        secs,
        chains,
        src_name           = 'cmu',
        trgt_name          = 'g1',
        joint_names_for_ik = None,
        max_ik_tick        = 100, # maximum IK iteration
        ik_stepsize        = 1, # IK stepsize
        ik_err_th          = 1e-3, # IK error threshold
        print_ik_failure   = False,
        ik_R_hand          = False,
        reset_env          = True,
        animate            = True,
        plot_joi_src       = False,
        plot_joi_trgt      = False,
        render_every       = 1,
        verbose            = True,
        print_every        = 10,
        pause_each_tick    = False,
        post_rigging       = True,
        knee_unbend_rate   = 0,
    ):
    """ 
        Directional pre-retargeting (aka unit-vector method)
        Return: (qpos_dir_list)
    """
    
    # Reset env
    if reset_env:
        env.reset(step=True)
        
    # Initialize viewer
    if animate:
        env.init_viewer(
            title       = 'Directional Pre-Retargeting Process',
            transparent = True,
        )
    
    # Initialize IK configuration
    if joint_names_for_ik is None: 
        joint_names_for_ik = env.rev_joint_names 
    joint_idxs_jac = env.get_idxs_jac(joint_names=joint_names_for_ik)
    joint_with_base_idxs_jac = np.concatenate(([0,1,2,3,4,5],joint_idxs_jac))
    
    # Target JOI body name
    """
        Target-specific
    """
    if trgt_name == 'g1':
        joi_body_name_trgt = get_joi_body_name_of_g1()
        T_joi_trgt         = get_T_joi_of_g1(env)
        ik_p_spine         = False 
        ik_p_neck          = False
    elif trgt_name == 'cr':
        joi_body_name_trgt = get_joi_body_name_of_cr()
        T_joi_trgt         = get_T_joi_of_cr(env)
        ik_p_spine         = True 
        ik_p_neck          = True
    elif trgt_name == 'humanoid':
        joi_body_name_trgt = get_joi_body_name_of_humanoid()
        T_joi_trgt         = get_T_joi_of_humanoid(env)
        ik_p_spine         = False 
        ik_p_neck          = False
    elif trgt_name == 'baymax':
        joi_body_name_trgt = get_joi_body_name_of_baymax()
        T_joi_trgt         = get_T_joi_of_baymax(env)
        ik_p_spine         = False 
        ik_p_neck          = False
    elif trgt_name == 'jf3':
        joi_body_name_trgt = get_joi_body_name_of_jf3()
        T_joi_trgt         = get_T_joi_of_jf3(env)
        ik_p_spine         = False 
        ik_p_neck          = False
    elif trgt_name == 'mahru':
        joi_body_name_trgt = get_joi_body_name_of_mahru()
        T_joi_trgt         = get_T_joi_of_mahru(env)
        ik_p_spine         = False 
        ik_p_neck          = False
    elif trgt_name == 'auto_gen':
        joi_body_name_trgt = get_joi_body_name_of_auto_gen()
        T_joi_trgt         = get_T_joi_of_auto_gen(env)
        ik_p_spine         = False
        ik_p_neck          = False

    if trgt_name == 'baymax' or trgt_name == 'jf3' or trgt_name == 'auto_gen':
        # Get target lengths
        len_base2spine = len_T_joi(T_joi_trgt,'base','spine')
        len_spine2neck = len_T_joi(T_joi_trgt,'spine','neck')
        len_neck2rs    = len_T_joi(T_joi_trgt,'neck','rs')
        len_rs2re      = len_T_joi(T_joi_trgt,'rs','re')
        len_re2rw      = len_T_joi(T_joi_trgt,'re','rw')
        len_neck2ls    = len_T_joi(T_joi_trgt,'neck','ls')
        len_ls2le      = len_T_joi(T_joi_trgt,'ls','le')
        len_le2lw      = len_T_joi(T_joi_trgt,'le','lw')
    else:
        # Get target lengths
        len_base2rp    = len_T_joi(T_joi_trgt,'base','rp')
        len_rp2rk      = len_T_joi(T_joi_trgt,'rp','rk')
        len_rk2ra      = len_T_joi(T_joi_trgt,'rk','ra')
        len_base2lp    = len_T_joi(T_joi_trgt,'base','lp')
        len_lp2lk      = len_T_joi(T_joi_trgt,'lp','lk')
        len_lk2la      = len_T_joi(T_joi_trgt,'lk','la')
        len_base2spine = len_T_joi(T_joi_trgt,'base','spine')
        len_spine2neck = len_T_joi(T_joi_trgt,'spine','neck')
        len_neck2rs    = len_T_joi(T_joi_trgt,'neck','rs')
        len_rs2re      = len_T_joi(T_joi_trgt,'rs','re')
        len_re2rw      = len_T_joi(T_joi_trgt,'re','rw')
        len_neck2ls    = len_T_joi(T_joi_trgt,'neck','ls')
        len_ls2le      = len_T_joi(T_joi_trgt,'ls','le')
        len_le2lw      = len_T_joi(T_joi_trgt,'le','lw')
        
    # Loop
    qpos_dir_list = []
    L = len(secs)
    for tick in range(L): # for each tick
        
        # Pause each tick (for debugging)
        if pause_each_tick:
            env.viewer_pause()
        
        # Source chain
        sec = secs[tick]
        chain_src = chains[tick]
        """
            Source-specific
        """
        if src_name == 'cmu':
            T_joi_src = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
        elif src_name == 'smpl':
            T_joi_src = get_T_joi_of_chain_smpl(chain_src,base_between_pelvis=True)
        
        # Move target base (once)
        if tick == 0:
            T_base_trgt  = T_joi_src['base']
            """
                Source-specific
            """
            if src_name == 'cmu': # for CMU-MoCap (y-up z-front), make it z-up x-front
                T_base_trgt = T_yuzf2zuxf(T_base_trgt)
            if trgt_name == 'humanoid' or trgt_name == 'baymax' or trgt_name == 'jf3' or trgt_name == 'mahru' or trgt_name =='auto_gen':
                pass
            else:
                env.set_T_base_body(body_name=joi_body_name_trgt['root'],T=T_base_trgt)
            env.forward()

        # Get unit vectors of the source rig
        uv_base2spine = uv_T_joi(T_joi_src,'base','spine')
        uv_spine2neck = uv_T_joi(T_joi_src,'spine','neck')
        uv_neck2rs    = uv_T_joi(T_joi_src,'neck','rs')
        uv_rs2re      = uv_T_joi(T_joi_src,'rs','re')
        uv_re2rw      = uv_T_joi(T_joi_src,'re','rw')
        uv_neck2ls    = uv_T_joi(T_joi_src,'neck','ls')
        uv_ls2le      = uv_T_joi(T_joi_src,'ls','le')
        uv_le2lw      = uv_T_joi(T_joi_src,'le','lw')
        uv_base2rp    = uv_T_joi(T_joi_src,'base','rp')
        uv_rp2rk      = uv_T_joi(T_joi_src,'rp','rk')
        uv_rk2ra      = uv_T_joi(T_joi_src,'rk','ra')
        uv_base2lp    = uv_T_joi(T_joi_src,'base','lp')
        uv_lp2lk      = uv_T_joi(T_joi_src,'lp','lk')
        uv_lk2la      = uv_T_joi(T_joi_src,'lk','la')

        # Set positional targets
        """
            Target-specific
        """
        if trgt_name == 'g1':
            T_joi_trgt   = get_T_joi_of_g1(env)
        elif trgt_name == 'cr':
            T_joi_trgt   = get_T_joi_of_cr(env)
        elif trgt_name == 'humanoid':
            T_joi_trgt   = get_T_joi_of_humanoid(env)
        elif trgt_name == 'baymax':
            T_joi_trgt   = get_T_joi_of_baymax(env)
        elif trgt_name == 'jf3':
            T_joi_trgt   = get_T_joi_of_jf3(env)
        elif trgt_name == 'mahru':
            T_joi_trgt   = get_T_joi_of_mahru(env)
        elif trgt_name == 'auto_gen':
            T_joi_trgt   = get_T_joi_of_auto_gen(env)
        
        # Set IK targets 
        ik_info_whole_body = init_ik_info()

        if post_rigging:
            if trgt_name == 'baymax' or trgt_name == 'jf3' or trgt_name == 'auto_gen':
                # Uprighting
                root_yaw_range = np.deg2rad([-0.1, 0.1])
                yaw_min_rad = root_yaw_range[0]
                yaw_max_rad = root_yaw_range[1]
                p_rp = t2p(T_joi_src['rp'])
                p_lp = t2p(T_joi_src['lp'])
                pelvis_rad = np.arctan2(p_rp[1]-p_lp[1],p_rp[0]-p_lp[0])
                yaw_rad = pelvis_rad + np.deg2rad(90)
                if yaw_rad > np.deg2rad(180):
                    yaw_rad = yaw_rad - np.deg2rad(360)
                elif yaw_rad < np.deg2rad(-180):
                    yaw_rad = yaw_rad + np.deg2rad(360)
                if yaw_rad < yaw_min_rad:
                    yaw_offset_rad = yaw_min_rad - yaw_rad
                elif yaw_rad > yaw_max_rad:
                    yaw_offset_rad = yaw_max_rad - yaw_rad
                else:
                    yaw_offset_rad = 0
                R_rot = rpy2r([0,0,yaw_offset_rad])
                chain_src.set_root_joint_R(R_rot@t2r(T_joi_src['base']))
                chain_src.forward_kinematics()

                if src_name == 'cmu':
                    T_joi_upright = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
                elif src_name == 'smpl':
                    T_joi_upright = get_T_joi_of_chain_smpl(chain_src,base_between_pelvis=True)

                # Upper body position targets
                p_base_trgt = t2p(T_joi_trgt['base'])
                p_spine_trgt = p_base_trgt + len_base2spine*uv_base2spine
                p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                if ik_p_spine: # spine target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
                if ik_p_neck: # neck target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
            else:
                # Uprighting
                root_yaw_range = np.deg2rad([-30, 30])
                yaw_min_rad = root_yaw_range[0]
                yaw_max_rad = root_yaw_range[1]
                p_rp = t2p(T_joi_src['rp'])
                p_lp = t2p(T_joi_src['lp'])
                pelvis_rad = np.arctan2(p_rp[1]-p_lp[1],p_rp[0]-p_lp[0])
                yaw_rad = pelvis_rad + np.deg2rad(90)
                if yaw_rad > np.deg2rad(180):
                    yaw_rad = yaw_rad - np.deg2rad(360)
                elif yaw_rad < np.deg2rad(-180):
                    yaw_rad = yaw_rad + np.deg2rad(360)
                if yaw_rad < yaw_min_rad:
                    yaw_offset_rad = yaw_min_rad - yaw_rad
                elif yaw_rad > yaw_max_rad:
                    yaw_offset_rad = yaw_max_rad - yaw_rad
                else:
                    yaw_offset_rad = 0
                R_rot = rpy2r([0,0,yaw_offset_rad])
                chain_src.set_root_joint_R(R_rot@t2r(T_joi_src['base']))
                chain_src.forward_kinematics()

                if src_name == 'cmu':
                    T_joi_upright = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
                elif src_name == 'smpl':
                    T_joi_upright = get_T_joi_of_chain_smpl(chain_src,base_between_pelvis=True)

                # Post-rigging
                p_base_upright = t2p(T_joi_upright['base'])
                p_rp_upright  = t2p(T_joi_upright['rp'])
                p_rk_upright   = t2p(T_joi_upright['rk'])
                p_ra_upright   = t2p(T_joi_upright['ra'])
                p_la_upright   = t2p(T_joi_upright['la'])
                # Set right and left ankle positions (fix the width between the feet)
                p_ca_z = 0.5*(p_ra_upright[2] + p_la_upright[2])
                p_ca   = [p_base_upright[0], p_base_upright[1], p_ca_z]
                if tick == 0:
                    rs2ls_dist = np.linalg.norm(t2p(T_joi_src['rs']) - t2p(T_joi_src['ls']))
                p_offset_ca2ra = 0.5*rs2ls_dist*np.array([0, -1, 0])
                p_offset_ca2la = 0.5*rs2ls_dist*np.array([0,1,0])
                p_ra_post = p_ca + p_offset_ca2ra
                p_la_post = p_ca + p_offset_ca2la
                # Set right and left knee positions (based on updated ankle positions)
                p_rk_post = p_ra_post
                p_lk_post = p_la_post
                p_rk_y = p_rk_post[1]
                p_lk_y = p_lk_post[1]
                p_ck_y = 0.5*(p_rk_y + p_lk_y)
                rp2lp_dist = np.linalg.norm(t2p(T_joi_upright['rp']) - t2p(T_joi_upright['lp']))
                p_rk_post[1] = p_ck_y +np.sqrt(rp2lp_dist/rs2ls_dist)*(p_rk_y - p_ck_y)
                p_lk_post[1] = p_ck_y +np.sqrt(rp2lp_dist/rs2ls_dist)*(p_lk_y - p_ck_y)
                a2k_dist = np.linalg.norm(p_ra_upright - p_rk_upright)
                k2p_dist = np.linalg.norm(p_rk_upright - p_rp_upright)
                p_rk_post[2] = p_rk_post[2] + a2k_dist
                p_lk_post[2] = p_lk_post[2] + a2k_dist
                # Push knee x-positions
                p_rk_post[0] = p_rk_post[0] + 0.1
                p_lk_post[0] = p_lk_post[0] + 0.1
                # Avoid too much knee bending
                a2p_dist   = a2k_dist + k2p_dist
                p2a_z_dist = p_rp_upright[2]-p_ra_post[2]
                z_diff4a   = a2p_dist - p2a_z_dist + 0.1 # 0.1 is a margin
                torso_x    = p_base_upright[0]
                torso_z    = p_base_upright[2]

                # Modify IK targets of x- and z-positions of ankles to reflect 'knee_unbend_rate'
                p_ra_post[0] = (1-knee_unbend_rate)*p_ra_post[0] + knee_unbend_rate*torso_x
                p_la_post[0] = (1-knee_unbend_rate)*p_la_post[0] + knee_unbend_rate*torso_x
                p_ra_post[2] = p_ra_post[2] - knee_unbend_rate*z_diff4a
                p_la_post[2] = p_la_post[2] - knee_unbend_rate*z_diff4a
                # Modify IK target of x- and z-position of knees
                z_knee_unbend = 0.5*(torso_z +p_ra_post[2])
                p_rk_post[0] = (1-knee_unbend_rate)*p_rk_post[0] + knee_unbend_rate*torso_x
                p_lk_post[0] = (1-knee_unbend_rate)*p_lk_post[0] + knee_unbend_rate*torso_x
                p_rk_post[2] = (1-knee_unbend_rate)*p_rk_post[2] + knee_unbend_rate*z_knee_unbend
                p_lk_post[2] = (1-knee_unbend_rate)*p_lk_post[2] + knee_unbend_rate*z_knee_unbend

                # Get unit vectors of the source rig
                uv_base2spine = uv_T_joi(T_joi_upright,'base','spine')
                uv_spine2neck = uv_T_joi(T_joi_upright,'spine','neck')
                uv_neck2rs    = uv_T_joi(T_joi_upright,'neck','rs')
                uv_rs2re      = uv_T_joi(T_joi_upright,'rs','re')
                uv_re2rw      = uv_T_joi(T_joi_upright,'re','rw')
                uv_neck2ls    = uv_T_joi(T_joi_upright,'neck','ls')
                uv_ls2le      = uv_T_joi(T_joi_upright,'ls','le')
                uv_le2lw      = uv_T_joi(T_joi_upright,'le','lw')
                uv_base2rp    = uv_T_joi(T_joi_upright,'base','rp')
                uv_base2lp    = uv_T_joi(T_joi_upright,'base','lp')

                # Lower body position targets
                p_rp_trgt   = p_base_upright + len_base2rp*uv_base2rp
                p_rk_trgt   = p_rk_post
                p_ra_trgt   = p_ra_post
                p_lp_trgt   = p_base_upright + len_base2lp*uv_base2lp
                p_lk_trgt   = p_lk_post
                p_la_trgt   = p_la_post
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rp'],p_trgt=p_rp_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lp'],p_trgt=p_lp_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rk'],p_trgt=p_rk_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lk'],p_trgt=p_lk_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ra'],p_trgt=p_ra_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['la'],p_trgt=p_la_trgt)

                # Upper body position targets
                p_spine_trgt = p_base_upright + len_base2spine*uv_base2spine
                p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                if ik_p_spine: # spine target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
                if ik_p_neck: # neck target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
            
        else:
            if trgt_name == 'baymax' or trgt_name == 'jf3' or trgt_name == 'auto_gen':
                # Upper body position targets
                p_base_trgt = t2p(T_joi_trgt['base'])
                p_spine_trgt = p_base_trgt + len_base2spine*uv_base2spine
                p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                if ik_p_spine: # spine target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
                if ik_p_neck: # neck target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
            else:
                # Lower body position targets
                p_base_trgt = t2p(T_joi_src['base'])
                p_rp_trgt   = p_base_trgt + len_base2rp*uv_base2rp
                p_rk_trgt   = p_rp_trgt + len_rp2rk*uv_rp2rk
                p_ra_trgt   = p_rk_trgt + len_rk2ra*uv_rk2ra
                p_lp_trgt   = p_base_trgt + len_base2lp*uv_base2lp
                p_lk_trgt   = p_lp_trgt + len_lp2lk*uv_lp2lk
                p_la_trgt   = p_lk_trgt + len_lk2la*uv_lk2la
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rp'],p_trgt=p_rp_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lp'],p_trgt=p_lp_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rk'],p_trgt=p_rk_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lk'],p_trgt=p_lk_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ra'],p_trgt=p_ra_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['la'],p_trgt=p_la_trgt)
                
                # Upper body position targets
                p_spine_trgt = p_base_trgt + len_base2spine*uv_base2spine
                p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                if ik_p_spine: # spine target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
                if ik_p_neck: # neck target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
            
        # Hand rotation targets
        if ik_R_hand:
            R_rh_trgt = t2r(T_joi_src['rh'])
            R_lh_trgt = t2r(T_joi_src['lh'])
            """
                Source and Target-specific
            """
            if (src_name=='cmu') and (trgt_name=='g1'): # CMU to G1
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([180,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,0]))
            elif (src_name=='cmu') and (trgt_name=='cr'): # CMU to Common-Rig
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([-90,0,0]))@rpy2r(np.deg2rad([0,0,-90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([-90,0,0]))@rpy2r(np.deg2rad([0,0,-90]))
            elif (src_name=='cmu') and (trgt_name=='humanoid'): # CMU to Humanoid
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,0]))@rpy2r(np.deg2rad([0,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,0]))@rpy2r(np.deg2rad([0,0,0]))

            elif (src_name=='smpl') and (trgt_name=='jf3'): # SMPL to JF3
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,180]))@rpy2r(np.deg2rad([90,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,0,0]))
            elif (src_name=='smpl') and (trgt_name=='baymax'): # SMPL to Baymax
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,0]))

            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rh'],R_trgt=R_rh_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lh'],R_trgt=R_lh_trgt)
        
        # Solve IK for the Lower Bodies
        ik_err_norm_list = []
        for ik_tick in range(max_ik_tick): # ik loop
            dq,ik_err_stack = get_dq_from_ik_info(
                env            = env,
                ik_info        = ik_info_whole_body,
                stepsize       = ik_stepsize,
                eps            = 1e-3,
                th             = np.radians(10.0),
                joint_idxs_jac = joint_with_base_idxs_jac,
            ) # dq:[43]
            qpos = env.get_qpos() # get current joint position  [44]
            mujoco.mj_integratePos(env.model,qpos,dq,1)
            env.forward(q=qpos)
            ik_err_norm = np.linalg.norm(ik_err_stack)
            ik_err_norm_list.append(ik_err_norm)
            if ik_err_norm < ik_err_th: break
        if print_ik_failure and (ik_err_norm>=ik_err_th):
            print ("Lower Body IK Error Too High: tick:[%d] ik_err_norm:[%.2e] ik_err_th:[%.2e]"%
                   (tick,ik_err_norm,ik_err_th))

        # Append joint position
        qpos_dir_list.append(env.get_qpos())

        if trgt_name == 'mahru':
            qpos_dir_list[tick][18] = 1.0
            qpos_dir_list[tick][22] = -1.0
        
        # Render
        if animate and (tick%render_every==0 or tick==0):
            chain_src.plot_chain_mujoco(
                env             = env,
                plot_joint      = False,
                plot_joint_name = False,
                r_link          = 0.0025,
                rgba_link       = (1,0,0,0.5),
            ) # source link with red lines
            env.plot_links_between_bodies(
                r    = 0.0025,
                rgba = (0,0,1,0.5),
            ) # plot target with blue
            env.plot_T()
            env.plot_text(p=np.array([0,0,1]),label='tick:[%d/%d time:[%.3f]sec'%(tick,L,sec))
            env.plot_joint_axis(axis_len=0.01,axis_r=0.001)
            env.plot_contact_info()
            plot_ik_info(env=env,ik_info=ik_info_whole_body)
            if plot_joi_src:
                """
                    Source-specific
                """
                if src_name == 'cmu':
                    T_joi_src = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
                for key in T_joi_src.keys():
                    env.plot_T(T=T_joi_src[key],plot_axis=True,axis_len=0.05,axis_width=0.005)
            if plot_joi_trgt:
                """
                    Target-specific
                """
                if trgt_name == 'g1':
                    T_joi_trgt   = get_T_joi_of_g1(env)
                elif trgt_name == 'cr':
                    T_joi_trgt   = get_T_joi_of_cr(env)
                elif trgt_name == 'humanoid':
                    T_joi_trgt   = get_T_joi_of_humanoid(env)
                elif trgt_name == 'baymax':
                    T_joi_trgt   = get_T_joi_of_baymax(env)
                elif trgt_name == 'jf3':
                    T_joi_trgt   = get_T_joi_of_jf3(env)
                elif trgt_name == 'mahru':
                    T_joi_trgt   = get_T_joi_of_mahru(env)
                elif trgt_name == 'auto_gen':
                    T_joi_trgt   = get_T_joi_of_auto_gen(env)
                for key in T_joi_trgt.keys():
                    env.plot_T(T=T_joi_trgt[key],plot_axis=True,axis_len=0.05,axis_width=0.005)
                    
            env.render()
            
        # Print process
        if verbose:
            if ((tick%print_every)==0) or (tick==(L-1)):
                print ("[%d/%d] [%.1f]%% ik_err_norm:[%.2e] ik_err_th:[%.2e]"%
                       (tick,L,100*tick/L,ik_err_norm,ik_err_th))
                
    # Close viewer
    if animate: env.close_viewer()  
    
    # Return
    return qpos_dir_list

def get_T_joi_trgt(env, trgt_name, override=None):
    # e.g. call the function through a dictionary
    if trgt_name not in T_JOI_FUNC:
        raise ValueError(f"Unknown target name: {trgt_name}")

    get_T_joi_func = T_JOI_FUNC[trgt_name]
    if override is not None:
        T_joi_trgt = get_T_joi_func(env, override=override)
    else:
        T_joi_trgt = get_T_joi_func(env)
    return T_joi_trgt

def get_joi_body_name_trgt(trgt_name, override=None):
    if trgt_name not in JOI_BODY_NAME_FUNC:
        raise ValueError(f"Unknown target name: {trgt_name}")

    joi_body_name_trgt_func = JOI_BODY_NAME_FUNC[trgt_name]
    if override is not None:
        joi_body_name_trgt = joi_body_name_trgt_func(override=override)
    else:
        joi_body_name_trgt = joi_body_name_trgt_func()
    return joi_body_name_trgt

def directional_preretarget_yoon(
        env,
        secs,
        chains,
        ik_p_spine=False,
        ik_p_neck=False,
        leg_fixed=False,
        src_name           = 'cmu',
        trgt_name          = 'g1',
        joint_names_for_ik = None,
        max_ik_tick        = 100, # maximum IK iteration
        ik_stepsize        = 1, # IK stepsize
        ik_err_th          = 1e-3, # IK error threshold
        print_ik_failure   = False,
        ik_R_hand          = False,
        ik_R_foot          = True,
        reset_env          = True,
        animate            = True,
        plot_joi_src       = False,
        plot_joi_trgt      = False,
        render_every       = 1,
        verbose            = True,
        print_every        = 10,
        pause_each_tick    = False,
        post_rigging       = True,
        knee_unbend_rate   = 0,
        override           = None,
    ):
    """ 
        Directional pre-retargeting (aka unit-vector method)
        Uses directional pre-retargeting, generalized so that it works across many robots
        get_joi_body_name_of_() for joi_body_name_trgt, or for T_joi_trgt
        moved get_T_joi_of_() out. JOI references are also organized through __init__.py
        for how __init__.py is used, see https://www.youtube.com/watch?v=8FceAhTVGeo
        Return: (qpos_dir_list)
    """

    # Reset env
    if reset_env:
        env.reset(step=True)
        
    # Initialize viewer
    if animate:
        env.init_viewer(
            title       = 'Directional Pre-Retargeting Process',
            transparent = True,
        )
    
    # Initialize IK configuration
    if joint_names_for_ik is None: 
        joint_names_for_ik = env.rev_joint_names 
    joint_idxs_jac = env.get_idxs_jac(joint_names=joint_names_for_ik)
    joint_with_base_idxs_jac = np.concatenate(([0,1,2,3,4],joint_idxs_jac))
    
    # Target JOI body name
    """
        Target-specific
    """
    if override is not None:
        joi_body_name_trgt = get_joi_body_name_trgt(trgt_name,override=override)
        T_joi_trgt = get_T_joi_trgt(env,trgt_name,override=override)
    else:
        joi_body_name_trgt = get_joi_body_name_trgt(trgt_name)
        T_joi_trgt = get_T_joi_trgt(env,trgt_name)
    if trgt_name == 'baymax' or trgt_name == 'jf3' or trgt_name == 'auto_gen' or leg_fixed:
        # Get target lengths
        len_base2spine = len_T_joi(T_joi_trgt,'base','spine')
        len_spine2neck = len_T_joi(T_joi_trgt,'spine','neck')
        len_neck2rs    = len_T_joi(T_joi_trgt,'neck','rs')
        len_rs2re      = len_T_joi(T_joi_trgt,'rs','re')
        len_re2rw      = len_T_joi(T_joi_trgt,'re','rw')
        len_neck2ls    = len_T_joi(T_joi_trgt,'neck','ls')
        len_ls2le      = len_T_joi(T_joi_trgt,'ls','le')
        len_le2lw      = len_T_joi(T_joi_trgt,'le','lw')
    else:
        # Get target lengths
        len_base2rp    = len_T_joi(T_joi_trgt,'base','rp')
        len_rp2rk      = len_T_joi(T_joi_trgt,'rp','rk')
        len_rk2ra      = len_T_joi(T_joi_trgt,'rk','ra')
        len_base2lp    = len_T_joi(T_joi_trgt,'base','lp')
        len_lp2lk      = len_T_joi(T_joi_trgt,'lp','lk')
        len_lk2la      = len_T_joi(T_joi_trgt,'lk','la')
        len_base2spine = len_T_joi(T_joi_trgt,'base','spine')
        len_spine2neck = len_T_joi(T_joi_trgt,'spine','neck')
        len_neck2rs    = len_T_joi(T_joi_trgt,'neck','rs')
        len_rs2re      = len_T_joi(T_joi_trgt,'rs','re')
        len_re2rw      = len_T_joi(T_joi_trgt,'re','rw')
        len_neck2ls    = len_T_joi(T_joi_trgt,'neck','ls')
        len_ls2le      = len_T_joi(T_joi_trgt,'ls','le')
        len_le2lw      = len_T_joi(T_joi_trgt,'le','lw')
        
    # Loop
    qpos_dir_list = []
    L = len(secs)
    for tick in range(L): # for each tick
        
        # Pause each tick (for debugging)
        if pause_each_tick:
            env.viewer_pause()
        
        # Source chain
        sec = secs[tick]
        chain_src = chains[tick]
        """
            Source-specific
        """
        if src_name == 'cmu':
            T_joi_src = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
        elif src_name == 'smpl':
            T_joi_src = get_T_joi_of_chain_smpl(chain_src,base_between_pelvis=True)
        
        # Move target base (once)
        if tick == 0:
            T_base_trgt  = T_joi_src['base']
            """
                Source-specific
            """
            if src_name == 'cmu': # for CMU-MoCap (y-up z-front), make it z-up x-front
                T_base_trgt = T_yuzf2zuxf(T_base_trgt)
            if trgt_name == 'humanoid' or trgt_name == 'baymax' or trgt_name == 'jf3' or trgt_name == 'mahru' or trgt_name =='auto_gen' or leg_fixed:
                pass
            else:
                env.set_T_base_body(body_name=joi_body_name_trgt['root'],T=T_base_trgt)
            env.forward()

        # Get unit vectors of the source rig
        uv_base2spine = uv_T_joi(T_joi_src,'base','spine')
        uv_spine2neck = uv_T_joi(T_joi_src,'spine','neck')
        uv_neck2rs    = uv_T_joi(T_joi_src,'neck','rs')
        uv_rs2re      = uv_T_joi(T_joi_src,'rs','re')
        uv_re2rw      = uv_T_joi(T_joi_src,'re','rw')
        uv_neck2ls    = uv_T_joi(T_joi_src,'neck','ls')
        uv_ls2le      = uv_T_joi(T_joi_src,'ls','le')
        uv_le2lw      = uv_T_joi(T_joi_src,'le','lw')
        uv_base2rp    = uv_T_joi(T_joi_src,'base','rp')
        uv_rp2rk      = uv_T_joi(T_joi_src,'rp','rk')
        uv_rk2ra      = uv_T_joi(T_joi_src,'rk','ra')
        uv_base2lp    = uv_T_joi(T_joi_src,'base','lp')
        uv_lp2lk      = uv_T_joi(T_joi_src,'lp','lk')
        uv_lk2la      = uv_T_joi(T_joi_src,'lk','la')

        # Set positional targets
        """
            Target-specific
        """
        if override is not None:

            T_joi_trgt = get_T_joi_trgt(env,trgt_name,override=override)
        else:
            T_joi_trgt = get_T_joi_trgt(env,trgt_name)
        
        # Set IK targets 
        ik_info_whole_body = init_ik_info()
        if src_name == 'cmu':
            R_base = t2r(T_joi_src['base'])
            yaw = np.arctan2(R_base[1, 0], R_base[0, 0])
            pitch = np.arctan2(-R_base[2, 0], np.sqrt(R_base[2, 1]**2 + R_base[2, 2]**2))
            roll = np.arctan2(R_base[2, 1], R_base[2, 2])
            R_base_modified = rpy2r([roll, pitch, yaw])
            chain_src.set_root_joint_R(R_base_modified)
        chain_src.forward_kinematics()
        if src_name == 'cmu':
            T_joi_src = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
        elif src_name == 'smpl':
            T_joi_src = get_T_joi_of_chain_smpl(chain_src,base_between_pelvis=True)


        if post_rigging:
            if trgt_name == 'baymax' or trgt_name == 'jf3' or trgt_name == 'auto_gen':
                # Uprighting
                root_yaw_range = np.deg2rad([-0.1, 0.1])
                yaw_min_rad = root_yaw_range[0]
                yaw_max_rad = root_yaw_range[1]
                p_rp = t2p(T_joi_src['rp'])
                p_lp = t2p(T_joi_src['lp'])
                pelvis_rad = np.arctan2(p_rp[1]-p_lp[1],p_rp[0]-p_lp[0])
                yaw_rad = pelvis_rad + np.deg2rad(90)
                if yaw_rad > np.deg2rad(180):
                    yaw_rad = yaw_rad - np.deg2rad(360)
                elif yaw_rad < np.deg2rad(-180):
                    yaw_rad = yaw_rad + np.deg2rad(360)
                if yaw_rad < yaw_min_rad:
                    yaw_offset_rad = yaw_min_rad - yaw_rad
                elif yaw_rad > yaw_max_rad:
                    yaw_offset_rad = yaw_max_rad - yaw_rad
                else:
                    yaw_offset_rad = 0
                R_rot = rpy2r([0,0,yaw_offset_rad])
                chain_src.set_root_joint_R(R_rot@t2r(T_joi_src['base']))
                chain_src.forward_kinematics()

                if src_name == 'cmu':
                    T_joi_upright = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
                elif src_name == 'smpl':
                    T_joi_upright = get_T_joi_of_chain_smpl(chain_src,base_between_pelvis=True)

                # Upper body position targets
                p_base_trgt = t2p(T_joi_trgt['base'])
                p_spine_trgt = p_base_trgt + len_base2spine*uv_base2spine
                p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                if ik_p_spine: # spine target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
                if ik_p_neck: # neck target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
            else:
                # Uprighting
                root_yaw_range = np.deg2rad([-30, 30])
                yaw_min_rad = root_yaw_range[0]
                yaw_max_rad = root_yaw_range[1]
                p_rp = t2p(T_joi_src['rp'])
                p_lp = t2p(T_joi_src['lp'])
                pelvis_rad = np.arctan2(p_rp[1]-p_lp[1],p_rp[0]-p_lp[0])
                yaw_rad = pelvis_rad + np.deg2rad(90)
                if yaw_rad > np.deg2rad(180):
                    yaw_rad = yaw_rad - np.deg2rad(360)
                elif yaw_rad < np.deg2rad(-180):
                    yaw_rad = yaw_rad + np.deg2rad(360)
                if yaw_rad < yaw_min_rad:
                    yaw_offset_rad = yaw_min_rad - yaw_rad
                elif yaw_rad > yaw_max_rad:
                    yaw_offset_rad = yaw_max_rad - yaw_rad
                else:
                    yaw_offset_rad = 0
                R_rot = rpy2r([0,0,yaw_offset_rad])
                chain_src.set_root_joint_R(R_rot@t2r(T_joi_src['base']))
                chain_src.forward_kinematics()

                if src_name == 'cmu':
                    T_joi_upright = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
                elif src_name == 'smpl':
                    T_joi_upright = get_T_joi_of_chain_smpl(chain_src,base_between_pelvis=True)

                # Post-rigging
                p_base_upright = t2p(T_joi_upright['base'])
                p_rp_upright  = t2p(T_joi_upright['rp'])
                p_rk_upright   = t2p(T_joi_upright['rk'])
                p_ra_upright   = t2p(T_joi_upright['ra'])
                p_la_upright   = t2p(T_joi_upright['la'])
                # Set right and left ankle positions (fix the width between the feet)
                p_ca_z = 0.5*(p_ra_upright[2] + p_la_upright[2])
                p_ca   = [p_base_upright[0], p_base_upright[1], p_ca_z]
                if tick == 0:
                    rs2ls_dist = np.linalg.norm(t2p(T_joi_src['rs']) - t2p(T_joi_src['ls']))
                p_offset_ca2ra = 0.5*rs2ls_dist*np.array([0, -1, 0])
                p_offset_ca2la = 0.5*rs2ls_dist*np.array([0,1,0])
                p_ra_post = p_ca + p_offset_ca2ra
                p_la_post = p_ca + p_offset_ca2la
                # Set right and left knee positions (based on updated ankle positions)
                p_rk_post = p_ra_post
                p_lk_post = p_la_post
                p_rk_y = p_rk_post[1]
                p_lk_y = p_lk_post[1]
                p_ck_y = 0.5*(p_rk_y + p_lk_y)
                rp2lp_dist = np.linalg.norm(t2p(T_joi_upright['rp']) - t2p(T_joi_upright['lp']))
                p_rk_post[1] = p_ck_y +np.sqrt(rp2lp_dist/rs2ls_dist)*(p_rk_y - p_ck_y)
                p_lk_post[1] = p_ck_y +np.sqrt(rp2lp_dist/rs2ls_dist)*(p_lk_y - p_ck_y)
                a2k_dist = np.linalg.norm(p_ra_upright - p_rk_upright)
                k2p_dist = np.linalg.norm(p_rk_upright - p_rp_upright)
                p_rk_post[2] = p_rk_post[2] + a2k_dist
                p_lk_post[2] = p_lk_post[2] + a2k_dist
                # Push knee x-positions
                p_rk_post[0] = p_rk_post[0] + 0.1
                p_lk_post[0] = p_lk_post[0] + 0.1
                # Avoid too much knee bending
                a2p_dist   = a2k_dist + k2p_dist
                p2a_z_dist = p_rp_upright[2]-p_ra_post[2]
                z_diff4a   = a2p_dist - p2a_z_dist + 0.1 # 0.1 is a margin
                torso_x    = p_base_upright[0]
                torso_z    = p_base_upright[2]

                # Modify IK targets of x- and z-positions of ankles to reflect 'knee_unbend_rate'
                p_ra_post[0] = (1-knee_unbend_rate)*p_ra_post[0] + knee_unbend_rate*torso_x
                p_la_post[0] = (1-knee_unbend_rate)*p_la_post[0] + knee_unbend_rate*torso_x
                p_ra_post[2] = p_ra_post[2] - knee_unbend_rate*z_diff4a
                p_la_post[2] = p_la_post[2] - knee_unbend_rate*z_diff4a
                # Modify IK target of x- and z-position of knees
                z_knee_unbend = 0.5*(torso_z +p_ra_post[2])
                p_rk_post[0] = (1-knee_unbend_rate)*p_rk_post[0] + knee_unbend_rate*torso_x
                p_lk_post[0] = (1-knee_unbend_rate)*p_lk_post[0] + knee_unbend_rate*torso_x
                p_rk_post[2] = (1-knee_unbend_rate)*p_rk_post[2] + knee_unbend_rate*z_knee_unbend
                p_lk_post[2] = (1-knee_unbend_rate)*p_lk_post[2] + knee_unbend_rate*z_knee_unbend

                # Get unit vectors of the source rig
                uv_base2spine = uv_T_joi(T_joi_upright,'base','spine')
                uv_spine2neck = uv_T_joi(T_joi_upright,'spine','neck')
                uv_neck2rs    = uv_T_joi(T_joi_upright,'neck','rs')
                uv_rs2re      = uv_T_joi(T_joi_upright,'rs','re')
                uv_re2rw      = uv_T_joi(T_joi_upright,'re','rw')
                uv_neck2ls    = uv_T_joi(T_joi_upright,'neck','ls')
                uv_ls2le      = uv_T_joi(T_joi_upright,'ls','le')
                uv_le2lw      = uv_T_joi(T_joi_upright,'le','lw')
                uv_base2rp    = uv_T_joi(T_joi_upright,'base','rp')
                uv_base2lp    = uv_T_joi(T_joi_upright,'base','lp')

                # Lower body position targets
                p_rp_trgt   = p_base_upright + len_base2rp*uv_base2rp
                p_rk_trgt   = p_rk_post
                p_ra_trgt   = p_ra_post
                p_lp_trgt   = p_base_upright + len_base2lp*uv_base2lp
                p_lk_trgt   = p_lk_post
                p_la_trgt   = p_la_post
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rp'],p_trgt=p_rp_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lp'],p_trgt=p_lp_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rk'],p_trgt=p_rk_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lk'],p_trgt=p_lk_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ra'],p_trgt=p_ra_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['la'],p_trgt=p_la_trgt)

                # Upper body position targets
                p_spine_trgt = p_base_upright + len_base2spine*uv_base2spine
                p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                if ik_p_spine: # spine target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
                if ik_p_neck: # neck target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
            
        else:
            if trgt_name == 'baymax' or trgt_name == 'jf3' or trgt_name == 'auto_gen' or leg_fixed:
                # Get unit vectors of the source rig
                uv_base2spine = uv_T_joi(T_joi_src,'base','spine')
                uv_spine2neck = uv_T_joi(T_joi_src,'spine','neck')
                uv_neck2rs    = uv_T_joi(T_joi_src,'neck','rs')
                uv_rs2re      = uv_T_joi(T_joi_src,'rs','re')
                uv_re2rw      = uv_T_joi(T_joi_src,'re','rw')
                uv_neck2ls    = uv_T_joi(T_joi_src,'neck','ls')
                uv_ls2le      = uv_T_joi(T_joi_src,'ls','le')
                uv_le2lw      = uv_T_joi(T_joi_src,'le','lw')
                # Upper body position targets
                p_base_trgt = t2p(T_joi_trgt['base'])
                p_spine_trgt = p_base_trgt + len_base2spine*uv_base2spine
                p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                if ik_p_spine: # spine target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
                if ik_p_neck: # neck target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
            else:

                # Lower body position targets
                p_base_trgt = t2p(T_joi_src['base'])
                p_rp_trgt   = p_base_trgt + len_base2rp*uv_base2rp
                p_rk_trgt   = p_rp_trgt + len_rp2rk*uv_rp2rk
                p_ra_trgt   = p_rk_trgt + len_rk2ra*uv_rk2ra
                p_lp_trgt   = p_base_trgt + len_base2lp*uv_base2lp
                p_lk_trgt   = p_lp_trgt + len_lp2lk*uv_lp2lk
                p_la_trgt   = p_lk_trgt + len_lk2la*uv_lk2la
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rp'],p_trgt=p_rp_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lp'],p_trgt=p_lp_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rk'],p_trgt=p_rk_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lk'],p_trgt=p_lk_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ra'],p_trgt=p_ra_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['la'],p_trgt=p_la_trgt)
                
                # Upper body position targets
                p_spine_trgt = p_base_trgt + len_base2spine*uv_base2spine
                p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                if ik_p_spine: # spine target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
                if ik_p_neck: # neck target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
            
        # Hand rotation targets
        if ik_R_hand:
            R_rh_trgt = t2r(T_joi_src['rh'])
            R_lh_trgt = t2r(T_joi_src['lh'])
            """
                Source and Target-specific
            """
            if (src_name=='cmu') and (trgt_name=='g1'): # CMU to G1
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([180,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,0]))
            elif (src_name=='cmu') and (trgt_name=='cr'): # CMU to Common-Rig
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([-90,0,0]))@rpy2r(np.deg2rad([0,0,-90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([-90,0,0]))@rpy2r(np.deg2rad([0,0,-90]))
            elif (src_name=='cmu') and (trgt_name=='humanoid'): # CMU to Humanoid
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,0]))@rpy2r(np.deg2rad([0,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,0]))@rpy2r(np.deg2rad([0,0,0]))

            elif (src_name=='smpl') and (trgt_name=='jf3'): # SMPL to JF3
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,180]))@rpy2r(np.deg2rad([90,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,0,0]))
            elif (src_name=='smpl') and (trgt_name=='baymax'): # SMPL to Baymax
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,0]))
            elif (src_name=='smpl') and (trgt_name=='romeo'): # SMPL to romeo
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,-90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,90]))
            elif (src_name=='smpl') and (trgt_name=='gundam'): # SMPL to gundam
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,90,60])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,-90,0]))
            elif (src_name=='smpl') and (trgt_name=='icub'): # SMPL to icub
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([90,0,90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,0,90]))
            elif (src_name=='smpl') and (trgt_name=='jaxon'): # SMPL to jaxon
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([-90,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,0,0]))
            elif (src_name=='smpl') and (trgt_name=='poppy'): # SMPL to poppy
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([180,90,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,-90,0]))
            elif (src_name=='smpl') and (trgt_name=='valkyrie'): # SMPL to valkyrie
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,90,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,90,0]))
            elif (src_name=='smpl') and (trgt_name=='armar6'): # SMPL to armar6
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,-90,90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([-90,-90,0]))
            elif (src_name=='smpl') and (trgt_name=='apollo'): # SMPL to apollo
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([-90,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,0,0]))
            elif (src_name=='smpl') and (trgt_name=='robonaut2'): # SMPL to robonaut2
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([180,0,-90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([180,0,90]))

            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rh'],R_trgt=R_rh_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lh'],R_trgt=R_lh_trgt)
        if ik_R_foot:
            R_rf_trgt = t2r(T_joi_src['rf'])
            R_lf_trgt = t2r(T_joi_src['lf'])

            if (src_name=='smpl') and (trgt_name=='g1'): # CMU to G1
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,0,0]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,0,0]))
            elif (src_name=='smpl') and (trgt_name=='digit'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,-90,-90]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,-90,90]))
            elif (src_name=='smpl') and (trgt_name=='gundam'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,0,90]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,0,90]))
            elif (src_name=='smpl') and (trgt_name=='poppy'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([90,0,90]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([90,0,90]))
            elif (src_name=='smpl') and (trgt_name=='armar4'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,90,180]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,-90,90]))
            elif (src_name=='smpl') and (trgt_name=='bruce'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([90,90,0]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([90,90,0]))
            elif (src_name=='smpl') and (trgt_name=='themis'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,90,0]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,90,0]))
            elif (src_name=='smpl') and (trgt_name=='x1'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([-90,0,-90]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([90,0,90]))
            elif (src_name=='smpl') and (trgt_name=='robonaut2'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,180,0]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,180,0]))

            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rf'],R_trgt=R_rf_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lf'],R_trgt=R_lf_trgt)
        # Solve IK for the Lower Bodies
        ik_err_norm_list = []
        for ik_tick in range(max_ik_tick): # ik loop
            dq,ik_err_stack = get_dq_from_ik_info(
                env            = env,
                ik_info        = ik_info_whole_body,
                stepsize       = ik_stepsize,
                eps            = 1e-3,
                th             = np.radians(10.0),
                joint_idxs_jac = joint_with_base_idxs_jac,
            ) # dq:[43]
            qpos = env.get_qpos() # get current joint position  [44]
            mujoco.mj_integratePos(env.model,qpos,dq,1)
            env.forward(q=qpos)
            ik_err_norm = np.linalg.norm(ik_err_stack)
            ik_err_norm_list.append(ik_err_norm)
            if ik_err_norm < ik_err_th: break
        if print_ik_failure and (ik_err_norm>=ik_err_th):
            print ("Lower Body IK Error Too High: tick:[%d] ik_err_norm:[%.2e] ik_err_th:[%.2e]"%
                   (tick,ik_err_norm,ik_err_th))

        # Append joint position
        qpos_dir_list.append(env.get_qpos())

        if trgt_name == 'mahru':
            qpos_dir_list[tick][18] = 1.0
            qpos_dir_list[tick][22] = -1.0
        
        # Render
        if animate and (tick%render_every==0 or tick==0):
            chain_src.plot_chain_mujoco(
                env             = env,
                plot_joint      = False,
                plot_joint_name = False,
                r_link          = 0.0025,
                rgba_link       = (1,0,0,0.5),
            ) # source link with red lines
            env.plot_links_between_bodies(
                r    = 0.0025,
                rgba = (0,0,1,0.5),
            ) # plot target with blue
            env.plot_T()
            env.plot_joint_axis(axis_len=0.01,axis_r=0.001)
            env.plot_contact_info()
            plot_ik_info(env=env,ik_info=ik_info_whole_body)
            if plot_joi_src:
                """
                    Source-specific
                """
                if src_name == 'cmu':
                    T_joi_src = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
                for key in T_joi_src.keys():
                    env.plot_T(T=T_joi_src[key],plot_axis=True,axis_len=0.05,axis_width=0.005)
            if plot_joi_trgt:
                """
                    Target-specific
                """
                if override is not None:
                    T_joi_trgt = get_T_joi_trgt(env,trgt_name,override=override)
                else:
                    T_joi_trgt = get_T_joi_trgt(env,trgt_name)
                for key in T_joi_trgt.keys():
                    env.plot_T(T=T_joi_trgt[key],plot_axis=True,axis_len=0.05,axis_width=0.005)
                    
            env.render()
            
        # Print process
        if verbose:
            if ((tick%print_every)==0) or (tick==(L-1)):
                print ("[%d/%d] [%.1f]%% ik_err_norm:[%.2e] ik_err_th:[%.2e]"%
                       (tick,L,100*tick/L,ik_err_norm,ik_err_th))
                
    # Close viewer
    if animate: env.close_viewer()  
    
    # Return
    return qpos_dir_list

"""Would be nice to implement (for now just a copy of directional_preretarget_yoon)"""
def directional_preretarget_and_procrustes(
        env,
        secs,
        chains,
        ik_p_spine=False,
        ik_p_neck=False,
        leg_fixed=False,
        src_name           = 'cmu',
        trgt_name          = 'g1',
        joint_names_for_ik = None,
        max_ik_tick        = 100, # maximum IK iteration
        ik_stepsize        = 1, # IK stepsize
        ik_err_th          = 1e-3, # IK error threshold
        print_ik_failure   = False,
        ik_R_hand          = False,
        ik_R_foot          = True,
        reset_env          = True,
        animate            = True,
        plot_joi_src       = False,
        plot_joi_trgt      = False,
        render_every       = 1,
        verbose            = True,
        print_every        = 10,
        pause_each_tick    = False,
        post_rigging       = True,
        knee_unbend_rate   = 0,
        override           = None,
    ):
    """ 
        Directional pre-retargeting (aka unit-vector method)
        Uses directional pre-retargeting, generalized so that it works across many robots
        get_joi_body_name_of_() for joi_body_name_trgt, or for T_joi_trgt
        moved get_T_joi_of_() out. JOI references are also organized through __init__.py
        for how __init__.py is used, see https://www.youtube.com/watch?v=8FceAhTVGeo
        Return: (qpos_dir_list)
    """
    
    # Reset env
    if reset_env:
        env.reset(step=True)
        
    # Initialize viewer
    if animate:
        env.init_viewer(
            title       = 'Directional Pre-Retargeting Process',
            transparent = True,
        )
    
    # Initialize IK configuration
    if joint_names_for_ik is None: 
        joint_names_for_ik = env.rev_joint_names 
    joint_idxs_jac = env.get_idxs_jac(joint_names=joint_names_for_ik)
    joint_with_base_idxs_jac = np.concatenate(([0],joint_idxs_jac))
    
    # Target JOI body name
    """
        Target-specific
    """
    if override is not None:
        joi_body_name_trgt = get_joi_body_name_trgt(trgt_name,override=override)
        T_joi_trgt = get_T_joi_trgt(env,trgt_name,override=override)
    else:
        joi_body_name_trgt = get_joi_body_name_trgt(trgt_name)
        T_joi_trgt = get_T_joi_trgt(env,trgt_name)
    if trgt_name == 'baymax' or trgt_name == 'jf3' or trgt_name == 'auto_gen' or leg_fixed:
        # Get target lengths
        len_base2spine = len_T_joi(T_joi_trgt,'base','spine')
        len_spine2neck = len_T_joi(T_joi_trgt,'spine','neck')
        len_neck2rs    = len_T_joi(T_joi_trgt,'neck','rs')
        len_rs2re      = len_T_joi(T_joi_trgt,'rs','re')
        len_re2rw      = len_T_joi(T_joi_trgt,'re','rw')
        len_neck2ls    = len_T_joi(T_joi_trgt,'neck','ls')
        len_ls2le      = len_T_joi(T_joi_trgt,'ls','le')
        len_le2lw      = len_T_joi(T_joi_trgt,'le','lw')
    else:
        # Get target lengths
        len_base2rp    = len_T_joi(T_joi_trgt,'base','rp')
        len_rp2rk      = len_T_joi(T_joi_trgt,'rp','rk')
        len_rk2ra      = len_T_joi(T_joi_trgt,'rk','ra')
        len_base2lp    = len_T_joi(T_joi_trgt,'base','lp')
        len_lp2lk      = len_T_joi(T_joi_trgt,'lp','lk')
        len_lk2la      = len_T_joi(T_joi_trgt,'lk','la')
        len_base2spine = len_T_joi(T_joi_trgt,'base','spine')
        len_spine2neck = len_T_joi(T_joi_trgt,'spine','neck')
        len_neck2rs    = len_T_joi(T_joi_trgt,'neck','rs')
        len_rs2re      = len_T_joi(T_joi_trgt,'rs','re')
        len_re2rw      = len_T_joi(T_joi_trgt,'re','rw')
        len_neck2ls    = len_T_joi(T_joi_trgt,'neck','ls')
        len_ls2le      = len_T_joi(T_joi_trgt,'ls','le')
        len_le2lw      = len_T_joi(T_joi_trgt,'le','lw')
        
    # Loop
    qpos_dir_list = []
    L = len(secs)
    for tick in range(L): # for each tick
        
        # Pause each tick (for debugging)
        if pause_each_tick:
            env.viewer_pause()
        
        # Source chain
        sec = secs[tick]
        chain_src = chains[tick]
        """
            Source-specific
        """
        if src_name == 'cmu':
            T_joi_src = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
        elif src_name == 'smpl':
            T_joi_src = get_T_joi_of_chain_smpl(chain_src,base_between_pelvis=True)
        
        # Move target base (once)
        if tick == 0:
            T_base_trgt  = T_joi_src['base']
            """
                Source-specific
            """
            if src_name == 'cmu': # for CMU-MoCap (y-up z-front), make it z-up x-front
                T_base_trgt = T_yuzf2zuxf(T_base_trgt)
            if trgt_name == 'humanoid' or trgt_name == 'baymax' or trgt_name == 'jf3' or trgt_name == 'mahru' or trgt_name =='auto_gen' or leg_fixed:
                pass
            else:
                env.set_T_base_body(body_name=joi_body_name_trgt['root'],T=T_base_trgt)
            env.forward()

        # Get unit vectors of the source rig
        uv_base2spine = uv_T_joi(T_joi_src,'base','spine')
        uv_spine2neck = uv_T_joi(T_joi_src,'spine','neck')
        uv_neck2rs    = uv_T_joi(T_joi_src,'neck','rs')
        uv_rs2re      = uv_T_joi(T_joi_src,'rs','re')
        uv_re2rw      = uv_T_joi(T_joi_src,'re','rw')
        uv_neck2ls    = uv_T_joi(T_joi_src,'neck','ls')
        uv_ls2le      = uv_T_joi(T_joi_src,'ls','le')
        uv_le2lw      = uv_T_joi(T_joi_src,'le','lw')
        uv_base2rp    = uv_T_joi(T_joi_src,'base','rp')
        uv_rp2rk      = uv_T_joi(T_joi_src,'rp','rk')
        uv_rk2ra      = uv_T_joi(T_joi_src,'rk','ra')
        uv_base2lp    = uv_T_joi(T_joi_src,'base','lp')
        uv_lp2lk      = uv_T_joi(T_joi_src,'lp','lk')
        uv_lk2la      = uv_T_joi(T_joi_src,'lk','la')

        # Set positional targets
        """
            Target-specific
        """
        if override is not None:

            T_joi_trgt = get_T_joi_trgt(env,trgt_name,override=override)
        else:

            T_joi_trgt = get_T_joi_trgt(env,trgt_name)
        
        # Set IK targets 
        ik_info_whole_body = init_ik_info()

        if post_rigging:
            if trgt_name == 'baymax' or trgt_name == 'jf3' or trgt_name == 'auto_gen':
                # Uprighting
                root_yaw_range = np.deg2rad([-0.1, 0.1])
                yaw_min_rad = root_yaw_range[0]
                yaw_max_rad = root_yaw_range[1]
                p_rp = t2p(T_joi_src['rp'])
                p_lp = t2p(T_joi_src['lp'])
                pelvis_rad = np.arctan2(p_rp[1]-p_lp[1],p_rp[0]-p_lp[0])
                yaw_rad = pelvis_rad + np.deg2rad(90)
                if yaw_rad > np.deg2rad(180):
                    yaw_rad = yaw_rad - np.deg2rad(360)
                elif yaw_rad < np.deg2rad(-180):
                    yaw_rad = yaw_rad + np.deg2rad(360)
                if yaw_rad < yaw_min_rad:
                    yaw_offset_rad = yaw_min_rad - yaw_rad
                elif yaw_rad > yaw_max_rad:
                    yaw_offset_rad = yaw_max_rad - yaw_rad
                else:
                    yaw_offset_rad = 0
                R_rot = rpy2r([0,0,yaw_offset_rad])
                chain_src.set_root_joint_R(R_rot@t2r(T_joi_src['base']))
                chain_src.forward_kinematics()

                if src_name == 'cmu':
                    T_joi_upright = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
                elif src_name == 'smpl':
                    T_joi_upright = get_T_joi_of_chain_smpl(chain_src,base_between_pelvis=True)

                # Upper body position targets
                p_base_trgt = t2p(T_joi_trgt['base'])
                p_spine_trgt = p_base_trgt + len_base2spine*uv_base2spine
                p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                if ik_p_spine: # spine target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
                if ik_p_neck: # neck target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
            else:
                # Uprighting
                root_yaw_range = np.deg2rad([-30, 30])
                yaw_min_rad = root_yaw_range[0]
                yaw_max_rad = root_yaw_range[1]
                p_rp = t2p(T_joi_src['rp'])
                p_lp = t2p(T_joi_src['lp'])
                pelvis_rad = np.arctan2(p_rp[1]-p_lp[1],p_rp[0]-p_lp[0])
                yaw_rad = pelvis_rad + np.deg2rad(90)
                if yaw_rad > np.deg2rad(180):
                    yaw_rad = yaw_rad - np.deg2rad(360)
                elif yaw_rad < np.deg2rad(-180):
                    yaw_rad = yaw_rad + np.deg2rad(360)
                if yaw_rad < yaw_min_rad:
                    yaw_offset_rad = yaw_min_rad - yaw_rad
                elif yaw_rad > yaw_max_rad:
                    yaw_offset_rad = yaw_max_rad - yaw_rad
                else:
                    yaw_offset_rad = 0
                R_rot = rpy2r([0,0,yaw_offset_rad])
                chain_src.set_root_joint_R(R_rot@t2r(T_joi_src['base']))
                chain_src.forward_kinematics()

                if src_name == 'cmu':
                    T_joi_upright = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
                elif src_name == 'smpl':
                    T_joi_upright = get_T_joi_of_chain_smpl(chain_src,base_between_pelvis=True)

                # Post-rigging
                p_base_upright = t2p(T_joi_upright['base'])
                p_rp_upright  = t2p(T_joi_upright['rp'])
                p_rk_upright   = t2p(T_joi_upright['rk'])
                p_ra_upright   = t2p(T_joi_upright['ra'])
                p_la_upright   = t2p(T_joi_upright['la'])
                # Set right and left ankle positions (fix the width between the feet)
                p_ca_z = 0.5*(p_ra_upright[2] + p_la_upright[2])
                p_ca   = [p_base_upright[0], p_base_upright[1], p_ca_z]
                if tick == 0:
                    rs2ls_dist = np.linalg.norm(t2p(T_joi_src['rs']) - t2p(T_joi_src['ls']))
                p_offset_ca2ra = 0.5*rs2ls_dist*np.array([0, -1, 0])
                p_offset_ca2la = 0.5*rs2ls_dist*np.array([0,1,0])
                p_ra_post = p_ca + p_offset_ca2ra
                p_la_post = p_ca + p_offset_ca2la
                # Set right and left knee positions (based on updated ankle positions)
                p_rk_post = p_ra_post
                p_lk_post = p_la_post
                p_rk_y = p_rk_post[1]
                p_lk_y = p_lk_post[1]
                p_ck_y = 0.5*(p_rk_y + p_lk_y)
                rp2lp_dist = np.linalg.norm(t2p(T_joi_upright['rp']) - t2p(T_joi_upright['lp']))
                p_rk_post[1] = p_ck_y +np.sqrt(rp2lp_dist/rs2ls_dist)*(p_rk_y - p_ck_y)
                p_lk_post[1] = p_ck_y +np.sqrt(rp2lp_dist/rs2ls_dist)*(p_lk_y - p_ck_y)
                a2k_dist = np.linalg.norm(p_ra_upright - p_rk_upright)
                k2p_dist = np.linalg.norm(p_rk_upright - p_rp_upright)
                p_rk_post[2] = p_rk_post[2] + a2k_dist
                p_lk_post[2] = p_lk_post[2] + a2k_dist
                # Push knee x-positions
                p_rk_post[0] = p_rk_post[0] + 0.1
                p_lk_post[0] = p_lk_post[0] + 0.1
                # Avoid too much knee bending
                a2p_dist   = a2k_dist + k2p_dist
                p2a_z_dist = p_rp_upright[2]-p_ra_post[2]
                z_diff4a   = a2p_dist - p2a_z_dist + 0.1 # 0.1 is a margin
                torso_x    = p_base_upright[0]
                torso_z    = p_base_upright[2]

                # Modify IK targets of x- and z-positions of ankles to reflect 'knee_unbend_rate'
                p_ra_post[0] = (1-knee_unbend_rate)*p_ra_post[0] + knee_unbend_rate*torso_x
                p_la_post[0] = (1-knee_unbend_rate)*p_la_post[0] + knee_unbend_rate*torso_x
                p_ra_post[2] = p_ra_post[2] - knee_unbend_rate*z_diff4a
                p_la_post[2] = p_la_post[2] - knee_unbend_rate*z_diff4a
                # Modify IK target of x- and z-position of knees
                z_knee_unbend = 0.5*(torso_z +p_ra_post[2])
                p_rk_post[0] = (1-knee_unbend_rate)*p_rk_post[0] + knee_unbend_rate*torso_x
                p_lk_post[0] = (1-knee_unbend_rate)*p_lk_post[0] + knee_unbend_rate*torso_x
                p_rk_post[2] = (1-knee_unbend_rate)*p_rk_post[2] + knee_unbend_rate*z_knee_unbend
                p_lk_post[2] = (1-knee_unbend_rate)*p_lk_post[2] + knee_unbend_rate*z_knee_unbend

                # Get unit vectors of the source rig
                uv_base2spine = uv_T_joi(T_joi_upright,'base','spine')
                uv_spine2neck = uv_T_joi(T_joi_upright,'spine','neck')
                uv_neck2rs    = uv_T_joi(T_joi_upright,'neck','rs')
                uv_rs2re      = uv_T_joi(T_joi_upright,'rs','re')
                uv_re2rw      = uv_T_joi(T_joi_upright,'re','rw')
                uv_neck2ls    = uv_T_joi(T_joi_upright,'neck','ls')
                uv_ls2le      = uv_T_joi(T_joi_upright,'ls','le')
                uv_le2lw      = uv_T_joi(T_joi_upright,'le','lw')
                uv_base2rp    = uv_T_joi(T_joi_upright,'base','rp')
                uv_base2lp    = uv_T_joi(T_joi_upright,'base','lp')

                # Lower body position targets
                p_rp_trgt   = p_base_upright + len_base2rp*uv_base2rp
                p_rk_trgt   = p_rk_post
                p_ra_trgt   = p_ra_post
                p_lp_trgt   = p_base_upright + len_base2lp*uv_base2lp
                p_lk_trgt   = p_lk_post
                p_la_trgt   = p_la_post
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rp'],p_trgt=p_rp_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lp'],p_trgt=p_lp_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rk'],p_trgt=p_rk_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lk'],p_trgt=p_lk_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ra'],p_trgt=p_ra_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['la'],p_trgt=p_la_trgt)

                # Upper body position targets
                p_spine_trgt = p_base_upright + len_base2spine*uv_base2spine
                p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                if ik_p_spine: # spine target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
                if ik_p_neck: # neck target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
            
        else:
            if trgt_name == 'baymax' or trgt_name == 'jf3' or trgt_name == 'auto_gen' or leg_fixed:
                # Upper body position targets
                p_base_trgt = t2p(T_joi_trgt['base'])
                p_spine_trgt = p_base_trgt + len_base2spine*uv_base2spine
                p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                if ik_p_spine: # spine target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
                if ik_p_neck: # neck target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
            else:
                # Lower body position targets
                p_base_trgt = t2p(T_joi_src['base'])
                p_rp_trgt   = p_base_trgt + len_base2rp*uv_base2rp
                p_rk_trgt   = p_rp_trgt + len_rp2rk*uv_rp2rk
                p_ra_trgt   = p_rk_trgt + len_rk2ra*uv_rk2ra
                p_lp_trgt   = p_base_trgt + len_base2lp*uv_base2lp
                p_lk_trgt   = p_lp_trgt + len_lp2lk*uv_lp2lk
                p_la_trgt   = p_lk_trgt + len_lk2la*uv_lk2la
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rp'],p_trgt=p_rp_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lp'],p_trgt=p_lp_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rk'],p_trgt=p_rk_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lk'],p_trgt=p_lk_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ra'],p_trgt=p_ra_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['la'],p_trgt=p_la_trgt)
                
                # Upper body position targets
                p_spine_trgt = p_base_trgt + len_base2spine*uv_base2spine
                p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                if ik_p_spine: # spine target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
                if ik_p_neck: # neck target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
            
        # Hand rotation targets
        if ik_R_hand:
            R_rh_trgt = t2r(T_joi_src['rh'])
            R_lh_trgt = t2r(T_joi_src['lh'])
            """
                Source and Target-specific
            """
            if (src_name=='cmu') and (trgt_name=='g1'): # CMU to G1
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([180,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,0]))
            elif (src_name=='cmu') and (trgt_name=='cr'): # CMU to Common-Rig
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([-90,0,0]))@rpy2r(np.deg2rad([0,0,-90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([-90,0,0]))@rpy2r(np.deg2rad([0,0,-90]))
            elif (src_name=='cmu') and (trgt_name=='humanoid'): # CMU to Humanoid
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,0]))@rpy2r(np.deg2rad([0,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,0]))@rpy2r(np.deg2rad([0,0,0]))

            elif (src_name=='smpl') and (trgt_name=='jf3'): # SMPL to JF3
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,180]))@rpy2r(np.deg2rad([90,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,0,0]))
            elif (src_name=='smpl') and (trgt_name=='baymax'): # SMPL to Baymax
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,0]))
            elif (src_name=='smpl') and (trgt_name=='romeo'): # SMPL to romeo
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,-90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,90]))
            elif (src_name=='smpl') and (trgt_name=='gundam'): # SMPL to gundam
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,90,60])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,-90,0]))
            elif (src_name=='smpl') and (trgt_name=='icub'): # SMPL to icub
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([90,0,90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,0,90]))
            elif (src_name=='smpl') and (trgt_name=='jaxon'): # SMPL to jaxon
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([-90,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,0,0]))
            elif (src_name=='smpl') and (trgt_name=='poppy'): # SMPL to poppy
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([180,90,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,-90,0]))
            elif (src_name=='smpl') and (trgt_name=='valkyrie'): # SMPL to valkyrie
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,90,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,90,0]))
            elif (src_name=='smpl') and (trgt_name=='armar6'): # SMPL to armar6
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,-90,90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([-90,-90,0]))
            elif (src_name=='smpl') and (trgt_name=='apollo'): # SMPL to apollo
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([-90,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,0,0]))
            elif (src_name=='smpl') and (trgt_name=='robonaut2'): # SMPL to robonaut2
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([180,0,-90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([180,0,90]))

            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rh'],R_trgt=R_rh_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lh'],R_trgt=R_lh_trgt)
        if ik_R_foot:
            R_rf_trgt = t2r(T_joi_src['rf'])
            R_lf_trgt = t2r(T_joi_src['lf'])

            if (src_name=='smpl') and (trgt_name=='g1'): # CMU to G1
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,0,0]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,0,0]))
            elif (src_name=='smpl') and (trgt_name=='digit'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,-90,-90]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,-90,90]))
            elif (src_name=='smpl') and (trgt_name=='gundam'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,0,90]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,0,90]))
            elif (src_name=='smpl') and (trgt_name=='poppy'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([90,0,90]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([90,0,90]))
            elif (src_name=='smpl') and (trgt_name=='armar4'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,90,180]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,-90,90]))
            elif (src_name=='smpl') and (trgt_name=='bruce'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([90,90,0]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([90,90,0]))
            elif (src_name=='smpl') and (trgt_name=='themis'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,90,0]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,90,0]))
            elif (src_name=='smpl') and (trgt_name=='x1'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([-90,0,-90]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([90,0,90]))
            elif (src_name=='smpl') and (trgt_name=='robonaut2'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,180,0]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,180,0]))

            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rf'],R_trgt=R_rf_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lf'],R_trgt=R_lf_trgt)
        # Solve IK for the Lower Bodies
        ik_err_norm_list = []
        for ik_tick in range(max_ik_tick): # ik loop
            dq,ik_err_stack = get_dq_from_ik_info(
                env            = env,
                ik_info        = ik_info_whole_body,
                stepsize       = ik_stepsize,
                eps            = 1e-3,
                th             = np.radians(10.0),
                joint_idxs_jac = joint_with_base_idxs_jac,
            ) # dq:[43]
            qpos = env.get_qpos() # get current joint position  [44]
            mujoco.mj_integratePos(env.model,qpos,dq,1)
            env.forward(q=qpos)
            ik_err_norm = np.linalg.norm(ik_err_stack)
            ik_err_norm_list.append(ik_err_norm)
            if ik_err_norm < ik_err_th: break
        if print_ik_failure and (ik_err_norm>=ik_err_th):
            print ("Lower Body IK Error Too High: tick:[%d] ik_err_norm:[%.2e] ik_err_th:[%.2e]"%
                   (tick,ik_err_norm,ik_err_th))

        # Append joint position
        qpos_dir_list.append(env.get_qpos())

        if trgt_name == 'mahru':
            qpos_dir_list[tick][18] = 1.0
            qpos_dir_list[tick][22] = -1.0
        
        # Render
        if animate and (tick%render_every==0 or tick==0):
            chain_src.plot_chain_mujoco(
                env             = env,
                plot_joint      = False,
                plot_joint_name = False,
                r_link          = 0.0025,
                rgba_link       = (1,0,0,0.5),
            ) # source link with red lines
            env.plot_links_between_bodies(
                r    = 0.0025,
                rgba = (0,0,1,0.5),
            ) # plot target with blue
            env.plot_T()
            env.plot_text(p=np.array([0,0,1]),label='tick:[%d/%d time:[%.3f]sec'%(tick,L,sec))
            env.plot_joint_axis(axis_len=0.01,axis_r=0.001)
            env.plot_contact_info()
            plot_ik_info(env=env,ik_info=ik_info_whole_body)
            if plot_joi_src:
                """
                    Source-specific
                """
                if src_name == 'cmu':
                    T_joi_src = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
                for key in T_joi_src.keys():
                    env.plot_T(T=T_joi_src[key],plot_axis=True,axis_len=0.05,axis_width=0.005)
            if plot_joi_trgt:
                """
                    Target-specific
                """
                if override is not None:
                    T_joi_trgt = get_T_joi_trgt(env,trgt_name,override=override)
                else:
                    T_joi_trgt = get_T_joi_trgt(env,trgt_name)
                for key in T_joi_trgt.keys():
                    env.plot_T(T=T_joi_trgt[key],plot_axis=True,axis_len=0.05,axis_width=0.005)
                    
            env.render()
            
        # Print process
        if verbose:
            if ((tick%print_every)==0) or (tick==(L-1)):
                print ("[%d/%d] [%.1f]%% ik_err_norm:[%.2e] ik_err_th:[%.2e]"%
                       (tick,L,100*tick/L,ik_err_norm,ik_err_th))
                
    # Close viewer
    if animate: env.close_viewer()  
    
    # Return
    return qpos_dir_list

def directional_preretarget_postrigging(
        env,
        secs,
        chains,
        ik_p_spine=False,
        ik_p_neck=False,
        leg_fixed=False,
        src_name           = 'cmu',
        trgt_name          = 'g1',
        joint_names_for_ik = None,
        max_ik_tick        = 100, # maximum IK iteration
        ik_stepsize        = 1, # IK stepsize
        ik_err_th          = 1e-3, # IK error threshold
        print_ik_failure   = False,
        ik_R_hand          = False,
        ik_R_foot          = False,
        reset_env          = True,
        animate            = True,
        plot_joi_src       = False,
        plot_joi_trgt      = False,
        render_every       = 1,
        verbose            = True,
        print_every        = 10,
        pause_each_tick    = False,
        post_rigging       = True,
        knee_unbend_rate   = 0,
        override           = None,
    ):
    """ 
        Directional pre-retargeting (aka unit-vector method)
        Uses directional pre-retargeting, generalized so that it works across many robots
        get_joi_body_name_of_() for joi_body_name_trgt, or for T_joi_trgt
        moved get_T_joi_of_() out. JOI references are also organized through __init__.py
        for how __init__.py is used, see https://www.youtube.com/watch?v=8FceAhTVGeo
        Return: (qpos_dir_list)
    """

    # Reset env
    if reset_env:
        env.reset(step=True)
        
    # Initialize viewer
    if animate:
        env.init_viewer(
            title       = 'Directional Pre-Retargeting Process',
            transparent = True,
        )
    
    # Initialize IK configuration
    if joint_names_for_ik is None: 
        joint_names_for_ik = env.rev_joint_names 
    joint_idxs_jac = env.get_idxs_jac(joint_names=joint_names_for_ik)
    joint_with_base_idxs_jac = np.concatenate(([0,1,2,3,4,5],joint_idxs_jac))
    
    # Target JOI body name
    """
        Target-specific
    """
    if override is not None:
        joi_body_name_trgt = get_joi_body_name_trgt(trgt_name,override=override)
        T_joi_trgt = get_T_joi_trgt(env,trgt_name,override=override)
    else:
        joi_body_name_trgt = get_joi_body_name_trgt(trgt_name)
        T_joi_trgt = get_T_joi_trgt(env,trgt_name)
    if trgt_name == 'baymax' or trgt_name == 'jf3' or trgt_name == 'auto_gen' or leg_fixed:
        # Get target lengths
        len_base2spine = len_T_joi(T_joi_trgt,'base','spine')
        len_spine2neck = len_T_joi(T_joi_trgt,'spine','neck')
        len_neck2rs    = len_T_joi(T_joi_trgt,'neck','rs')
        len_rs2re      = len_T_joi(T_joi_trgt,'rs','re')
        len_re2rw      = len_T_joi(T_joi_trgt,'re','rw')
        len_neck2ls    = len_T_joi(T_joi_trgt,'neck','ls')
        len_ls2le      = len_T_joi(T_joi_trgt,'ls','le')
        len_le2lw      = len_T_joi(T_joi_trgt,'le','lw')
    else:
        # Get target lengths
        len_base2rp    = len_T_joi(T_joi_trgt,'base','rp')
        len_rp2rk      = len_T_joi(T_joi_trgt,'rp','rk')
        len_rk2ra      = len_T_joi(T_joi_trgt,'rk','ra')
        len_base2lp    = len_T_joi(T_joi_trgt,'base','lp')
        len_lp2lk      = len_T_joi(T_joi_trgt,'lp','lk')
        len_lk2la      = len_T_joi(T_joi_trgt,'lk','la')
        len_base2spine = len_T_joi(T_joi_trgt,'base','spine')
        len_spine2neck = len_T_joi(T_joi_trgt,'spine','neck')
        len_neck2rs    = len_T_joi(T_joi_trgt,'neck','rs')
        len_rs2re      = len_T_joi(T_joi_trgt,'rs','re')
        len_re2rw      = len_T_joi(T_joi_trgt,'re','rw')
        len_neck2ls    = len_T_joi(T_joi_trgt,'neck','ls')
        len_ls2le      = len_T_joi(T_joi_trgt,'ls','le')
        len_le2lw      = len_T_joi(T_joi_trgt,'le','lw')
        
    # Loop
    qpos_dir_list = []
    L = len(secs)
    for tick in range(L): # for each tick
        
        # Pause each tick (for debugging)
        if pause_each_tick:
            env.viewer_pause()
        
        # Source chain
        sec = secs[tick]
        chain_src = chains[tick]
        """
            Source-specific
        """
        if src_name == 'cmu':
            T_joi_src = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
        elif src_name == 'smpl':
            T_joi_src = get_T_joi_of_chain_smpl(chain_src,base_between_pelvis=True)
        
        # Move target base (once)
        if tick == 0:
            T_base_trgt  = T_joi_src['base']
            """
                Source-specific
            """
            if src_name == 'cmu': # for CMU-MoCap (y-up z-front), make it z-up x-front
                T_base_trgt = T_yuzf2zuxf(T_base_trgt)
            if trgt_name == 'humanoid' or trgt_name == 'baymax' or trgt_name == 'jf3'\
                or trgt_name == 'mahru' or trgt_name =='auto_gen' or trgt_name == 'lego_v2'\
                or trgt_name == 'lego_easy' or trgt_name == 'lego_middle' or trgt_name == 'lego_hard' or leg_fixed:
                pass
            else:
                env.set_T_base_body(body_name=joi_body_name_trgt['root'],T=T_base_trgt)
            env.forward()

        # Get unit vectors of the source rig
        uv_base2spine = uv_T_joi(T_joi_src,'base','spine')
        uv_spine2neck = uv_T_joi(T_joi_src,'spine','neck')
        uv_neck2rs    = uv_T_joi(T_joi_src,'neck','rs')
        uv_rs2re      = uv_T_joi(T_joi_src,'rs','re')
        uv_re2rw      = uv_T_joi(T_joi_src,'re','rw')
        uv_neck2ls    = uv_T_joi(T_joi_src,'neck','ls')
        uv_ls2le      = uv_T_joi(T_joi_src,'ls','le')
        uv_le2lw      = uv_T_joi(T_joi_src,'le','lw')
        uv_base2rp    = uv_T_joi(T_joi_src,'base','rp')
        uv_rp2rk      = uv_T_joi(T_joi_src,'rp','rk')
        uv_rk2ra      = uv_T_joi(T_joi_src,'rk','ra')
        uv_base2lp    = uv_T_joi(T_joi_src,'base','lp')
        uv_lp2lk      = uv_T_joi(T_joi_src,'lp','lk')
        uv_lk2la      = uv_T_joi(T_joi_src,'lk','la')

        # Set positional targets
        """
            Target-specific
        """
        if override is not None:
            T_joi_trgt = get_T_joi_trgt(env,trgt_name,override=override)
        else:
            T_joi_trgt = get_T_joi_trgt(env,trgt_name)
        
        # Set IK targets 
        ik_info_whole_body = init_ik_info()


        if post_rigging:
            if trgt_name == 'baymax' or trgt_name == 'jf3' or trgt_name == 'auto_gen' \
                or leg_fixed:
                # Uprighting
                root_yaw_range = np.deg2rad([-0.1, 0.1])
                yaw_min_rad = root_yaw_range[0]
                yaw_max_rad = root_yaw_range[1]
                p_rp = t2p(T_joi_src['rp'])
                p_lp = t2p(T_joi_src['lp'])
                pelvis_rad = np.arctan2(p_rp[1]-p_lp[1],p_rp[0]-p_lp[0])
                yaw_rad = pelvis_rad + np.deg2rad(90)
                if yaw_rad > np.deg2rad(180):
                    yaw_rad = yaw_rad - np.deg2rad(360)
                elif yaw_rad < np.deg2rad(-180):
                    yaw_rad = yaw_rad + np.deg2rad(360)
                if yaw_rad < yaw_min_rad:
                    yaw_offset_rad = yaw_min_rad - yaw_rad
                elif yaw_rad > yaw_max_rad:
                    yaw_offset_rad = yaw_max_rad - yaw_rad
                else:
                    yaw_offset_rad = 0
                R_rot = rpy2r([0,0,yaw_offset_rad])
                chain_src_upright = copy.deepcopy(chain_src)

                if src_name == 'cmu':
                    R_base = t2r(T_joi_src['base'])
                    yaw = np.arctan2(R_base[1, 0], R_base[0, 0])
                    pitch = np.arctan2(-R_base[2, 0], np.sqrt(R_base[2, 1]**2 + R_base[2, 2]**2))
                    roll = np.arctan2(R_base[2, 1], R_base[2, 2])
                    R_base_modified = rpy2r([roll, pitch, yaw])
                    chain_src_upright.set_root_joint_R(R_rot@R_base_modified)
                elif src_name == 'smpl':
                    R_base_modified = rpy2r([0, 0, 0])
                    chain_src_upright.set_root_joint_R(R_base_modified)

                chain_src_upright.forward_kinematics()
                if src_name == 'cmu':
                    T_joi_upright = get_T_joi_of_chain_cmu(chain_src_upright,base_between_pelvis=True)
                elif src_name == 'smpl':
                    T_joi_upright = get_T_joi_of_chain_smpl(chain_src_upright,base_between_pelvis=True)

                # Get unit vectors of the source rig
                uv_base2spine = uv_T_joi(T_joi_upright,'base','spine')
                uv_spine2neck = uv_T_joi(T_joi_upright,'spine','neck')
                uv_neck2rs    = uv_T_joi(T_joi_upright,'neck','rs')
                uv_rs2re      = uv_T_joi(T_joi_upright,'rs','re')
                uv_re2rw      = uv_T_joi(T_joi_upright,'re','rw')
                uv_neck2ls    = uv_T_joi(T_joi_upright,'neck','ls')
                uv_ls2le      = uv_T_joi(T_joi_upright,'ls','le')
                uv_le2lw      = uv_T_joi(T_joi_upright,'le','lw')


                # Upper body position targets
                p_base_trgt = t2p(T_joi_trgt['base'])
                p_spine_trgt = p_base_trgt + len_base2spine*uv_base2spine
                p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                if ik_p_spine: # spine target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
                if ik_p_neck: # neck target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
            else:
                # Uprighting
                root_yaw_range = np.deg2rad([-0.01, 0.01])
                yaw_min_rad = root_yaw_range[0]
                yaw_max_rad = root_yaw_range[1]
                p_rp = t2p(T_joi_src['rp'])
                p_lp = t2p(T_joi_src['lp'])
                pelvis_rad = np.arctan2(p_rp[1]-p_lp[1],p_rp[0]-p_lp[0])
                yaw_rad = pelvis_rad + np.deg2rad(90)
                if yaw_rad > np.deg2rad(180):
                    yaw_rad = yaw_rad - np.deg2rad(360)
                elif yaw_rad < np.deg2rad(-180):
                    yaw_rad = yaw_rad + np.deg2rad(360)
                if yaw_rad < yaw_min_rad:
                    yaw_offset_rad = yaw_min_rad - yaw_rad
                elif yaw_rad > yaw_max_rad:
                    yaw_offset_rad = yaw_max_rad - yaw_rad
                else:
                    yaw_offset_rad = 0
                
                R_rot = rpy2r([0,0,yaw_offset_rad])
                chain_src_upright = copy.deepcopy(chain_src)

                if src_name == 'cmu':
                    R_base = t2r(T_joi_src['base'])
                    yaw = np.arctan2(R_base[1, 0], R_base[0, 0])
                    pitch = np.arctan2(-R_base[2, 0], np.sqrt(R_base[2, 1]**2 + R_base[2, 2]**2))
                    roll = np.arctan2(R_base[2, 1], R_base[2, 2])
                    R_base_modified = rpy2r([roll, pitch, yaw])
                    chain_src_upright.set_root_joint_R(R_rot@R_base_modified)
                elif src_name == 'smpl':
                    R_base_modified = rpy2r([0, 0, 0])
                    chain_src_upright.set_root_joint_R(R_base_modified)

                chain_src_upright.forward_kinematics()
                if src_name == 'cmu':
                    T_joi_upright = get_T_joi_of_chain_cmu(chain_src_upright,base_between_pelvis=True)
                elif src_name == 'smpl':
                    T_joi_upright = get_T_joi_of_chain_smpl(chain_src_upright,base_between_pelvis=True)
                

                if tick == 0:
                    base2ankle_dist_source = np.linalg.norm(t2p(T_joi_src['base']) - t2p(T_joi_src['ra']))
                    base2ankle_dist_target = np.linalg.norm(t2p(T_joi_trgt['base']) - t2p(T_joi_trgt['ra']))
                    scale_factor = base2ankle_dist_target / base2ankle_dist_source
                # Post-rigging
                p_base_upright = t2p(T_joi_upright['base'])
                p_base_upright_scaled = p_base_upright * scale_factor
                p_base_trgt = t2p(T_joi_trgt['base'])
                
                # Fix base X position to prevent front-back swaying, keep original Y position
                if tick == 0:
                    base_x_fixed = p_base_trgt[0]  # Fix X position at first frame
                    base_y_fixed = p_base_trgt[1]  # Fix Y position at first frame
                p_base_trgt[0] = base_x_fixed  # Prevent front-back movement
                p_base_trgt[1] = base_y_fixed  # Prevent left-right movement  
                p_base_trgt[2] = p_base_upright_scaled[2]  # Keep dynamic height
                p_ra_trgt_ref = t2p(T_joi_trgt['ra'])  # Target robot right ankle reference
                p_la_trgt_ref = t2p(T_joi_trgt['la'])  # Target robot left ankle reference
                if tick == 0:
                    ankle_x_fixed = 0.5 * (p_ra_trgt_ref[0] + p_la_trgt_ref[0])   # Fixed X position (center of target robot ankles)
                    ankle_z_ground = min(p_ra_trgt_ref[2], p_la_trgt_ref[2])      # Ground level from target robot
                p_ra_post = np.array([ankle_x_fixed, p_ra_trgt_ref[1], ankle_z_ground])  # Right foot: original Y
                p_la_post = np.array([ankle_x_fixed, p_la_trgt_ref[1], ankle_z_ground])  # Left foot: original Y
                p_rk_post = p_ra_post.copy()
                p_lk_post = p_la_post.copy()
                
                # Push knees forward slightly for natural stance
                a2k_dist = np.linalg.norm(t2p(T_joi_trgt['ra']) - t2p(T_joi_trgt['rk']))
                k2p_dist = np.linalg.norm(t2p(T_joi_trgt['rk']) - t2p(T_joi_trgt['rp']))
                p_rk_post[2] = p_rk_post[2] + a2k_dist
                p_lk_post[2] = p_lk_post[2] + a2k_dist
                # Push knee x-positions
                knee_forward_offset = 0.07
                p_rk_post[0] = p_rk_post[0] + knee_forward_offset
                p_lk_post[0] = p_lk_post[0] + knee_forward_offset

                # Get unit vectors of the source rig
                uv_base2spine = uv_T_joi(T_joi_upright,'base','spine')
                uv_spine2neck = uv_T_joi(T_joi_upright,'spine','neck')
                uv_neck2rs    = uv_T_joi(T_joi_upright,'neck','rs')
                uv_rs2re      = uv_T_joi(T_joi_upright,'rs','re')
                uv_re2rw      = uv_T_joi(T_joi_upright,'re','rw')
                uv_neck2ls    = uv_T_joi(T_joi_upright,'neck','ls')
                uv_ls2le      = uv_T_joi(T_joi_upright,'ls','le')
                uv_le2lw      = uv_T_joi(T_joi_upright,'le','lw')
                uv_base2rp    = uv_T_joi(T_joi_upright,'base','rp')
                uv_base2lp    = uv_T_joi(T_joi_upright,'base','lp')

                # Lower body position targets
                p_rp_trgt   = p_base_trgt + len_base2rp*uv_base2rp
                p_rk_trgt   = p_rk_post
                p_ra_trgt   = p_ra_post
                p_lp_trgt   = p_base_trgt + len_base2lp*uv_base2lp
                p_lk_trgt   = p_lk_post
                p_la_trgt   = p_la_post
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rp'],p_trgt=p_rp_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lp'],p_trgt=p_lp_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rk'],p_trgt=p_rk_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lk'],p_trgt=p_lk_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ra'],p_trgt=p_ra_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['la'],p_trgt=p_la_trgt)
                
                # Add foot rotation targets (flat on ground, parallel to ground plane)
                R_foot_flat = np.eye(3)  # Identity = flat orientation aligned with ground
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rf'],R_trgt=R_foot_flat)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lf'],R_trgt=R_foot_flat)

                # Upper body position targets
                p_spine_trgt = p_base_trgt + len_base2spine*uv_base2spine
                p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                if ik_p_spine: # spine target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
                if ik_p_neck: # neck target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
            
        else:
            if trgt_name == 'baymax' or trgt_name == 'jf3' or trgt_name == 'auto_gen' or leg_fixed:
                # Get unit vectors of the source rig
                uv_base2spine = uv_T_joi(T_joi_src,'base','spine')
                uv_spine2neck = uv_T_joi(T_joi_src,'spine','neck')
                uv_neck2rs    = uv_T_joi(T_joi_src,'neck','rs')
                uv_rs2re      = uv_T_joi(T_joi_src,'rs','re')
                uv_re2rw      = uv_T_joi(T_joi_src,'re','rw')
                uv_neck2ls    = uv_T_joi(T_joi_src,'neck','ls')
                uv_ls2le      = uv_T_joi(T_joi_src,'ls','le')
                uv_le2lw      = uv_T_joi(T_joi_src,'le','lw')
                # Upper body position targets
                p_base_trgt = t2p(T_joi_trgt['base'])
                p_spine_trgt = p_base_trgt + len_base2spine*uv_base2spine
                p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                if ik_p_spine: # spine target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
                if ik_p_neck: # neck target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
            else:
                # Lower body position targets
                p_base_trgt = t2p(T_joi_src['base'])
                p_rp_trgt   = p_base_trgt + len_base2rp*uv_base2rp
                p_rk_trgt   = p_rp_trgt + len_rp2rk*uv_rp2rk
                p_ra_trgt   = p_rk_trgt + len_rk2ra*uv_rk2ra
                p_lp_trgt   = p_base_trgt + len_base2lp*uv_base2lp
                p_lk_trgt   = p_lp_trgt + len_lp2lk*uv_lp2lk
                p_la_trgt   = p_lk_trgt + len_lk2la*uv_lk2la
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rp'],p_trgt=p_rp_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lp'],p_trgt=p_lp_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rk'],p_trgt=p_rk_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lk'],p_trgt=p_lk_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ra'],p_trgt=p_ra_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['la'],p_trgt=p_la_trgt)
                
                # Upper body position targets
                p_spine_trgt = p_base_trgt + len_base2spine*uv_base2spine
                p_neck_trgt  = p_spine_trgt + len_spine2neck*uv_spine2neck
                p_rs_trgt    = p_neck_trgt + len_neck2rs*uv_neck2rs
                p_ls_trgt    = p_neck_trgt + len_neck2ls*uv_neck2ls
                p_re_trgt    = p_rs_trgt + len_rs2re*uv_rs2re
                p_le_trgt    = p_ls_trgt + len_ls2le*uv_ls2le
                p_rw_trgt    = p_re_trgt + len_re2rw*uv_re2rw
                p_lw_trgt    = p_le_trgt + len_le2lw*uv_le2lw
                if ik_p_spine: # spine target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
                if ik_p_neck: # neck target is optional
                    add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
                add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
            
        # Hand rotation targets
        if ik_R_hand:
            R_rh_trgt = t2r(T_joi_src['rh'])
            R_lh_trgt = t2r(T_joi_src['lh'])
            """
                Source and Target-specific
            """
            if (src_name=='cmu') and (trgt_name=='g1'): # CMU to G1
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([180,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,0]))
            elif (src_name=='cmu') and (trgt_name=='cr'): # CMU to Common-Rig
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([-90,0,0]))@rpy2r(np.deg2rad([0,0,-90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([-90,0,0]))@rpy2r(np.deg2rad([0,0,-90]))
            elif (src_name=='cmu') and (trgt_name=='humanoid'): # CMU to Humanoid
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,0]))@rpy2r(np.deg2rad([0,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,0]))@rpy2r(np.deg2rad([0,0,0]))

            elif (src_name=='smpl') and (trgt_name=='jf3'): # SMPL to JF3
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,180]))@rpy2r(np.deg2rad([90,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,0,0]))
            elif (src_name=='smpl') and (trgt_name=='baymax'): # SMPL to Baymax
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,0]))
            elif (src_name=='smpl') and (trgt_name=='romeo'): # SMPL to romeo
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,0,-90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,0,90]))
            elif (src_name=='smpl') and (trgt_name=='gundam'): # SMPL to gundam
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,90,60])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,-90,0]))
            elif (src_name=='smpl') and (trgt_name=='icub'): # SMPL to icub
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([90,0,90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,0,90]))
            elif (src_name=='smpl') and (trgt_name=='jaxon'): # SMPL to jaxon
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([-90,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,0,0]))
            elif (src_name=='smpl') and (trgt_name=='poppy'): # SMPL to poppy
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([180,90,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,-90,0]))
            elif (src_name=='smpl') and (trgt_name=='valkyrie'): # SMPL to valkyrie
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,90,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([0,90,0]))
            elif (src_name=='smpl') and (trgt_name=='armar6'): # SMPL to armar6
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([0,-90,90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([-90,-90,0]))
            elif (src_name=='smpl') and (trgt_name=='apollo'): # SMPL to apollo
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([-90,0,0])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([90,0,0]))
            elif (src_name=='smpl') and (trgt_name=='robonaut2'): # SMPL to robonaut2
                R_rh_trgt = R_rh_trgt@rpy2r(np.deg2rad([180,0,-90])) # global offset
                R_lh_trgt = R_lh_trgt@rpy2r(np.deg2rad([180,0,90]))

            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rh'],R_trgt=R_rh_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lh'],R_trgt=R_lh_trgt)
        if ik_R_foot:
            R_rf_trgt = t2r(T_joi_src['rf'])
            R_lf_trgt = t2r(T_joi_src['lf'])

            if (src_name=='smpl') and (trgt_name=='g1'): # CMU to G1
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,0,0]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,0,0]))
            elif (src_name=='smpl') and (trgt_name=='digit'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,-90,-90]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,-90,90]))
            elif (src_name=='smpl') and (trgt_name=='gundam'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,0,90]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,0,90]))
            elif (src_name=='smpl') and (trgt_name=='poppy'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([90,0,90]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([90,0,90]))
            elif (src_name=='smpl') and (trgt_name=='armar4'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,90,180]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,-90,90]))
            elif (src_name=='smpl') and (trgt_name=='bruce'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([90,90,0]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([90,90,0]))
            elif (src_name=='smpl') and (trgt_name=='themis'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,90,0]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,90,0]))
            elif (src_name=='smpl') and (trgt_name=='x1'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([-90,0,-90]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([90,0,90]))
            elif (src_name=='smpl') and (trgt_name=='robonaut2'):
                R_rf_trgt = R_rf_trgt@rpy2r(np.deg2rad([0,180,0]))
                R_lf_trgt = R_lf_trgt@rpy2r(np.deg2rad([0,180,0]))

            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['rf'],R_trgt=R_rf_trgt)
            add_ik_info(ik_info_whole_body,body_name=joi_body_name_trgt['lf'],R_trgt=R_lf_trgt)
        # Solve IK for the Lower Bodies
        ik_err_norm_list = []
        for ik_tick in range(max_ik_tick): # ik loop
            dq,ik_err_stack = get_dq_from_ik_info(
                env            = env,
                ik_info        = ik_info_whole_body,
                stepsize       = ik_stepsize,
                eps            = 1e-3,
                th             = np.radians(10.0),
                joint_idxs_jac = joint_with_base_idxs_jac,
            ) # dq:[43]
            qpos = env.get_qpos() # get current joint position  [44]
            mujoco.mj_integratePos(env.model,qpos,dq,1)
            env.forward(q=qpos)
            ik_err_norm = np.linalg.norm(ik_err_stack)
            ik_err_norm_list.append(ik_err_norm)
            if ik_err_norm < ik_err_th: break
        if print_ik_failure and (ik_err_norm>=ik_err_th):
            print ("Lower Body IK Error Too High: tick:[%d] ik_err_norm:[%.2e] ik_err_th:[%.2e]"%
                   (tick,ik_err_norm,ik_err_th))

        # Append joint position
        qpos_dir_list.append(env.get_qpos())

        if trgt_name == 'mahru':
            qpos_dir_list[tick][18] = 1.0
            qpos_dir_list[tick][22] = -1.0
        
        # Render
        if animate and (tick%render_every==0 or tick==0):
            chain_src.plot_chain_mujoco(
                env             = env,
                plot_joint      = False,
                plot_joint_name = False,
                r_link          = 0.0025,
                rgba_link       = (1,0,0,0.5),
            ) # source link with red lines
            env.plot_links_between_bodies(
                r    = 0.0025,
                rgba = (0,0,1,0.5),
            ) # plot target with blue
            env.plot_T()
            env.plot_text(p=np.array([0,0,1]),label='tick:[%d/%d time:[%.3f]sec'%(tick,L,sec))
            env.plot_joint_axis(axis_len=0.01,axis_r=0.001)
            env.plot_contact_info()
            plot_ik_info(env=env,ik_info=ik_info_whole_body)
            if plot_joi_src:
                """
                    Source-specific
                """
                if src_name == 'cmu':
                    T_joi_src = get_T_joi_of_chain_cmu(chain_src,base_between_pelvis=True)
                for key in T_joi_src.keys():
                    env.plot_T(T=T_joi_src[key],plot_axis=True,axis_len=0.05,axis_width=0.005)
            if plot_joi_trgt:
                """
                    Target-specific
                """
                if override is not None:
                    T_joi_trgt = get_T_joi_trgt(env,trgt_name,override=override)
                else:
                    T_joi_trgt = get_T_joi_trgt(env,trgt_name)
                for key in T_joi_trgt.keys():
                    env.plot_T(T=T_joi_trgt[key],plot_axis=True,axis_len=0.05,axis_width=0.005)
                    
            env.render()
            
        # Print process
        if verbose:
            if ((tick%print_every)==0) or (tick==(L-1)):
                print ("[%d/%d] [%.1f]%% ik_err_norm:[%.2e] ik_err_th:[%.2e]"%
                       (tick,L,100*tick/L,ik_err_norm,ik_err_th))
                
    # Close viewer
    if animate: env.close_viewer()  
    
    # Return
    return qpos_dir_list
