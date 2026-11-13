import mujoco
import numpy as np
import matplotlib.pyplot as plt

from joi_cmu import get_T_joi_of_chain_cmu
from joi_g1 import get_joi_body_name_of_g1,get_T_joi_of_g1
from joi_cr import get_joi_body_name_of_cr,get_T_joi_of_cr
from joi_humanoid import get_joi_body_name_of_humanoid,get_T_joi_of_humanoid
"""
    sys.path.append('../../package/utility/') # for 'utils'
    sys.path.append('../../package/kinematics_helper/') # for 'transforms' and 'ik'
"""
from utils import (
    len_T_joi,
    uv_T_joi,
    get_A_vel_acc_jerk,
    smooth_gaussian_1d,
    smooth_optm_1d,
    get_consecutive_subarrays,
)
from transforms import (t2p,t2r,rpy2r,T_yuzf2zuxf)
from ik import (init_ik_info,add_ik_info,get_dq_from_ik_info,plot_ik_info)

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
        ik_p_spine         = True 
        ik_p_neck          = True
    
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
        
        # Move target base (once)
        if tick == 0:
            T_base_trgt  = T_joi_src['base']
            """
                Source-specific
            """
            if src_name == 'cmu': # for CMU-MoCap (y-up z-front), make it z-up x-front
                T_base_trgt = T_yuzf2zuxf(T_base_trgt)
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
        
        # Set IK targets 
        ik_info_whole_body = init_ik_info()
        
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
                if src_name == 'g1':
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

def get_contact_segments(
        secs,
        rtoe_traj,
        ltoe_traj,
        zvel_th     = 0.2, # toe z velocity threshold to detect contact
        min_seg_sec = 0.1, # minimum segment time (to filter out false-positve contact segments)
        jerk_limit  = 10.0,
        verbose     = True,
        plot        = True,
        figsize     = (8,3),
    ):
    """ 
        Get contact segments from right and left feet (or toe) trajectories
    """
    # Smooth toe trajectories and get velocity
    L,dt = len(secs),secs[1]-secs[0]
    HZ   = int(1/dt)
    A_vel,_,_ = get_A_vel_acc_jerk(n=L,dt=1/HZ)
    rtoe_traj_z,ltoe_traj_z = rtoe_traj[:,2],ltoe_traj[:,2]
    rtoe_traj_z_smt = smooth_optm_1d(
        traj = rtoe_traj_z,
        dt   = secs[1]-secs[0],
        jerk_limit = jerk_limit,
    )
    ltoe_traj_z_smt = smooth_optm_1d(
        traj = ltoe_traj_z,
        dt   = secs[1]-secs[0],
        jerk_limit = jerk_limit,
    )
    
    rtoe_veltraj_z,ltoe_veltraj_z = A_vel@rtoe_traj_z,A_vel@ltoe_traj_z
    rtoe_veltraj_z_smt,ltoe_veltraj_z_smt = A_vel@rtoe_traj_z_smt,A_vel@ltoe_traj_z_smt
    
    # Contact detection using smoothed trajectories
    ticks_rcontact = np.where(np.abs(rtoe_veltraj_z_smt) <= zvel_th)[0]
    ticks_lcontact = np.where(np.abs(ltoe_veltraj_z_smt) <= zvel_th)[0]
    
    # Get contact segments
    min_seg_len = int(min_seg_sec*HZ)
    rcontact_segs = get_consecutive_subarrays(ticks_rcontact,min_element=min_seg_len)
    lcontact_segs = get_consecutive_subarrays(ticks_lcontact,min_element=min_seg_len)
    
    # Print
    if verbose:
        print ("min_seg_sec:[%.2f]sec min_seg_len:[%d]"%(min_seg_sec,min_seg_len))
        print ("We have [%d] right contact segment(s)"%(len(rcontact_segs)))
        for seg_idx,seg in enumerate(rcontact_segs):
            print (" [%d] len:[%d]"%(seg_idx,len(seg)))
        print ("We have [%d] left contact segment(s)"%(len(lcontact_segs)))
        for seg_idx,seg in enumerate(lcontact_segs):
            print (" [%d] len:[%d]"%(seg_idx,len(seg)))
    
    # Plot results
    if plot:
        # Plot feet height
        plt.figure(figsize=figsize)
        plt.plot(secs,rtoe_traj_z,'-',color='r',lw=1/3,marker='none',mfc='none',ms=3,mew=0.5,
                label='Raw Right Toe')
        plt.plot(secs,ltoe_traj_z,'-',color='b',lw=1/3,marker='none',mfc='none',ms=3,mew=0.5,
                label='Raw Left Toe')
        plt.plot(secs,rtoe_traj_z_smt,'-',color='r',lw=1,marker='none',mfc='none',ms=3,mew=0.5,
                label='Smoothed Right Toe')
        plt.plot(secs,ltoe_traj_z_smt,'-',color='b',lw=1,marker='none',mfc='none',ms=3,mew=0.5,
                label='Smoothed Left Toe')
        # plt.plot(secs[ticks_rcontact],rtoe_traj_z[ticks_rcontact],
        #         linestyle='none',color='r',lw=1/5,marker='x',mfc='none',ms=3,mew=1/3,
        #         label='Raw Right Contact')
        # plt.plot(secs[ticks_lcontact],ltoe_traj_z[ticks_lcontact],
        #         linestyle='none',color='b',lw=1/5,marker='x',mfc='none',ms=3,mew=1/3,
        #         label='Raw Left Contact')
        markers = ['o','v','^','<','>','s','p']
        for seg_idx,rseg in enumerate(rcontact_segs):
            plt.plot(secs[rseg],rtoe_traj_z[rseg],
                    linestyle='none',color='r',lw=1/3,marker=markers[seg_idx],mfc='none',ms=4,mew=1/3,
                    label='Filtered Right Contact [%d]'%(seg_idx))
        for seg_idx,lseg in enumerate(lcontact_segs):
            plt.plot(secs[lseg],ltoe_traj_z[lseg],
                    linestyle='none',color='b',lw=1/3,marker=markers[seg_idx],mfc='none',ms=4,mew=1/3,
                    label='Filtered Left Contact [%d]'%(seg_idx))
        plt.title('Toe Height',fontsize=8)
        plt.xlabel('Time (sec)',fontsize=8)
        plt.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", borderaxespad=0,fontsize=6)
        plt.tight_layout(); plt.show()

        # Plot feet velocity
        plt.figure(figsize=figsize)
        plt.plot(secs,zvel_th*np.ones(L),'--',color='k',lw=1,marker='none',mfc='none',ms=3,mew=0.5)
        plt.plot(secs,-zvel_th*np.ones(L),'--',color='k',lw=1,marker='none',mfc='none',ms=3,mew=0.5)
        plt.plot(secs,rtoe_veltraj_z,'-',color='r',lw=1/3,marker='none',mfc='none',ms=3,mew=0.5,
                label='Raw Right Toe')
        plt.plot(secs,ltoe_veltraj_z,'-',color='b',lw=1/3,marker='none',mfc='none',ms=3,mew=0.5,
                label='Raw Left Toe')
        plt.plot(secs,rtoe_veltraj_z_smt,'-',color='r',lw=1,marker='none',mfc='none',ms=3,mew=0.5,
                label='Smoothed Right Toe')
        plt.plot(secs,ltoe_veltraj_z_smt,'-',color='b',lw=1,marker='none',mfc='none',ms=3,mew=0.5,
                label='Smoothed Left Toe')
        # plt.plot(secs[ticks_rcontact],rtoe_veltraj_z[ticks_rcontact],
        #         linestyle='none',color='r',lw=1/5,marker='x',mfc='none',ms=3,mew=1/3,
        #         label='Raw Right Contact')
        # plt.plot(secs[ticks_lcontact],ltoe_veltraj_z[ticks_lcontact],
        #         linestyle='none',color='b',lw=1/5,marker='x',mfc='none',ms=3,mew=1/3,
        #         label='Raw Left Contact')
        for seg_idx,rseg in enumerate(rcontact_segs):
            plt.plot(secs[rseg],rtoe_veltraj_z_smt[rseg],
                    linestyle='none',color='r',lw=1/3,marker=markers[seg_idx],mfc='none',ms=4,mew=1/3,
                    label='Filtered Right Contact [%d]'%(seg_idx))
        for seg_idx,lseg in enumerate(lcontact_segs):
            plt.plot(secs[lseg],ltoe_veltraj_z_smt[lseg],
                    linestyle='none',color='b',lw=1/3,marker=markers[seg_idx],mfc='none',ms=4,mew=1/3,
                    label='Filtered Left Contact [%d]'%(seg_idx))
        plt.title('Toe Velocity',fontsize=8)
        plt.xlabel('Time (sec)',fontsize=8)
        plt.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left", borderaxespad=0,fontsize=6)
        plt.tight_layout(); plt.show()
        
    # Print contact segments
    if verbose:
        for c_idx,rcontact_seg in enumerate(rcontact_segs):
            print ("Right Contact [%d/%d] [%d]~[%d]"%
                (c_idx,len(rcontact_segs),rcontact_seg[0],rcontact_seg[-1]))
        for c_idx,lcontact_seg in enumerate(lcontact_segs):
            print ("Left Contact [%d/%d] [%d]~[%d]"%
                (c_idx,len(lcontact_segs),lcontact_seg[0],lcontact_seg[-1]))      
               
    # Return contact segments
    return rcontact_segs,lcontact_segs