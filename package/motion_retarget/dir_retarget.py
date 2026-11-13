import glfw
import mujoco
import numpy as np

from transforms import (
    pr2t,
    t2p,
    t2r,
)
from utils import (
    len_T_joi,
    get_idxs,
    uv_T_joi,
    len_T_joi,
    MultiSliderQtClass,
)

from ik import (
    init_ik_info,
    add_ik_info,
    get_dq_from_ik_info,
    plot_ik_info,
)

def get_T_joi_of_chain_smpl(
        chain,
        base_between_pelvis = True,
    ):
    """ 
        Get joints of interest of SMPL chain
    """
    p_base,R_base = chain.get_joint_pR(joint_name='base')
    p_spine,R_spine = chain.get_joint_pR(joint_name='spine1')
    p_rs,R_rs = chain.get_joint_pR(joint_name='rs')
    p_re,R_re = chain.get_joint_pR(joint_name='re')
    p_rw,R_rw = chain.get_joint_pR(joint_name='rw')
    p_rh,R_rh = chain.get_joint_pR(joint_name='rh')
    p_ls,R_ls = chain.get_joint_pR(joint_name='ls')
    p_le,R_le = chain.get_joint_pR(joint_name='le')
    p_lw,R_lw = chain.get_joint_pR(joint_name='lw')
    p_lh,R_lh = chain.get_joint_pR(joint_name='lh')
    p_neck,R_neck = chain.get_joint_pR(joint_name='neck')
    # z neck position to be the center of two shoulder positions
    p_neck_z = 0.5*(p_rs+p_ls) 
    p_neck = np.array([p_neck[0],p_neck[1],p_neck_z[2]])
    p_head,R_head = chain.get_joint_pR(joint_name='head')
    p_rp,R_rp = chain.get_joint_pR(joint_name='rp')
    p_rk,R_rk = chain.get_joint_pR(joint_name='rk')
    p_ra,R_ra = chain.get_joint_pR(joint_name='ra')
    p_rf,R_rf = chain.get_joint_pR(joint_name='ra')
    p_lp,R_lp = chain.get_joint_pR(joint_name='lp')
    p_lk,R_lk = chain.get_joint_pR(joint_name='lk')
    p_la,R_la = chain.get_joint_pR(joint_name='la')
    p_lf,R_lf = chain.get_joint_pR(joint_name='la')
    p_r_toe,R_r_toe = chain.get_joint_pR(joint_name='rf')
    p_l_toe,R_l_toe = chain.get_joint_pR(joint_name='lf')
    # Modify base position
    if base_between_pelvis: p_base = 0.5*(p_rp+p_lp)
    # Return
    T_joi = {
        'base': pr2t(p_base,R_base),
        'spine': pr2t(p_spine,R_spine),
        'rs': pr2t(p_rs,R_rs),
        're': pr2t(p_re,R_re),
        'rw': pr2t(p_rw,R_rw),
        'rh': pr2t(p_rh,R_rh),
        'ls': pr2t(p_ls,R_ls),
        'le': pr2t(p_le,R_le),
        'lw': pr2t(p_lw,R_lw),
        'lh': pr2t(p_lh,R_lh),
        'neck':pr2t(p_neck,R_neck),
        'head':pr2t(p_head,R_head),
        'rp': pr2t(p_rp,R_rp),
        'rk': pr2t(p_rk,R_rk),
        'ra': pr2t(p_ra,R_ra),
        'rf': pr2t(p_rf,R_rf),
        'rtoe': pr2t(p_r_toe,R_r_toe),
        'lp': pr2t(p_lp,R_lp),
        'lk': pr2t(p_lk,R_lk),
        'la': pr2t(p_la,R_la),
        'lf': pr2t(p_lf,R_lf),
        'ltoe': pr2t(p_l_toe,R_l_toe),
    }
    return T_joi

def get_joi_body_name_of_smplrig():
    """
        Get the body names of JOI of SMPL rig
    """
    joi_body_name = {
        'root':'root',
        'base':'base',
        'spine':'spine1',
        'neck':'head',
        'rs':'right_shoulder',
        're':'right_elbow',
        'rw':'right_wrist',
        'rh':'right_wrist',
        'ls':'left_shoulder',
        'le':'left_elbow',
        'lw':'left_wrist',
        'lh':'left_wrist',
        'rp':'right_pelvis',
        'rk':'right_knee',
        'ra':'right_ankle',
        'rf':'right_ankle',
        'lp':'left_pelvis',
        'lk':'left_knee',
        'la':'left_ankle',
        'lf':'left_ankle'
    }
    return joi_body_name

def get_T_joi_of_smplrig(
        env,
        neck_between_shoulder = False,
        base_between_pelvis   = False,
    ):
    """
        Get T of JOI of SMPL rig
    """
    joi_body_name = get_joi_body_name_of_smplrig()
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

def dir_preretarget_smpl_chains_to_smpl_rig(
        env,
        chains,
        render = False,
    ):
    """
    Directionl pre retargeting

    Parameters:
        env (Environment): The environment to run the simulation in.
        chains (list): List of motion chains to retarget.
        render (bool): Whether to render the simulation.    

    Returns:
        qpos_list (list): List of joint positions for each tick.
    """
    # Buffer
    qpos_list = []
    # Reset
    env.reset()
    if render:
        env.init_viewer(height=env.monitor_height,transparent=True)
        env.set_geom_alpha(alpha=0.5)
    # Loop
    L = len(chains)
    for tick in range(L): # for each tick

        # Pause (press 'P') for resuming press 'space'
        if env.is_key_pressed_once(glfw.KEY_P):
            env.viewer_pause()

        # Update
        chain = chains[tick]
        
        # Get joint of interest
        T_joi_src  = get_T_joi_of_chain_smpl(chain,base_between_pelvis=False)
        T_joi_trgt = get_T_joi_of_smplrig(env,neck_between_shoulder=False)
        joi_body_name_trgt = get_joi_body_name_of_smplrig()

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

        # Get unit vectors
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

        # Initialize IK
        joint_names_for_ik = env.rev_joint_names
        joint_idxs_jac = env.get_idxs_jac(joint_names=joint_names_for_ik)
        joint_idxs_jac_with_base = np.concatenate(([0,1,2,3,4,5],joint_idxs_jac))
        q_mins = env.joint_ranges[get_idxs(env.joint_names,joint_names_for_ik),0]
        q_maxs = env.joint_ranges[get_idxs(env.joint_names,joint_names_for_ik),1]

        # Set IK targets
        ik_info = init_ik_info()
        p_base_src = t2p(T_joi_src['base'])
        p_base_trgt = p_base_src.copy() # base
        p_rp_trgt = p_base_trgt + len_base2rp*uv_base2rp # right pelvis
        p_lp_trgt = p_base_trgt + len_base2lp*uv_base2lp # left pelvis
        add_ik_info(ik_info,body_name=joi_body_name_trgt['base'],p_trgt=p_base_trgt)
        add_ik_info(ik_info,body_name=joi_body_name_trgt['rp'],p_trgt=p_rp_trgt)
        add_ik_info(ik_info,body_name=joi_body_name_trgt['lp'],p_trgt=p_lp_trgt)
        p_spine_trgt = p_base_trgt + len_base2spine*uv_base2spine # spine
        p_neck_trgt = p_spine_trgt + len_spine2neck*uv_spine2neck # neck
        add_ik_info(ik_info,body_name=joi_body_name_trgt['spine'],p_trgt=p_spine_trgt)
        add_ik_info(ik_info,body_name=joi_body_name_trgt['neck'],p_trgt=p_neck_trgt)
        p_rs_trgt = p_neck_trgt + len_neck2rs*uv_neck2rs # right shoulder
        p_ls_trgt = p_neck_trgt + len_neck2ls*uv_neck2ls # left shoulder
        add_ik_info(ik_info,body_name=joi_body_name_trgt['rs'],p_trgt=p_rs_trgt)
        add_ik_info(ik_info,body_name=joi_body_name_trgt['ls'],p_trgt=p_ls_trgt)

        # Interate IK for 'base', 'spine', 'neck', 'shoulder'
        max_ik_tick = 100
        for ik_tick in range(max_ik_tick): # ik loop
            ik_stepsize = 1
            dq,ik_err_stack = get_dq_from_ik_info(
                env            = env,
                ik_info        = ik_info,
                stepsize       = ik_stepsize,
                eps            = 1e-3,
                th             = np.radians(10.0),
                joint_idxs_jac = joint_idxs_jac_with_base,
            )
            qpos = env.get_qpos() 
            mujoco.mj_integratePos(env.model,qpos,dq,1)
            qpos[7:] = np.clip(qpos[7:],q_mins,q_maxs) # clip range
            env.forward(q=qpos)

        # Solve IK once more for arms and legs
        T_joi_trgt = get_T_joi_of_smplrig(env) # recompute T joi for target

        # Initialize IK
        joint_names_for_ik = ['l_shoulder1','l_shoulder2','l_shoulder3','l_elbow',
                            'l_wrist1','l_wrist2','l_wrist3',
                            'r_shoulder1','r_shoulder2','r_shoulder3','r_elbow',
                            'r_wrist1','r_wrist2','r_wrist3',
                            'l_pelvis1','l_pelvis2','l_pelvis3','l_knee',
                            'l_ankle1','l_ankle2','l_ankle3',
                            'r_pelvis1','r_pelvis2','r_pelvis3','r_knee',
                            'r_ankle1','r_ankle2','r_ankle3']
        joint_idxs_jac_arm = env.get_idxs_jac(joint_names=joint_names_for_ik)
        
        # Set IK targets
        ik_info_armleg = init_ik_info()
        p_re_trgt = t2p(T_joi_trgt['rs']) + len_rs2re*uv_rs2re # right elbow
        p_le_trgt = t2p(T_joi_trgt['ls']) + len_ls2le*uv_ls2le # left elbow
        add_ik_info(ik_info_armleg,body_name=joi_body_name_trgt['re'],p_trgt=p_re_trgt)
        add_ik_info(ik_info_armleg,body_name=joi_body_name_trgt['le'],p_trgt=p_le_trgt)
        p_rw_trgt = p_re_trgt + len_re2rw*uv_re2rw # right wrist
        p_lw_trgt = p_le_trgt + len_le2lw*uv_le2lw # left wrist
        add_ik_info(ik_info_armleg,body_name=joi_body_name_trgt['rw'],p_trgt=p_rw_trgt)
        add_ik_info(ik_info_armleg,body_name=joi_body_name_trgt['lw'],p_trgt=p_lw_trgt)
        p_rk_trgt = t2p(T_joi_trgt['rp']) + len_rp2rk*uv_rp2rk # right knee
        p_lk_trgt = t2p(T_joi_trgt['lp']) + len_lp2lk*uv_lp2lk # leff knee
        add_ik_info(ik_info_armleg,body_name=joi_body_name_trgt['rk'],p_trgt=p_rk_trgt)
        add_ik_info(ik_info_armleg,body_name=joi_body_name_trgt['lk'],p_trgt=p_lk_trgt)
        p_ra_trgt = p_rk_trgt + len_rk2ra*uv_rk2ra # right ankle
        p_la_trgt = p_lk_trgt + len_lk2la*uv_lk2la # left ankle
        add_ik_info(ik_info_armleg,body_name=joi_body_name_trgt['ra'],p_trgt=p_ra_trgt)
        add_ik_info(ik_info_armleg,body_name=joi_body_name_trgt['la'],p_trgt=p_la_trgt)
        
        # Interate IK for 'elbow', 'wrist', 
        max_ik_tick = 100
        for ik_tick in range(max_ik_tick): # ik loop
            ik_stepsize = 1
            dq,ik_err_stack = get_dq_from_ik_info(
                env            = env,
                ik_info        = ik_info_armleg,
                stepsize       = ik_stepsize,
                eps            = 1e-3,
                th             = np.radians(10.0),
                joint_idxs_jac = joint_idxs_jac_arm,
            )
            qpos = env.get_qpos() 
            mujoco.mj_integratePos(env.model,qpos,dq,1)
            qpos[7:] = np.clip(qpos[7:],q_mins,q_maxs) # clip range
            env.forward(q=qpos)

        # Append joint positions
        qpos_list.append(env.get_qpos())
        
        # Render
        if render:
            if env.is_viewer_alive():
                env.viewer_text_overlay('Tick','%d/%d'%(tick,L))
                env.plot_joint_axis(rate=0.5)
                env.plot_links_between_bodies()
                chain.plot_chain_mujoco(
                    env,
                    rgba_link       = (0,0,1,0.5),
                    rgba_joint      = (0,0,1,0.5),
                    plot_joint_axis = False,
                    plot_joint_name = False,
                )
                plot_ik_info(env=env,ik_info=ik_info)
                plot_ik_info(env=env,ik_info=ik_info_armleg)
                env.render()
            else:
                print ("No viewer, break at tick:[%d/%d]"%(tick,L))
                break

    # Close
    if render: env.close_viewer()

    # Return
    return qpos_list

def animate_chains_qpos_list(
        env,
        secs,
        chains,
        qpos_list,
        use_slider      = False,
        width           = 0.8,
        height          = 0.8,
        plot_chain_axis = True,
        plot_rig_axis   = True,
        rig_axis_rate   = 1.0,
        render_every    = 10,
    ):
    """
    Animate chains with given qpos_list in the environment.

    Parameters:
        env (Environment): The environment to animate in.
        secs (list): List of time in seconds for each tick.
        chains (list): List of motion chains to animate.
        qpos_list (list): List of joint positions corresponding to each tick.
        use_slider (bool): Whether to use a slider for controlling the animation.
        width (float): Width of the viewer.
        height (float): Height of the viewer.
        plot_chain_axis (bool): Whether to plot the chain axis.
        plot_rig_axis (bool): Whether to plot the rig axis.
        rig_axis_rate (float): Rate at which to plot the rig axis.
        render_every (int): Frequency of rendering frames.

    Returns:
        None
    """
    # Reset environment
    env.reset()
    env.init_viewer(width=width,height=height,transparent=True)
    env.set_geom_alpha(alpha=0.5)
    L = len(chains) # motion length

    # Slider
    if use_slider:
        sliders = MultiSliderQtClass(
            n_slider        = 1,
            window_width    = 500,
            window_height   = 50,
            slider_width    = 400,
            label_width     = 100,
            label_texts     = ['Tick'],
            label_colors    = ['w'],
            slider_mins     = np.array([0]),
            slider_maxs     = np.array([L-1]),
            slider_vals     = np.array([0,0,0,0,0,0]),
            verbose         = False,
        )

    # Loop 
    tick,is_running = 0,False
    while env.is_viewer_alive():
        
        # Key inputs handling
        if use_slider and not is_running: # get tick from slider only when paused
            tick = int(sliders.get_values()[0])
            env.set_wall_time(secs[tick]) # set wall time to match tick
        if env.is_key_pressed_once(glfw.KEY_P): # [p]: pause
            is_running = not is_running 
        if env.is_key_pressed_once(glfw.KEY_R): # [r]: reset
            tick = 0
            is_running = False
            env.reset_wall_time()
            sliders.set_values([tick])

        # Update
        sec      = secs[tick] # time in [seconds]
        chain    = chains[tick] # mocap chain
        qpos_dpr = qpos_list[tick] # joint position (44)
        env.forward(q=qpos_dpr) # forward kinematics
        env.increase_wall_time(step_flag=is_running)

        # Proceed to next tick if running
        if env.get_wall_time() > sec and is_running:
            tick = tick + 1

        # Check tick bounds
        if tick > L-1: 
            tick = L-1
            is_running = False # stop at the end of motion

        # Update slider
        if use_slider and is_running:
            sliders.set_values([tick])
        
        # Render
        if (tick % render_every == 0) or not is_running:
            stat_text = 'Run' if is_running else 'Paused'
            env.viewer_text_overlay('Status',stat_text)
            info_text = '[p] to pause' if is_running else '[p] to resume'
            env.viewer_text_overlay('Press',info_text)
            env.viewer_text_overlay('Press','[r] to reset')
            env.viewer_text_overlay('Tick','[%d/%d]'%(tick,L))
            env.viewer_text_overlay('Sim Time','[%.2f]sec'%(sec))
            env.viewer_text_overlay('Wall Time','[%.2f]sec'%(env.get_wall_time()))
            chain.plot_chain_mujoco(
                env,
                rgba_link       = (0,0,1,0.5),
                rgba_joint      = (0,0,1,0.5),
                plot_joint_axis = plot_chain_axis,
                plot_joint_name = False,
            )

            # Plot rig axis
            if plot_rig_axis:
                env.plot_joint_axis(rate=rig_axis_rate)

            env.render()

    # Close
    env.close_viewer()
    if use_slider: sliders.close()