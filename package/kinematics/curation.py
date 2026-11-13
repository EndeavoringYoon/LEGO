import mujoco as mj
import time
from copy import deepcopy
import numpy as np

def _find_avg_joint_pos_recursive(env,
                                geom_id,
                                p_joint_global,
                                axis_world,
                                distance1=None,
                                distance2=None,
                                th=0.001,
                                tol_diff=0.005,
                                max_iter=200,
                                iteration=0,
                                vis=False):
    """
    Recursive approach:
    1) if distance1 and distance2 are both None:
        - cast a ray in each direction (+axis, -axis) and update the distances
    2) if only one of them is None:
        - move p_joint_global by th toward the side that has a value
        - cast rays again and update distance1, distance2
        - recursive call
    3) if both have values:
        - move to the 'middle' so that distance1 and distance2 become equal
        - stop if the difference between the two values is below tol_diff
        - otherwise update the rays and recurse
    4) stop when max_iter (recursion limit) is reached
    """
    geom_id1 = int(geom_id[0])
    geom_id2 = int(geom_id[1])
    if vis:
        env.plot_sphere(
                    p=p_joint_global,
                    r=0.03,  # sphere radius
                    rgba=[1,0,0,1],
                    )
    # ============= Base / Safety check =============
    if iteration >= max_iter:
        # reached the maximum recursion depth
        if vis:
            print("max_iter_apporached")
        return p_joint_global, distance1, distance2
    if geom_id1==-1 or geom_id2==-1:
        move_sign = 1.0 if geom_id2==-1 else -1.0
        d= mj.mj_rayMesh(env.model, env.data,
                        geom_id1, p_joint_global, axis_world) if geom_id2==-1 else mj.mj_rayMesh(env.model, env.data,
                        geom_id2, p_joint_global, -axis_world)
        p_joint_global += axis_world * d * move_sign
        return p_joint_global, distance1, distance2
    # ============= 1) both distance1 and distance2 are None?  =============
    if distance1 is None and distance2 is None:
        d1 = mj.mj_rayMesh(env.model, env.data,
                        geom_id1, p_joint_global, axis_world)
        d2 = mj.mj_rayMesh(env.model, env.data,
                        geom_id2, p_joint_global, -axis_world)

        # -1 means no collision -> back to None
        distance1 = None if d1 == -1 else d1
        distance2 = None if d2 == -1 else d2

        # continue with the logic below (no immediate recursion). Handled by the conditions below.

    # ============= 2) only one of them is None =============
    if distance1 is not None and distance2 is None:
        # only distance1 exists -> there is a mesh in the +axis direction
        # move p_joint_global by th toward +axis
        
        p_joint_global += axis_world * (distance1 + th)
        if vis:
            env.plot_sphere(
                        p=p_joint_global,
                        r=0.03,  # sphere radius
                        rgba=[0,0,1,1],
                        )
        # after moving, cast rays on both the positive and negative side again and update
        d1 = mj.mj_rayMesh(env.model, env.data,
                        geom_id1, p_joint_global, axis_world)
        d2 = mj.mj_rayMesh(env.model, env.data,
                        geom_id2, p_joint_global, -axis_world)

        distance1 = None if d1 == -1 else d1
        distance2 = None if d2 == -1 else d2
        if vis:
            env.render()
            print(f"iter:{iteration}, d2 was None -> now dist1: {distance1}, dist2: {distance2}")
            time.sleep(0.5)
        return _find_avg_joint_pos_recursive(
            geom_id, p_joint_global, axis_world,
            distance1, distance2,
            th, tol_diff, max_iter, iteration+1, vis
        )

    elif distance1 is None and distance2 is not None:
        # only distance2 exists -> there is a mesh in the -axis direction
        # move p_joint_global by th toward -axis
        p_joint_global -= axis_world * (distance2 + th)
        if vis:
            env.plot_sphere(
                        p=p_joint_global,
                        r=0.03,  # sphere radius
                        rgba=[0,0,1,0.5],
                        )
        # cast the ray again after moving
        d1 = mj.mj_rayMesh(env.model, env.data,
                        geom_id1, p_joint_global, axis_world)
        d2 = mj.mj_rayMesh(env.model, env.data,
                        geom_id2, p_joint_global, -axis_world)

        distance1 = None if d1 == -1.0 else d1
        distance2 = None if d2 == -1.0 else d2
        if vis:
            env.render()
            print(f"iter:{iteration}, d1 was None -> now dist1: {distance1}, dist2: {distance2}")
            time.sleep(0.5)
        return _find_avg_joint_pos_recursive(
            geom_id, p_joint_global, axis_world,
            distance1, distance2,
            th, tol_diff, max_iter, iteration+1, vis
        )

    # ============= 3) both have values =============
    if distance1 is not None and distance2 is not None:
        # if both distance1 and distance2 exist
        diff = abs(distance1 - distance2)
        if diff <= tol_diff:
            # if the two are almost the same, stop here
            if geom_id1!=geom_id2:
                #check whether the joint is in specific mesh
                d3 = mj.mj_rayMesh(env.model, env.data,
                            geom_id1, p_joint_global, -axis_world)
                d4 = mj.mj_rayMesh(env.model, env.data,
                            geom_id2, p_joint_global, axis_world)
                if d3!=-1.0 and d4==-1.0: # |parent|<--axis--|body|
                    p_joint_global -= axis_world * d3
                elif d3==-1.0 and d4!=-1.0: # |body|<--axis--|parent|
                    p_joint_global += axis_world * d4
            if vis:
                env.plot_sphere(
                        p=p_joint_global,
                        r=0.03,  # sphere radius
                        rgba=[0,0,1,1],
                        )
                env.render()
                print(f"iter:{iteration}, middle_point_reached -> now dist1: {distance1}, dist2: {distance2}")
                time.sleep(0.5)
            return p_joint_global, distance1, distance2
        else:
            # if the two differ, move a little toward the 'midpoint'
            # e.g. if distance1 > distance2, move a little toward +axis
            move_sign = 1.0 if distance1 > distance2 else -1.0
            # how much to move is a matter of taste
            # here only about (diff/2) * 0.5 is applied
            move_amount = (diff / 2.0)
            p_joint_global += move_sign * axis_world * move_amount
            if vis:
                env.plot_sphere(
                            p=p_joint_global,
                            r=0.03,  # sphere radius
                            rgba=[0,0,1,1],
                            )
            # ray again after moving
            d1 = mj.mj_rayMesh(env.model, env.data,
                            geom_id1, p_joint_global, axis_world)
            d2 = mj.mj_rayMesh(env.model, env.data,
                            geom_id2, p_joint_global, -axis_world)

            distance1 = None if d1 == -1 else d1
            distance2 = None if d2 == -1 else d2
            if vis:
                env.render()
                print(f"iter:{iteration}, moving slightly to middle point -> now dist1: {distance1}, dist2: {distance2}")
                time.sleep(0.5)
            return _find_avg_joint_pos_recursive(
                geom_id, p_joint_global, axis_world,
                distance1, distance2,
                th, tol_diff, max_iter, iteration+1, vis
            )

    # ============= 4) otherwise (neither both None nor caught by the ifs above) =============
    # this is almost never reached, but for safety
    if vis:
        print("both are not none, and couldn't pass if")
        env.render()
        print(f"iter:{iteration}, p_glo: {p_joint_global},d1: {d1}, d2: {d2}")
    return p_joint_global, distance1, distance2

def _find_valid_geoms_in_joint(env, p_joint_global, axis_world,body_id, verbose=False):
    """
    find a mesh geom among the geoms of body_id.
    (a body can actually have several geoms, so the name or type should be checked more)
    """
    body_name=mj.mj_id2name(env.model,mj.mjtObj.mjOBJ_BODY,body_id)
    if verbose:
        print(f"body_name: {body_name}")
    # the geoms of a body are model.body_geomnum[body_id] geoms starting at model.body_geomadr[body_id]
    parent_geom_idx=None
    geom_idx=None
    parent_id = env.model.body_parentid[body_id]
    start_g = env.model.body_geomadr[parent_id]
    count_g = env.model.body_geomnum[parent_id]
    if count_g==0:
        if verbose:
            print(f"{body_name}: no mesh in parent")
        parent_geom_idx = None
    else:
        distance=[]
        for gid in range(start_g, start_g + count_g):
            geom_type = env.model.geom_type[gid]
            d=0.0
            if  geom_type == mj.mjtGeom.mjGEOM_MESH:
                d1 = mj.mj_rayMesh(env.model, env.data, gid, p_joint_global, axis_world)
                d2 = mj.mj_rayMesh(env.model, env.data, gid, p_joint_global, -axis_world)
            else:
                d1=0.0
                d2=0.0

            if d1!=-1.0 and d2!=-1.0:
                d=max(d1,d2)
            elif d1==-1.0:
                d=d2
            elif d2==-1.0:
                d=d1
            distance.append(d)
        d_max = max(distance)
        d_max_idx = distance.index(d_max)
        if d_max!=0.0:
            parent_geom_idx = deepcopy(start_g+d_max_idx)
    start_g = env.model.body_geomadr[body_id]
    count_g = env.model.body_geomnum[body_id]
    if count_g==0:
        if verbose:
            print(f"{body_name}: no mesh in this body")
        geom_idx = None
    else:
        distance=[]
        for gid in range(start_g, start_g + count_g):
            geom_type = env.model.geom_type[gid]
            d=0.0
            if  geom_type == mj.mjtGeom.mjGEOM_MESH:
                d1 = mj.mj_rayMesh(env.model, env.data, gid, p_joint_global, axis_world)
                d2 = mj.mj_rayMesh(env.model, env.data, gid, p_joint_global, -axis_world)
            else:
                d1=0.0
                d2=0.0

            if d1!=-1.0 and d2!=-1.0:
                d=max(d1,d2)
            elif d1==-1.0:
                d=d2
            elif d2==-1.0:
                d=d1
            distance.append(d)
        d_max = max(distance)
        d_max_idx = distance.index(d_max)
        if d_max!=0.0:
            geom_idx = deepcopy(start_g+d_max_idx)
    if parent_geom_idx is None and geom_idx is not None:
        if verbose:
            print(f"{body_name}: no parent geom_idx -> {[geom_idx, geom_idx]}")
        return [geom_idx, geom_idx]
    elif geom_idx is None:
        if verbose:
            print(f"{body_name}: no geom_idx in body")
        return False
    else:
        """Case where both the parent geom and the body geom exist"""
        d1_parent = mj.mj_rayMesh(env.model, env.data, parent_geom_idx, p_joint_global, axis_world)
        d2_parent = mj.mj_rayMesh(env.model, env.data, parent_geom_idx, p_joint_global, -axis_world)
        d1_body = mj.mj_rayMesh(env.model, env.data, geom_idx, p_joint_global, axis_world)
        d2_body = mj.mj_rayMesh(env.model, env.data, geom_idx, p_joint_global, -axis_world)
        
        if d1_parent!=-1 and d2_parent!=-1:
            between_parent=2
        elif d1_parent * d2_parent<0: # negative if either one is -1, since it is a distance
            between_parent=1
        elif d1_parent==-1 and d2_parent==-1:
            between_parent=0
        else:
            raise ValueError("weird parent")
        if d1_body!=-1 and d2_body!=-1:
            between_body=2
        elif d1_body * d2_body<0: # negative if either one is -1, since it is a distance
            between_body=1
        elif d1_body==-1 and d2_body==-1:
            between_body=0
        else:
            raise ValueError("weird body")
        
        if between_parent==2 and between_body==2:
            """U-shaped frame + double-sided motor (e.g. OP3): must go to the midpoint"""
            if verbose:
                print(f"{body_name}: btw_parent = {between_parent}, btw_body = {between_body} -> {[geom_idx, geom_idx]}")
            return [geom_idx, geom_idx]
        elif between_parent==2 and between_body==1:
            """L-shaped frame + single-sided motor that ended up inside the parent: must go between the two meshes"""
            """OP3's l_sho_pitch_link falls here; it should probably be attached to the side closer to the body"""
            """iCub's l_elbow_1 falls here, but r_elbow_1 is (2,2).."""
            if d1_body>d2_body:
                result = [-1, geom_idx]
            else:
                result = [geom_idx, -1]
            if verbose:
                print(f"{body_name}: btw_parent = {between_parent}, btw_body = {between_body} -> {result}")
                print(f"{body_name}: d1_parent: {d1_parent}, d2_parent: {d2_parent}")
                print(f"{body_name}: d1_body: {d1_body}, d2_body: {d2_body}")
        elif between_parent==2 and between_body==0:
            """Happens at Poppy's shin: U-shaped frame that sticks out too far from the body -> center it on the parent"""
            if verbose:
                print(f"{body_name}: btw_parent = {between_parent}, btw_body = {between_body} -> {[parent_geom_idx,parent_geom_idx]}")
            return [parent_geom_idx,parent_geom_idx]
        elif between_parent==1 and between_body==2:
            """
            L-shaped frame + single-sided motor that ended up inside the body, or
            case where the body is defined inside the parent mesh, as with the G1 shoulder pitch
            in this case, should it just be placed between the parent and the body..?
            JAXON's leg also falls here, and so does TIAGo dual's elbow
            then do not fix the G1 shoulder pitch, and
            fix TIAGo dual with this, and
            JAXON leg is curated
            """
            if d1_parent==-1:
                idx1 = geom_idx
            else:
                idx1 = parent_geom_idx
            if d2_parent==-1:
                idx2 = geom_idx
            else:
                idx2 = parent_geom_idx
            
            if verbose:
                print(f"{body_name}: btw_parent = {between_parent}, btw_body = {between_body} -> {[idx1, idx2]}")
            return [idx1, idx2]
        elif between_parent==1 and between_body==1:
            """Case where the joint sits properly between two meshes"""
            if verbose:
                print(f"{body_name}: btw_parent = {between_parent}, btw_body = {between_body} -> {False}")
            return False

        elif between_parent==1 and between_body==0:
            """Happens at Poppy's foot. Align it to the parent as well"""
            if verbose:
                print(f"{body_name}: btw_parent = {between_parent}, btw_body = {between_body} -> {[parent_geom_idx, parent_geom_idx]}")
            return [parent_geom_idx,parent_geom_idx]

        elif between_parent==0 and between_body==2:
            """Case where the joint must lie between bodies: no choice"""
            if verbose:
                print(f"{body_name}: btw_parent = {between_parent}, btw_body = {between_body} -> {[geom_idx, geom_idx]}")
            return [geom_idx, geom_idx]
        
        elif between_parent==0 and between_body==1:
            """Case where the joint must be attached to the body: no choice"""
            if d1_body>d2_body:
                result = [-1, geom_idx]
            else:
                result = [geom_idx, -1]
            if verbose:
                print(f"{body_name}: not between parent, between body -> {result}")
                print(f"{body_name}: d1_parent: {d1_parent},d2_parent: {d2_parent}")
                print(f"{body_name}: d1_body: {d1_body},d2_body: {d2_body}")
            return result
        
        elif between_parent==0 and between_body==0:
            """Impossible case"""
            if verbose:
                print(f"{body_name}: btw_parent = {between_parent}, btw_body = {between_body} -> {False}")
            return False
            
def check_n_or_d_shape(env,
                            geom_idx,
                            p_joint_global,
                            axis_world,
                            ray_length=0.5,
                            dist_epsilon=1e-4):
    """Classify the frame shape around a joint by ray casting along its axis.

    Function version of RobotCharacteristics.check_n_or_d_shape: casts rays from
    p_joint_global in the +axis and -axis directions and checks which meshes they hit, to
    tell U-shaped frames (meshes on both sides) from L-shaped ones (mesh on one side).

    Returns:
        dict with 'shape' and the hit points/distances ('pos_hit1', 'pos_hit2', ...).
    """
    # 1) normalize the axis
    axis = axis_world

    # 2) test_pos1, test_pos2
    test_pos1 = p_joint_global + ray_length * axis
    test_pos2 = p_joint_global - ray_length * axis

    # 3) ray casting
    d1 = mj.mj_rayMesh(env.model, env.data, geom_idx,
                    test_pos1, -axis)  # first ray
    d2 = mj.mj_rayMesh(env.model, env.data, geom_idx,
                    test_pos2, axis)   # second ray

    found1 = (d1 >= 0)
    found2 = (d2 >= 0)

    if found1:
        pos_hit1 = test_pos1 + (-axis)*d1
    else:
        pos_hit1 = test_pos1

    if found2:
        pos_hit2 = test_pos2 + axis*d2
    else:
        pos_hit2 = test_pos2

    # 4) in case dist(pos_hit1, pos_hit2) < eps ?
    dist_hit12 = np.linalg.norm(pos_hit1 - pos_hit2)

    # -- handle "single" / "none" first --
    if found1 ^ found2:
        # only one side intersects
        # but pos_hit1 ≈ pos_hit2 is still possible.
        # how to handle this case is up to the user.
        # here 'collapsed' could be tried first, or it could be treated as 'single'.
        if dist_hit12 < dist_epsilon:
            shape_type = 'collapsed'
        else:
            shape_type = 'single'
        return {
            'shape': shape_type,
            'pos_hit1': pos_hit1,
            'pos_hit2': pos_hit2,
            'pos_mid': 0.5*(pos_hit1 + pos_hit2)
        }

    if (not found1) and (not found2):
        # neither intersects
        if dist_hit12 < dist_epsilon:
            # since pos_hit1 and pos_hit2 are 1 m apart anyway
            # dist_hit12 < eps should rarely happen, but
            shape_type = 'collapsed'
        else:
            shape_type = 'none'
        return {
            'shape': shape_type,
            'pos_hit1': pos_hit1,
            'pos_hit2': pos_hit2,
            'pos_mid': 0.5*(pos_hit1 + pos_hit2)
        }

    # here => found1=True & found2=True (both sides intersect)
    # check the 'collapsed' case first
    if dist_hit12 < dist_epsilon:
        # 'pos_hit1' and 'pos_hit2' are effectively the same point
        return {
            'shape': 'collapsed',
            'pos_hit1': pos_hit1,
            'pos_hit2': pos_hit2,
            'pos_mid': pos_hit1  # they are the same, so either one
        }

    # -- the rest is the original L/d logic --
    pos_mid = 0.5*(pos_hit1 + pos_hit2)

    dm_up = mj.mj_rayMesh(env.model, env.data, geom_idx,
                        pos_mid, axis)
    dm_dn = mj.mj_rayMesh(env.model, env.data, geom_idx,
                        pos_mid, -axis)

    if dm_up >= 0:
        pos_hit_up = pos_mid + axis*dm_up
    else:
        pos_hit_up = pos_mid
    if dm_dn >= 0:
        pos_hit_dn = pos_mid + (-axis)*dm_dn
    else:
        pos_hit_dn = pos_mid

    def dist(a, b):
        return np.linalg.norm(a - b)

    same_up_hit1 = (dist(pos_hit_up, pos_hit1) < dist_epsilon)
    same_up_hit2 = (dist(pos_hit_up, pos_hit2) < dist_epsilon)
    same_dn_hit1 = (dist(pos_hit_dn, pos_hit1) < dist_epsilon)
    same_dn_hit2 = (dist(pos_hit_dn, pos_hit2) < dist_epsilon)

    if (same_up_hit1 and same_dn_hit2) or (same_up_hit2 and same_dn_hit1):
        shape_type = 'n'
    else:
        shape_type = 'd'

    return {
        'shape': shape_type,
        'pos_hit1': pos_hit1,
        'pos_hit2': pos_hit2,
        'pos_mid': pos_mid
    }
