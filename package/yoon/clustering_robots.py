"""Robot kinematics in a common representation, and back.

RobotCharacteristics loads the robot MJCFs listed in asset/_robot_info.py, puts each
robot in its A-pose, and encodes its upper-body joints as fixed-size per-group arrays
(screw: axis w + position q, 6 values per joint; DH: 5 values per joint). It also
does the reverse: decodes a flat (interpolated / decoded) vector back into a MuJoCo
model, which is how latent points are turned into robots.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco as mj
import json
from moviepy.editor import *
import cv2
from sklearn.decomposition import PCA
import colorsys
import datetime
from copy import deepcopy
import networkx as nx
import torch
import math
import pathlib
import time
import sys
from transforms import r2quat
from models import Joint, RobotModel
from representations import (
    KinematicsExtractor,
    ScrewKinematicsExtractor,
    SDHKinematicsExtractor,
)
from dataset import KinematicsDatasetBuilder

def is_macos() -> bool:
    """Return True on macOS."""
    return sys.platform == "darwin"

def is_ubuntu() -> bool:
    """Return True on Ubuntu Linux."""
    if sys.platform != "linux":
        return False
    os_release = pathlib.Path("/etc/os-release")
    if os_release.exists():
        with os_release.open() as f:
            data = f.read()
        return "ubuntu" in data.lower()
    return False   # possibly a different Linux


# Add path to import robot info
# add the absolute path of the project root directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
from mujoco_parser import *
from utils import get_idxs, MultiSliderClass, MultiSliderQtClass
from _robot_info import get_robot_info, mjcf_list
from kinematic_chain import KinematicChainClass
from torch_chain import *

a_pose_path = '../../asset/_robot_info/A-pose.json'
with open(a_pose_path, 'r') as f:
        A_POSE_DATA = json.load(f)
robot_names=A_POSE_DATA['robot_name'] # list of robot names that have an a_pose

classify_path = '../../asset/_robot_info/classify.json'
with open(classify_path, 'r') as f:
    CLASSIFY_DATA = json.load(f)

class RobotCharacteristics():
    """Encode robots into the LEGO representation and decode vectors back into robots.

    Typical use:
        rc = RobotCharacteristics()
        rc.max_joints_per_group = np.load('data/mjcf_test/screw/njnt.npy')
        full = rc.build_full_interp_from_symmetric(decoded, threshold=0.5, repr_kind='screw')
        robots = rc.convert_flat_interp_to_dict(full, repr_kind='screw')
        rc.visualize_data_to_classify(robots, save_video=False, vis=False, xml=True)
        xml_string = rc.xml[name]   # one MJCF string per decoded sample

    Args:
        curate: use the curated joint positions (asset/<robot>/*_curated.xml) when available.
    """
    def __init__(self,curate=True):
        self.env = None
        self.classifying_joint_info = {}  # dictionary storing the joint positions of each robot
        self.robot_list = A_POSE_DATA['robot_name']
        self.a_pose_data = A_POSE_DATA
        self.pca_result = None
        self.robot_models={}
        self.classify_data_shape = None
        self.classify_data = CLASSIFY_DATA
        self.curate=curate
        self.mps_or_cpu='cpu'
        self._initialize_robots()
        self._AXIS_EPS = 1e-2
        # define the classification groups (changes classifying_all_joints)
        self.classifying_all_joints = [
            ["torso"],
            ["neck"],
            ["left_shoulder_girdle"], 
            ["left_shoulder"], 
            ["left_upperarm"],
            ["left_elbow"],
            ["left_forearm"],
            ["left_wrist"],
            ["right_shoulder_girdle"],
            ["right_shoulder"],
            ["right_upperarm"],
            ["right_elbow"],
            ["right_forearm"],
            ["right_wrist"]
        ]
        self.classifying_all_joints_dh = [
            ["torso"],
            ["neck"],
            ["left_shoulder_girdle"], 
            ["left_shoulder"], 
            ["left_upperarm"],
            ["left_elbow"],
            ["left_forearm"],
            ["left_wrist"],
            ["left_tcp"],
            ["right_shoulder_girdle"],
            ["right_shoulder"],
            ["right_upperarm"],
            ["right_elbow"],
            ["right_forearm"],
            ["right_wrist"],
            ["right_tcp"]
        ]
        self.classifying_joints_except_torso_and_neck = [
            ["left_shoulder_girdle"], 
            ["left_shoulder"], 
            ["left_upperarm"],
            ["left_elbow"],
            ["left_forearm"],
            ["left_wrist"],
            ["right_shoulder_girdle"],
            ["right_shoulder"],
            ["right_upperarm"],
            ["right_elbow"],
            ["right_forearm"],
            ["right_wrist"],
        ]

        self.classifying_symmetric_joints = [
            ["torso"],
            ["neck"],
            ["right_shoulder_girdle"],
            ["right_shoulder"],
            ["right_upperarm"],
            ["right_elbow"],
            ["right_forearm"],
            ["right_wrist"]
        ]

        self.classifying_symmetric_joints_dh = [
            ["torso"],
            ["neck"],
            ["right_shoulder_girdle"],
            ["right_shoulder"],
            ["right_upperarm"],
            ["right_elbow"],
            ["right_forearm"],
            ["right_wrist"],
            ["right_tcp"]
        ]

        self.fallback_order = {
                'torso': ['base_link'],
                'neck': ['torso', 'base_link'],
                # 'right_shoulder_girdle': ['neck','torso','base_link'],
                # 'right_shoulder': ['right_shoulder_girdle', 'neck', 'torso', 'base_link'],
                # 'right_upperarm': ['right_shoulder', 'right_shoulder_girdle', 'neck','torso', 'base_link'],
                # 'right_elbow': ['right_upperarm', 'right_shoulder', 'right_shoulder_girdle', 'neck', 'torso', 'base_link'],
                # 'right_forearm':['right_elbow','right_upperarm', 'right_shoulder', 'right_shoulder_girdle', 'neck', 'torso', 'base_link'],
                # 'right_wrist': ['right_forearm', 'right_elbow', 'right_upperarm', 'right_shoulder', 'right_shoulder_girdle', 'neck', 'torso', 'base_link'],
                # 'left_shoulder_girdle':['neck', 'torso', 'base_link'],
                # 'left_shoulder': ['left_shoulder_girdle', 'neck', 'torso', 'base_link'],
                # 'left_upperarm': ['left_shoulder', 'left_shoulder_girdle', 'neck', 'torso', 'base_link'],
                # 'left_elbow': ['left_upperarm', 'left_shoulder', 'left_shoulder_girdle', 'neck', 'torso', 'base_link'],
                # 'left_forearm': ['left_elbow', 'left_upperarm', 'left_shoulder', 'left_shoulder_girdle', 'neck', 'torso', 'base_link'],
                # 'left_wrist': ['left_forearm', 'left_elbow', 'left_upperarm', 'left_shoulder', 'left_shoulder_girdle', 'neck', 'torso', 'base_link'],
                'right_shoulder_girdle': ['torso','base_link'],
                'right_shoulder': ['right_shoulder_girdle', 'torso', 'base_link'],
                'right_upperarm': ['right_shoulder', 'right_shoulder_girdle', 'neck','torso', 'base_link'],
                'right_elbow': ['right_upperarm', 'right_shoulder', 'right_shoulder_girdle', 'torso', 'base_link'],
                'right_forearm':['right_elbow','right_upperarm', 'right_shoulder', 'right_shoulder_girdle', 'torso', 'base_link'],
                'right_wrist': ['right_forearm', 'right_elbow', 'right_upperarm', 'right_shoulder', 'right_shoulder_girdle', 'torso', 'base_link'],
                'left_shoulder_girdle':['torso', 'base_link'],
                'left_shoulder': ['left_shoulder_girdle', 'torso', 'base_link'],
                'left_upperarm': ['left_shoulder', 'left_shoulder_girdle', 'torso', 'base_link'],
                'left_elbow': ['left_upperarm', 'left_shoulder', 'left_shoulder_girdle', 'torso', 'base_link'],
                'left_forearm': ['left_elbow', 'left_upperarm', 'left_shoulder', 'left_shoulder_girdle', 'torso', 'base_link'],
                'left_wrist': ['left_forearm', 'left_elbow', 'left_upperarm', 'left_shoulder', 'left_shoulder_girdle', 'torso', 'base_link'],
            }
        self.fallback_order_dh = {
            'torso': ['base_link'],
            'neck': ['torso', 'base_link'],
            'right_shoulder_girdle': ['torso','base_link'],
            'right_shoulder': ['right_shoulder_girdle', 'torso', 'base_link'],
            'right_upperarm': ['right_shoulder', 'right_shoulder_girdle', 'neck','torso', 'base_link'],
            'right_elbow': ['right_upperarm', 'right_shoulder', 'right_shoulder_girdle', 'torso', 'base_link'],
            'right_forearm':['right_elbow','right_upperarm', 'right_shoulder', 'right_shoulder_girdle', 'torso', 'base_link'],
            'right_wrist': ['right_forearm', 'right_elbow', 'right_upperarm', 'right_shoulder', 'right_shoulder_girdle', 'torso', 'base_link'],
            'right_tcp': ['right_wrist','right_forearm', 'right_elbow', 'right_upperarm', 'right_shoulder', 'right_shoulder_girdle', 'torso', 'base_link'],
            'left_shoulder_girdle':['torso', 'base_link'],
            'left_shoulder': ['left_shoulder_girdle', 'torso', 'base_link'],
            'left_upperarm': ['left_shoulder', 'left_shoulder_girdle', 'torso', 'base_link'],
            'left_elbow': ['left_upperarm', 'left_shoulder', 'left_shoulder_girdle', 'torso', 'base_link'],
            'left_forearm': ['left_elbow', 'left_upperarm', 'left_shoulder', 'left_shoulder_girdle', 'torso', 'base_link'],
            'left_wrist': ['left_forearm', 'left_elbow', 'left_upperarm', 'left_shoulder', 'left_shoulder_girdle', 'torso', 'base_link'],
            'left_tcp': ['left_wrist','left_forearm', 'left_elbow', 'left_upperarm', 'left_shoulder', 'left_shoulder_girdle', 'torso', 'base_link'],
        }

    def _find_avg_joint_pos_recursive(self,
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
            self.env.plot_sphere(
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
            d= mj.mj_rayMesh(self.env.model, self.env.data,
                            geom_id1, p_joint_global, axis_world) if geom_id2==-1 else mj.mj_rayMesh(self.env.model, self.env.data,
                            geom_id2, p_joint_global, -axis_world)
            p_joint_global += axis_world * d * move_sign
            return p_joint_global, distance1, distance2
        # ============= 1) both distance1 and distance2 are None?  =============
        if distance1 is None and distance2 is None:
            d1 = mj.mj_rayMesh(self.env.model, self.env.data,
                            geom_id1, p_joint_global, axis_world)
            d2 = mj.mj_rayMesh(self.env.model, self.env.data,
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
                self.env.plot_sphere(
                            p=p_joint_global,
                            r=0.03,  # sphere radius
                            rgba=[0,0,1,1],
                            )
            # after moving, cast rays on both the positive and negative side again and update
            d1 = mj.mj_rayMesh(self.env.model, self.env.data,
                            geom_id1, p_joint_global, axis_world)
            d2 = mj.mj_rayMesh(self.env.model, self.env.data,
                            geom_id2, p_joint_global, -axis_world)

            distance1 = None if d1 == -1 else d1
            distance2 = None if d2 == -1 else d2
            if vis:
                self.env.render()
                print(f"iter:{iteration}, d2 was None -> now dist1: {distance1}, dist2: {distance2}")
                time.sleep(0.5)
            return self._find_avg_joint_pos_recursive(
                geom_id, p_joint_global, axis_world,
                distance1, distance2,
                th, tol_diff, max_iter, iteration+1, vis
            )

        elif distance1 is None and distance2 is not None:
            # only distance2 exists -> there is a mesh in the -axis direction
            # move p_joint_global by th toward -axis
            p_joint_global -= axis_world * (distance2 + th)
            if vis:
                self.env.plot_sphere(
                            p=p_joint_global,
                            r=0.03,  # sphere radius
                            rgba=[0,0,1,0.5],
                            )
            # cast the ray again after moving
            d1 = mj.mj_rayMesh(self.env.model, self.env.data,
                            geom_id1, p_joint_global, axis_world)
            d2 = mj.mj_rayMesh(self.env.model, self.nv.data,
                            geom_id2, p_joint_global, -axis_world)

            distance1 = None if d1 == -1.0 else d1
            distance2 = None if d2 == -1.0 else d2
            if vis:
                self.env.render()
                print(f"iter:{iteration}, d1 was None -> now dist1: {distance1}, dist2: {distance2}")
                time.sleep(0.5)
            return self._find_avg_joint_pos_recursive(
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
                    d3 = mj.mj_rayMesh(self.env.model, self.env.data,
                                geom_id1, p_joint_global, -axis_world)
                    d4 = mj.mj_rayMesh(self.env.model, self.env.data,
                                geom_id2, p_joint_global, axis_world)
                    if d3!=-1.0 and d4==-1.0: # |parent|<--axis--|body|
                        p_joint_global -= axis_world * d3
                    elif d3==-1.0 and d4!=-1.0: # |body|<--axis--|parent|
                        p_joint_global += axis_world * d4
                if vis:
                    self.env.plot_sphere(
                            p=p_joint_global,
                            r=0.03,  # sphere radius
                            rgba=[0,0,1,1],
                            )
                    self.env.render()
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
                    self.env.plot_sphere(
                                p=p_joint_global,
                                r=0.03,  # sphere radius
                                rgba=[0,0,1,1],
                                )
                # ray again after moving
                d1 = mj.mj_rayMesh(self.env.model, self.env.data,
                                geom_id1, p_joint_global, axis_world)
                d2 = mj.mj_rayMesh(self.env.model, self.env.data,
                                geom_id2, p_joint_global, -axis_world)

                distance1 = None if d1 == -1 else d1
                distance2 = None if d2 == -1 else d2
                if vis:
                    self.env.render()
                    print(f"iter:{iteration}, moving slightly to middle point -> now dist1: {distance1}, dist2: {distance2}")
                    time.sleep(0.5)
                return self._find_avg_joint_pos_recursive(
                    geom_id, p_joint_global, axis_world,
                    distance1, distance2,
                    th, tol_diff, max_iter, iteration+1, vis
                )

        # ============= 4) otherwise (neither both None nor caught by the ifs above) =============
        # this is almost never reached, but for safety
        if vis:
            print("both are not none, and couldn't pass if")
            self.env.render()
            print(f"iter:{iteration}, p_glo: {p_joint_global},d1: {d1}, d2: {d2}")
        return p_joint_global, distance1, distance2

    def _find_valid_geoms_in_joint(self, p_joint_global, axis_world,body_id, verbose=False):
        """
        find a mesh geom among the geoms of body_id.
        (a body can actually have several geoms, so the name or type should be checked more)
        """
        body_name=mj.mj_id2name(self.env.model,mj.mjtObj.mjOBJ_BODY,body_id)
        if verbose:
            print(f"body_name: {body_name}")
        # the geoms of a body are model.body_geomnum[body_id] geoms starting at model.body_geomadr[body_id]
        parent_geom_idx=None
        geom_idx=None
        parent_id = self.env.model.body_parentid[body_id]
        start_g = self.env.model.body_geomadr[parent_id]
        count_g = self.env.model.body_geomnum[parent_id]
        if count_g==0:
            if verbose:
                print(f"{body_name}: no mesh in parent")
            parent_geom_idx = None
        else:
            distance=[]
            for gid in range(start_g, start_g + count_g):
                geom_type = self.env.model.geom_type[gid]
                d=0.0
                if  geom_type == mj.mjtGeom.mjGEOM_MESH:
                    d1 = mj.mj_rayMesh(self.env.model, self.env.data, gid, p_joint_global, axis_world)
                    d2 = mj.mj_rayMesh(self.env.model, self.env.data, gid, p_joint_global, -axis_world)
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
        start_g = self.env.model.body_geomadr[body_id]
        count_g = self.env.model.body_geomnum[body_id]
        if count_g==0:
            if verbose:
                print(f"{body_name}: no mesh in this body")
            geom_idx = None
        else:
            distance=[]
            for gid in range(start_g, start_g + count_g):
                geom_type = self.env.model.geom_type[gid]
                d=0.0
                if  geom_type == mj.mjtGeom.mjGEOM_MESH:
                    d1 = mj.mj_rayMesh(self.env.model, self.env.data, gid, p_joint_global, axis_world)
                    d2 = mj.mj_rayMesh(self.env.model, self.env.data, gid, p_joint_global, -axis_world)
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
            d1_parent = mj.mj_rayMesh(self.env.model, self.env.data, parent_geom_idx, p_joint_global, axis_world)
            d2_parent = mj.mj_rayMesh(self.env.model, self.env.data, parent_geom_idx, p_joint_global, -axis_world)
            d1_body = mj.mj_rayMesh(self.env.model, self.env.data, geom_idx, p_joint_global, axis_world)
            d2_body = mj.mj_rayMesh(self.env.model, self.env.data, geom_idx, p_joint_global, -axis_world)
            
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
                
    def check_n_or_d_shape(self,
                                geom_idx,
                                p_joint_global,
                                axis_world,
                                ray_length=0.5,
                                dist_epsilon=1e-4):
        """Classify the frame shape around a joint by ray casting along its axis.

        Casts rays from p_joint_global in the +axis and -axis directions and checks which
        meshes they hit, to tell U-shaped frames (meshes on both sides) from L-shaped ones
        (mesh on one side). Used during curation to move joints onto the motor axis.

        Returns:
            dict with 'shape' and the hit points/distances ('pos_hit1', 'pos_hit2', ...).
        """
        # 1) normalize the axis
        axis = axis_world

        # 2) test_pos1, test_pos2
        test_pos1 = p_joint_global + ray_length * axis
        test_pos2 = p_joint_global - ray_length * axis

        # 3) ray casting
        d1 = mj.mj_rayMesh(self.env.model, self.env.data, geom_idx,
                        test_pos1, -axis)  # first ray
        d2 = mj.mj_rayMesh(self.env.model, self.env.data, geom_idx,
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

        dm_up = mj.mj_rayMesh(self.env.model, self.env.data, geom_idx,
                            pos_mid, axis)
        dm_dn = mj.mj_rayMesh(self.env.model, self.env.data, geom_idx,
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

    def _draw_trees(self,verbose=False):
        for i in range(0,len(self.robot_list)):
            robot_name=self.robot_list[i]
            if verbose:
                print("current robot name: ", robot_name)
            self._initialize_env(robot_name)

            tc = TorchChainClass(name=self.env.name,device=self.mps_or_cpu) # 'cpu' / 'mps'
            tc.build_chain_from_mujoco_env(env=self.env)
            # Forward kinematics
            tc.forward(root_body_name=self.env.body_names[0])
            n_body=tc.get_n_body()
            tree=tc.chain
            for layer, nodes in enumerate(nx.topological_generations(tree)):
                    for node in nodes:
                        tree.nodes[node]['layer'] = layer

            # Plot nodes
            fig, ax = plt.subplots(figsize=(10,6))
            node_colors = []
            for body_idx in range(n_body):
                a = tc.get_body(body_idx)['a']
                if torch.norm(a) < 1e-6:
                    node_color = (1,1,1,0.5)
                else:
                    node_color=tuple(np.append(np.eye(3)[:,torch.argmax(torch.abs(a)).item()],0.5))
                node_colors.append(node_color)
            nx.draw_networkx(tree, 
                            pos=nx.drawing.nx_agraph.graphviz_layout(tree, prog='dot'),
                            node_color=node_colors,
                            linewidths=1,
                            edgecolors='k',
                            )
            ax.set_title(f'{tree.name}', fontsize=10)
            fig.tight_layout()
            save_dir='./tree/'
            save_name = str(tree.name)+'_tree.png'

            # for example, assume saving in the same directory without project_root
            save_path = os.path.join(project_root,save_dir, save_name)
            plt.savefig(save_path)
            if verbose:
                plt.show()
                plt.close(fig)
            else:
                plt.close(fig)

    def _initialize_robots(self):
        # get the list of all robots
        """Initialize only robots that currently have an A-pose"""
        
        # initialize joint info for each robot
        for robot_name in self.robot_list:
            try:
                self.classifying_joint_info[robot_name] = self._get_robot_joint_info(robot_name)
            except Exception as e:
                print(f"error in processing {robot_name}: {str(e)}")
                continue
        print(f"RobotCharacteristics for {len(self.classifying_joint_info.keys())}initialized")
    
    def _initialize_env(self,robot_name=None):
        """
        Visualize the joint positions and rotation axes of the selected robot
        Args:
            robot_name: name of the robot to visualize
        """
        if robot_name is None:
            rel_xml_path = '../../asset/floor/floor_nothing.xml'
            self.env = MuJoCoParserClass(name=robot_name, rel_xml_path=rel_xml_path, verbose=False, use_spec=True)
            self.env.reset()
        else:
            # bringing robot information
            robot_info = get_robot_info(robot_name)
            if self.curate and robot_info['is_curated']:
                path = robot_info['env_curated_path']
            else:
                path = robot_info['env_path']
            # MuJoCo env initialization
            self.env = MuJoCoParserClass(name=robot_name, rel_xml_path=path, verbose=False, use_spec=True)
            self.env.reset()
            self.set_robot_a_pose(robot_name)
    
    def _get_rainbow_color(self,i, n):
        """
        i: current idx (0 <= i < n)
        n: total number
        """
        # spread Hue over 0 ~ 1 in HSV space
        h = i / n
        s = 1.0    # saturation
        v = 1.0    # value (brightness)

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return [r, g, b, 1.0]  # RGBA (alpha=1.0)

    def _get_robot_joint_info_old(self, robot_name, verbose=False):
        # bringing robot information
        if verbose:
            print(f"robot_name: {robot_name}")
        
        # MuJoCo parser initialization
        self._initialize_env(robot_name)

        if verbose:
            vis=True
            try:
                self.env.is_viewer_alive()
            except:
                self.env.init_viewer(
                        transparent       = True,
                        black_sky         = False,
                        azimuth           = 180, # None,
                        distance          = 5.0, # None,
                        elevation         = -20, # None,
                        lookat            = [0.0,0.0,0.3], # None,
                        use_rgb_overlay   = False,
                        loc_rgb_overlay   = 'top right',
                        fontscale         = mj.mjtFontScale.mjFONTSCALE_50.value,
                        )
        else:
            vis=False
        # store per-joint info in a dictionary
        joint_info = {}
        etc_joint_names = self.get_etc_joint_names(robot_name)
        joint_names_to_fix = self.get_joint_names_to_fix(robot_name)
        if len(joint_names_to_fix)!=0 and self.curate and verbose:
            print(f"{robot_name} joint_names_to_fix: {joint_names_to_fix}")
        
        
        # keep only joint_names that are not in etc_joint_names
        filtered_joint_names = [joint_name for joint_name in self.env.rev_joint_names if joint_name not in etc_joint_names]
        virtual_joint_names = [
                name
                for sublist in CLASSIFY_DATA[robot_name].values()
                for name in sublist
                if name.startswith("virtual_")
            ]
        for joint_name in filtered_joint_names:
            if joint_name is None:  # added None check
                print(f"joint_name is None: {robot_name}")
                continue
                
            joint_idx = mj.mj_name2id(self.env.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            try:
                if self.env.model.jnt_type[joint_idx] == mj.mjtJoint.mjJNT_SLIDE:
                    joint_type = 'slide'
                    axis_joint = np.array([0,0,0])
                elif self.env.model.jnt_type[joint_idx] == mj.mjtJoint.mjJNT_HINGE:
                    joint_type = 'revolute'
                    axis_joint = self.env.model.jnt_axis[joint_idx]
                elif self.env.model.jnt_type[joint_idx] == mj.mjtJoint.mjJNT_FREE:
                    joint_type = 'free'
                    axis_joint = np.array([0,0,0])
                else:
                    raise ValueError(f"Unsupported joint type: {self.env.model.jnt_type[joint_idx]}")
                R_joint = self.env.get_R_joint(joint_name=joint_name)
                p_joint_global = self.env.data.xanchor[joint_idx]
                axis_world = R_joint@axis_joint
                joint_tag = self._classify_joint_type(axis_world)
                if joint_name in joint_names_to_fix and self.curate:
                    body_id = self.env.model.jnt_bodyid[joint_idx]
                    geom_ids = self._find_valid_geoms_in_joint(p_joint_global, axis_world, body_id, verbose=verbose)
                    if joint_tag == 'roll' or joint_tag == 'pitch':
                        if geom_ids:
                            p_result, _, _ = self._find_avg_joint_pos_recursive(
                                geom_ids,
                                p_joint_global,
                                axis_world,
                                th=0.001,
                                tol_diff=0.001,
                                max_iter=200,
                                iteration=0,
                                vis=vis
                            )
                            p_joint_global = p_result
                joint_info[joint_name] = {
                    'position': p_joint_global,
                    'type': joint_type,
                    'joint_tag': joint_tag,
                    'axis_world': axis_world
                }
            except Exception as e:
                print(f"Error while processing joint {joint_name}: {str(e)}")
                continue
        for virtual_joint_name in virtual_joint_names:
            p_joint_global = self.env.get_p_site(virtual_joint_name)
            joint_type = "virtual"
            joint_tag = "virtual"
            axis_world = np.array([0,0,0])
            joint_info[virtual_joint_name] = {
                    'position': p_joint_global,
                    'type': joint_type,
                    'joint_tag': joint_tag,
                    'axis_world': axis_world
                }
            
        if vis:
            self.env.close_viewer()
        return joint_info
    
    def _get_robot_joint_info(self, robot_name, verbose=False):
        # bringing robot information
        if verbose:
            print(f"robot_name: {robot_name}")
        
        # MuJoCo parser initialization
        self._initialize_env(robot_name)

        if verbose:
            vis=True
            try:
                self.env.is_viewer_alive()
            except:
                self.env.init_viewer(
                        transparent       = True,
                        black_sky         = False,
                        azimuth           = 180, # None,
                        distance          = 5.0, # None,
                        elevation         = -20, # None,
                        lookat            = [0.0,0.0,0.3], # None,
                        use_rgb_overlay   = False,
                        loc_rgb_overlay   = 'top right',
                        fontscale         = mj.mjtFontScale.mjFONTSCALE_50.value,
                        )
        else:
            vis=False
        # store per-joint info in a dictionary
        joint_info = {} # backward-compat output (dict of dict)
        # also build structured model in parallel (no behavior change)
        robot_model = RobotModel(
            name=robot_name,
            ignored_joints=self.get_etc_joint_names(robot_name),
            fixed_joints=self.get_joint_names_to_fix(robot_name),
        )
        etc_joint_names = self.get_etc_joint_names(robot_name)
        joint_names_to_fix = self.get_joint_names_to_fix(robot_name)
        if len(joint_names_to_fix)!=0 and self.curate and verbose:
            print(f"{robot_name} joint_names_to_fix: {joint_names_to_fix}")
        
        
        # keep only joint_names that are not in etc_joint_names
        filtered_joint_names = [joint_name for joint_name in self.env.rev_joint_names if joint_name not in etc_joint_names]
        virtual_joint_names = [
                name
                for sublist in CLASSIFY_DATA[robot_name].values()
                for name in sublist
                if name.startswith("virtual_")
            ]
        for joint_name in filtered_joint_names:
            if joint_name is None:  # added None check
                print(f"joint_name is None: {robot_name}")
                continue
                
            joint_idx = mj.mj_name2id(self.env.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            try:
                if self.env.model.jnt_type[joint_idx] == mj.mjtJoint.mjJNT_SLIDE:
                    joint_type = 'slide'
                    axis_joint = np.array([0,0,0])
                elif self.env.model.jnt_type[joint_idx] == mj.mjtJoint.mjJNT_HINGE:
                    joint_type = 'revolute'
                    axis_joint = self.env.model.jnt_axis[joint_idx]
                elif self.env.model.jnt_type[joint_idx] == mj.mjtJoint.mjJNT_FREE:
                    joint_type = 'free'
                    axis_joint = np.array([0,0,0])
                else:
                    raise ValueError(f"Unsupported joint type: {self.env.model.jnt_type[joint_idx]}")
                R_joint = self.env.get_R_joint(joint_name=joint_name)
                p_joint_global = self.env.data.xanchor[joint_idx]
                axis_world = R_joint@axis_joint
                joint_tag = self._classify_joint_type(axis_world)
                if joint_name in joint_names_to_fix and self.curate:
                    body_id = self.env.model.jnt_bodyid[joint_idx]
                    geom_ids = self._find_valid_geoms_in_joint(p_joint_global, axis_world, body_id, verbose=verbose)
                    if joint_tag == 'roll' or joint_tag == 'pitch':
                        if geom_ids:
                            p_result, _, _ = self._find_avg_joint_pos_recursive(
                                geom_ids,
                                p_joint_global,
                                axis_world,
                                th=0.001,
                                tol_diff=0.001,
                                max_iter=200,
                                iteration=0,
                                vis=vis
                            )
                            p_joint_global = p_result
                j = Joint(
                    name=joint_name,
                    joint_type=joint_type,
                    axis_world=axis_world,
                    position=p_joint_global,
                    joint_tag=joint_tag,
                )
                robot_model.add_joint(j)
                joint_info[joint_name] = j.to_dict()
            except Exception as e:
                print(f"Error while processing joint {joint_name}: {str(e)}")
                continue
        for virtual_joint_name in virtual_joint_names:
            p_joint_global = self.env.get_p_site(virtual_joint_name)
            joint_type = "virtual"
            joint_tag = "virtual"
            axis_world = np.array([0,0,0])
            jv = Joint(
                name=virtual_joint_name,
                joint_type=joint_type,
                axis_world=axis_world,
                position=p_joint_global,
                joint_tag=joint_tag,
            )
            robot_model.add_joint(jv)
            joint_info[virtual_joint_name] = jv.to_dict()
            
        if vis:
            self.env.close_viewer()
        # keep both: legacy dict + structured model for later stages
        self.robot_models[robot_name] = robot_model
        return joint_info

    def get_representation(self, robot_name: str,
                           extractor: KinematicsExtractor | None = None):
        """
        Returns a representation dict keyed by joint name.
        Default = screw (w, q). Behavior of the rest of the pipeline unchanged.
        """
        # ensure RobotModel is prepared
        if robot_name not in self.robot_models:
            # build joint info (also populates self.robot_models)
            _ = self._get_robot_joint_info(robot_name)
        model = self.robot_models[robot_name]
        if extractor is None:
            extractor = ScrewKinematicsExtractor()
        return extractor.extract(model)

    def _classify_joint_type(self, axis):
        # convert the given axis to a numpy array
        axis = np.array(axis, dtype=float)
        # define reference axes
        roll_axis = np.array([1, 0, 0])
        pitch_axis = np.array([0, 1, 0])
        yaw_axis = np.array([0, 0, 1])
        
        # compute the angle to each axis
        cos_roll = np.abs(np.dot(axis, roll_axis))
        cos_pitch = np.abs(np.dot(axis, pitch_axis))
        cos_yaw = np.abs(np.dot(axis, yaw_axis))
        
        # pick the most similar axis
        max_cos = max(cos_roll, cos_pitch, cos_yaw)
        if max_cos == cos_roll:
            return 'roll'
        elif max_cos == cos_pitch:
            return 'pitch'
        elif max_cos == cos_yaw:
            return 'yaw'
        else:
            return 'unknown'
    
    def visualize_robot_joints(self):
        # initialize 3D plot
        """Scatter-plot the joint positions of all loaded robots in 3D."""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # visualize the joint positions of each robot
        for robot_name, joints in self.cartesian_joint_pos.items():
            positions = np.array([joint_data['position'] for joint_data in joints.values()])
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], label=robot_name)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Robot joint positions')
        ax.legend()
        plt.show()

    def print_joint_tags(self):
        """Print the joint grouping (classification tags) of every robot."""
        for robot_name, joints in self.classifying_joint_info.items():
            joint_tags = [joint_data['joint_tag'] for joint_data in joints.values()]
            print(f"{robot_name}: {joint_tags}\n")
    
    def plot_mjdata_joint_axis(
            self,
            axis_len    = 0.1,
            p_base      = np.array([0,0,0]),
            axis_r      = 0.01,
            joint_names = None,
            rate        = 1.0,
            print_name  = False,
        ):
        """ 
            Plot revolute joint axis 
        """
        if self.env is None:
            raise Exception(f"env is not defined")
        rev_joint_idxs  = self.env.rev_joint_idxs
        rev_joint_names = self.env.rev_joint_names
        
        if joint_names is not None:
            idxs = get_idxs(rev_joint_names,joint_names)
            rev_joint_idxs_to_use  = rev_joint_idxs[idxs]
            rev_joint_names_to_use = [rev_joint_names[i] for i in idxs]
        else:
            rev_joint_idxs_to_use  = rev_joint_idxs
            rev_joint_names_to_use = rev_joint_names

        for rev_joint_idx,rev_joint_name in zip(rev_joint_idxs_to_use,rev_joint_names_to_use):
            axis_joint      = self.env.model.jnt_axis[rev_joint_idx]
            p_joint,R_joint = self.env.get_pR_joint(joint_name=rev_joint_name)
            axis_world      = R_joint@axis_joint
            axis_rgba       = np.append(np.eye(3)[:,np.argmax(np.abs(axis_world))],0.8)
            p_fr = (p_joint+p_base).reshape((3,))
            p_to = (p_joint+rate*axis_len*axis_world+p_base).reshape((3,))
            self.env.plot_arrow_fr2to(
                p_fr = p_fr,
                p_to = p_to,
                r    = rate*axis_r,
                rgba = axis_rgba
            )
            if print_name:
                self.env.plot_text(p=p_joint+p_base,label=rev_joint_name)

    def set_robot_a_pose(self,robot_name,a_pose_data=A_POSE_DATA):
        """Set the loaded MuJoCo model of robot_name to its A-pose (values from a_pose_data)."""
        a_pose_data=a_pose_data['A-pose'][robot_name]
        joint_names = list(a_pose_data.keys())
        joint_values = list(a_pose_data.values())
        joint_values_converted = []
        for value in joint_values:
            if 'pi' in value:
                # convert strings containing pi into numpy expressions
                expr = value.replace('pi', 'np.pi')
                converted_value = float(eval(expr))
                joint_values_converted.append(converted_value)
            else:
                # convert plain number strings to float
                joint_values_converted.append(float(value))
        joint_values = joint_values_converted
        self.env.forward(joint_names=joint_names, q=joint_values)

    def get_etc_joint_names(self,robot_name,classify_data=CLASSIFY_DATA):
        """Return the joints of robot_name that are in the 'etc' category (not used by the representation)."""
        try:
            etc_joint_names = classify_data[robot_name]["etc"]
        except:
            etc_joint_names = []
        return etc_joint_names
    
    def get_joint_names_to_fix(self,robot_name,classify_data=CLASSIFY_DATA):
        """Return the joints of robot_name that are kept fixed when building the representation."""
        try:
            joint_names_to_fix = classify_data[robot_name]["fix"]
        except:
            joint_names_to_fix = []
        return joint_names_to_fix
    
    def visualize_robot_joints_with_arrows(self, robot_name, classify_data=CLASSIFY_DATA):
        """Plot the joints of the selected robots in 3D as spheres with arrows along their axes."""
        self.env.init_viewer(
            distance          = 4.5,
            transparent       = True,
            black_sky         = True,
            use_rgb_overlay = True,
            loc_rgb_overlay = 'bottom right',
            fontscale         = mj.mjtFontScale.mjFONTSCALE_50.value,
            )
        joint_names_to_fix = self.get_joint_names_to_fix(robot_name)
        # load tree image
        tree_img_path = os.path.join(project_root,'tree', f'{robot_name}_tree.png')
        if os.path.exists(tree_img_path):
            tree_img = cv2.imread(tree_img_path)
            tree_img = cv2.cvtColor(tree_img, cv2.COLOR_BGR2RGB)  # BGR to RGB
        else:
            tree_img = None
            print(f"Warning: Tree image not found at {tree_img_path}")
        # Loop
        while self.env.is_viewer_alive():
            
            # Update
            self.env.forward()

            # Render
            if self.env.loop_every(tick_every=10):
                self.env.plot_time()
                # draw a sphere and an arrow for each joint
                joint_names_to_plot = []
                for joint_idx in self.env.rev_joint_idxs:
                    joint_name = self.env.model.joint(joint_idx).name
                    etc_joint_names = self.get_etc_joint_names(robot_name)
                    # skip joint_names in the 'etc' category
                    if joint_name in etc_joint_names :
                        continue
                    else:
                        joint_names_to_plot.append(joint_name)
                        # get joint positions and rotation axes
                        p_joint = self.classifying_joint_info[robot_name][joint_name]['position']
                        axis_world = self.classifying_joint_info[robot_name][joint_name]['axis_world']
                        if joint_name in joint_names_to_fix and self.curate:
                            body_id = self.env.model.joint(joint_name).bodyid[0]
                            body_name = mj.mj_id2name(self.env.model, mj.mjtObj.mjOBJ_BODY, body_id)
                            R = self.env.data.xmat[body_id].reshape(3,3)
                            t = self.env.data.xpos[body_id]
                            T = np.eye(4)
                            T[:3,:3] = R
                            T[:3, 3] = t
                            T_global2local = np.linalg.inv(T)
                            joint_local_pos = (T_global2local@(np.array([p_joint[0],p_joint[1],p_joint[2],1]).T)).T[0:3]
                            label=str(joint_local_pos)
                            rgba=[0.5,0,0.5,0.5]
                        else:
                            rgba=[1,1,1,0.5]
                            label=str(joint_name)

                        # draw joint positions as spheres
                        self.env.plot_sphere(
                            p=p_joint,
                            r=0.03,
                            rgba=rgba,
                            label=label
                        )
                        self.env.plot_arrow_fr2to(
                            p_fr = p_joint,
                            p_to = p_joint+axis_world*0.1,
                            r    = 0.01,
                            rgba = np.append(np.eye(3)[:,np.argmax(np.abs(axis_world))],0.8)
                        )
                    
                # show tree image
                if tree_img is not None:
                    self.env.viewer.add_rgb_overlay(rgb_img_raw=tree_img, fix_ratio=True)
                
                self.env.render()

    def compute_spherical_joint_center(p_list, d_list):
        """
        p_list: shape (N, 3), reference point of each joint
        d_list: shape (N, 3), unit direction vector of each joint
        """
        I = np.eye(3)
        A_sum = np.zeros((3, 3))
        b_sum = np.zeros(3)
        
        for p, d in zip(p_list, d_list):
            d = d / np.linalg.norm(d)  # ensure unit vector
            A = I - np.outer(d, d)
            A_sum += A
            b_sum += A.dot(p)
        
        # assume A_sum is invertible
        x = np.linalg.inv(A_sum).dot(b_sum)
        return x

    def create_data_to_classify(
            self,
            repr_kind: str = "screw",      # "screw" | "dh"
            use_shoulder_pos: bool = True,
            axis_fill: str = "zero",
            pos_fill: str = "local_average",
            classify_joints: str = "all",
            normalize: str = "each",
            save: bool = True,
            verbose: bool = False,
        ):
        """
        (behavior unchanged) delegates the original (w,q) path to KinematicsDatasetBuilder.
        Save format/path/shape/selected_indices are the same as before.
        """
        builder = KinematicsDatasetBuilder(
            repr_kind=repr_kind,
            classify_joints=classify_joints,
            axis_fill=axis_fill,
            pos_fill=pos_fill,
            normalize=normalize,
            use_shoulder_pos=use_shoulder_pos,
        )
        dataset = builder.build(self, verbose=verbose)

        # save
        if save:
            tag = 'screw' if repr_kind == 'screw' else 'dh'
            save_dir = 'data/mjcf_test/'+ tag + '/'
            
            save_name = 'data.npy'
            label_name = 'label.npy'
            njnt_name  = 'njnt.npy'
            indices_name = 'indices.npy'
            save_path   = os.path.join(project_root, save_dir, save_name)
            label_path  = os.path.join(project_root, save_dir, label_name)
            njnt_path   = os.path.join(project_root, save_dir, njnt_name)
            indices_path= os.path.join(project_root, save_dir, indices_name)
            np.save(save_path, dataset.arrays)
            np.save(label_path, np.array(dataset.labels))
            np.save(njnt_path, np.array(self.max_joints_per_group))
            np.save(indices_path, np.array(dataset.selected_indices))
            print(f"[Info] data_to_classify({repr_kind}) saved in '{save_path}'. shape={dataset.arrays.shape}")

        return dataset

    def get_avg_neck_and_base(self,
                robot_name,
                use_shoulder_pos=True,
                verbose=False,
                                ):
        """Return (avg_base, avg_neck), the mean base and neck positions of robot_name.

        The neck-base distance is the length scale used to normalize every robot.
        """
        robot_info = get_robot_info(robot_name)
        # 4-1) compute the mean position of the joints in the neck, torso, shoulder parts
        #     if the neck or torso key itself is missing, set diff=0 (or raise)
        neck_joint_names = self.classify_data[robot_name].get("neck", [])
        torso_joint_names = self.classify_data[robot_name].get("torso", [])

        if verbose:
            print("neck_joint_names: ",neck_joint_names)
            print("torso_joint_names: ",torso_joint_names)
        neck_positions = []
        torso_positions = []
        
        # neck_position
        for joint_name in neck_joint_names:
            if joint_name in self.classifying_joint_info[robot_name]:
                # pos could be found in data, but it can be taken directly if it is already in joint_info
                pos = self.classifying_joint_info[robot_name][joint_name]['position']
                neck_positions.append(pos)
        #torso_position
        for joint_name in torso_joint_names:
            if joint_name in self.classifying_joint_info[robot_name]:
                pos = self.classifying_joint_info[robot_name][joint_name]['position']
                torso_positions.append(pos)
        
        
        base_positions = []
        if robot_info['leg_fixed']: #base position
            self._initialize_env(robot_name)
            base_link = self.env.body_names[1] # [0]: world, [1]: base/base_link/Tibia/etc..
            if verbose:
                print("leg_fixed")
                print(f"base link in robot {robot_name}: {base_link}")
            base_positions.append(self.env.get_p_body(base_link))
        else: # pelvis_position
            if verbose:
                print('getting pelvis position')
            base_joint_names = [
                joint for key, joint in self.classify_data[robot_name].items() 
                if "pelvis" in key
                ]
            base_joint_names = [joint for sublist in base_joint_names for joint in sublist]
            for joint_name in base_joint_names:
                pos = self.classifying_joint_info[robot_name][joint_name]['position']
                base_positions.append(pos)
        if use_shoulder_pos:
            if verbose:
                print("using shoulder position as neck")
            shoulder_joint_names = [
                joint for key, joint in self.classify_data[robot_name].items() 
                if "shoulder" in key
                ]
            shoulder_joint_names = [joint for sublist in shoulder_joint_names for joint in sublist]
            shoulder_positions = []
            for joint_name in shoulder_joint_names:
                pos = self.classifying_joint_info[robot_name][joint_name]['position']
                shoulder_positions.append(pos)
            avg_neck = np.mean(shoulder_positions, axis=0)
        else:
            if verbose:
                print("using neck position as neck")
            avg_neck = np.mean(neck_positions, axis=0)
            if len(neck_positions)==0:
                raise ValueError(f"There's no neck in this robot: {robot_name}")
        
        if len(base_positions) > 0:
            avg_base = np.mean(base_positions, axis=0)
        else:
            avg_base = np.array([0., 0., 0.])
        if verbose:
            print("avg_neck: ", avg_neck)
            print("avg_base: ", avg_base)
        return avg_base, avg_neck
    
    def reconstruct_kinematic_chain(self, name,robot_data,repr_kind: str = "screw"):
        """Order the decoded joints of one robot into a kinematic chain.

        Sorts the valid joints of each group, applies the fallback order when a group is empty
        and stores the result in self.sorted_data, self.ref_data and self.ref_index, which
        build_spec() then turns into bodies and joints.
        """
        if repr_kind=="screw":
            all_joint_names = [j[0] for j in self.classifying_all_joints]
            fallback_dict = self.fallback_order
        elif repr_kind=="dh":
            all_joint_names = [j[0] for j in self.classifying_all_joints_dh]
            fallback_dict = self.fallback_order_dh
        # create variables
        sorted_data = deepcopy(robot_data)
        invalid_data = deepcopy(robot_data)
        ref_data = deepcopy(robot_data)
        ref_index = deepcopy(robot_data)

        for all_joint_names_idx, part in enumerate(all_joint_names): # classifying invalid/valid joints
            group = robot_data.get(part, [])
            valid_group = self.get_group(group, group_type='valid')
            invalid_group = self.get_group(group, group_type='invalid')

            if repr_kind=="screw" and part.endswith("_wrist") and len(valid_group)==0:
                valid_group   = group[[0], :]     # shape: (1, cols), keeps the dimension
                invalid_group = group[1:, :]      # shape: (n-1, cols)
            else:
                if len(valid_group) == 0:
                    sorted_data[part] = []
                if len(invalid_group) == 0:
                    invalid_data[part] = []

            # ---- Find fallback ref ---- #
            ref_name, ref_idx = None, 0
            fallback_list = fallback_dict.get(part,[])
            for candidate in fallback_list:
                source = sorted_data if candidate in sorted_data else robot_data
                cand_group = self.get_group(source.get(candidate, []))
                cand_group = self.get_group(source.get(candidate, []))
                if len(cand_group) > 0:
                    ref_idx = len(cand_group) - 1 # neck removed from the girdle fallback order (it would add an extra DoF)
                    ref_name = candidate
                    break
            # this used to sort by height, but not sorting seems right, so leave it out for now
            sorted_data[part] = np.array(valid_group)
            invalid_data[part] = np.array(invalid_group)
            ref_data[part] = ref_name if ref_name is not None else 'base_link'
            ref_index[part] = ref_idx

        self.sorted_data = sorted_data
        self.ref_data = ref_data
        self.ref_index = ref_index
    
    def build_spec(self, robot_data,repr_kind="screw"):
        """Build the MuJoCo spec (bodies, joints, capsule links) from self.sorted_data and recompile the model."""
        body = self.env.spec.worldbody.add_body(
                                name="base_link",
                                pos = [0.0,0.0,0.0]
                                    )
        self.env.spec.compile()
        self.env.forward(increase_tick=False)
        if self.sorted_data is None:
            raise ValueError("sorted data is None")
        for group_idx, part in enumerate(self.sorted_data): # generating link & joints
            group_data = np.array(self.sorted_data[part])
            n_joint = group_data.shape[0]
            if n_joint!=0:
                for jnt_idx in range(n_joint):
                    # # get joint positions and rotation axes
                    joint_color = self._get_rainbow_color(group_idx,len(robot_data.keys()))
                    body_name=part+'_'+str(jnt_idx)
                    # jnt_idx==0: should be connected to parent in other part
                    if jnt_idx == 0:
                        # if base_link: parent==ground
                        # else: connecting point is in parent body
                        if self.ref_data[part]=="base_link":
                            parent_name = "base_link"
                            from_pos=np.array([0.0,0.0,0.0])
                        else:
                            data_part = self.ref_data[part] # finding parent
                            data_ref_idx = self.ref_index[part]
                            parent_name = data_part+'_'+str(self.ref_index[part])
                            # the parent's q exists only for screw (6D). DH is drawn locally, so leave it as 0.
                            parent_row = np.asarray(self.sorted_data[data_part][data_ref_idx])
                            from_pos = parent_row[3:6] if repr_kind=="screw" else np.array([0.0,0.0,0.0])
                        parent_body = self.env.spec.body(parent_name)
                    # jnt_idx!=0: should be connected to parent in same part   
                    else:
                        data_part = part
                        parent_name = data_part+'_'+str(jnt_idx-1)
                        parent_body = self.env.spec.body(parent_name)
                        prev_row = np.asarray(self.sorted_data[data_part][jnt_idx-1])
                        from_pos = prev_row[3:6] if prev_row.shape[-1] >= 6 else np.array([0.0,0.0,0.0])
                    if parent_body is None:
                        raise ValueError(f"Parent body '{parent_name}' not found in spec. current: {body_name}")

                    if repr_kind=="screw":
                        # ===== screw (6D): original path unchanged =====
                        p_joint = np.asarray(group_data[jnt_idx][3:6])  # world
                        axis    = np.asarray(group_data[jnt_idx][0:3])  # world
                        body = parent_body.add_body(
                            name=body_name,
                            pos = p_joint - from_pos,
                        )
                        if np.linalg.norm(axis) >= self._AXIS_EPS:  # valid axis
                            body.add_geom(
                                name=body_name+"_geom",
                                type=mj.mjtGeom.mjGEOM_CYLINDER,
                                size=[0.075, 0.075, 0.075],
                                rgba=joint_color,
                                quat=self.axis_to_quat(axis),
                            )
                            body.add_joint(
                                name=body_name+'_joint',
                                type=mj.mjtJoint.mjJNT_HINGE,
                                axis=axis,
                            )
                        fr2to = p_joint - from_pos
                        if np.linalg.norm(fr2to) >= self._AXIS_EPS:
                            parent_body.add_geom(
                                name=parent_name+"_link_to_"+body_name,
                                type=mj.mjtGeom.mjGEOM_CAPSULE,
                                size=[0.04, 0.04, 0.04],
                                rgba=joint_color,
                                fromto=np.concatenate([np.array([0.0,0.0,0.0]), fr2to]),
                            )

                    elif repr_kind=="dh":
                        row = np.asarray(group_data[jnt_idx], float)
                        theta, d, a, alpha = map(float, row[:4].tolist())
                        axis_offset = float(row[4]) if row.size >= 5 else 0.0

                        # rotation matrix (parent -> child)
                        c, s = math.cos(theta), math.sin(theta)
                        Rz = np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]], dtype=float)
                        sa, ca = math.sin(alpha), math.cos(alpha)
                        Rx = np.array([[1.0,0.0,0.0],[0.0,ca,-sa],[0.0,sa,ca]], dtype=float)

                        Rchild = Rz.dot(Rx)                          # the child rotation is defined only by this
                        child_quat = r2quat(Rchild)          # take the quaternion from here as well

                        # position (Q_i relative to the parent)
                        child_pos  = Rz.dot(np.array([a, 0.0, d], dtype=float))
                        z_child_in_parent = Rchild[:, 2]
                        child_tip_parent  = child_pos + axis_offset * z_child_in_parent

                        # ---- child body (joint frame): keep the kinematics ----
                        body = parent_body.add_body(name=body_name, pos=child_tip_parent, quat=child_quat)

                        # ---- joint & geom: at the child origin (Q_i), along the child z axis ----
                        is_valid_row = not np.allclose(row[:4], 0.0, atol=1e-12)
                        if is_valid_row and not part.endswith('_tcp'):
                            axis_local      = np.array([0.0, 0.0, 1.0], dtype=float)
                            joint_pos_local = np.array([0.0, 0.0, 0.0], dtype=float)
                            # draw joints as SPHEREs
                            body.add_geom(
                                name=body_name+"_geom",
                                type=mj.mjtGeom.mjGEOM_SPHERE,
                                size=[0.07, 0.07, 0.07],
                                rgba=joint_color,
                                pos = joint_pos_local,
                            )
                            body.add_joint(
                                name=body_name+'_joint',
                                type=mj.mjtJoint.mjJNT_HINGE,
                                axis=axis_local,
                                pos = joint_pos_local,
                            )

                        # ---- link as a CAPSULE: parent tip -> child tip ----
                        if np.linalg.norm(child_tip_parent) >= self._AXIS_EPS:
                            link_len = np.linalg.norm(child_tip_parent)
                            link_radius = float(np.clip(link_len * 0.08, 0.012, 0.035))
                            parent_body.add_geom(
                                name=parent_name+"_link_to_"+body_name,
                                type=mj.mjtGeom.mjGEOM_CAPSULE,
                                size=[link_radius, link_radius, link_radius],
                                rgba=joint_color,
                                fromto=np.concatenate([[0.0,0.0,0.0],child_tip_parent]),
                            )
                    else:
                        raise ValueError(f"Unsupported feature dim: {feat_dim}")

                    self.env.spec.compile()

        self.env.model = self.env.spec.compile()
        self.env.data = mj.MjData(self.env.model)
        self.env.forward(increase_tick=False)
        self.env._parse_xml(update=True)

    def build_spec_wo_cylinder(self, robot_data,repr_kind="screw"):
        """Same as build_spec() but without the capsule link geoms (joints only)."""
        body = self.env.spec.worldbody.add_body(
                                name="base_link",
                                pos = [0.0,0.0,0.0]
                                    )
        self.env.spec.compile()
        self.env.forward(increase_tick=False)
        if self.sorted_data is None:
            raise ValueError("sorted data is None")
        for group_idx, part in enumerate(self.sorted_data): # generating link & joints
            group_data = np.array(self.sorted_data[part])
            n_joint = group_data.shape[0]
            if n_joint!=0:
                for jnt_idx in range(n_joint):
                    # # get joint positions and rotation axes
                    joint_color = [0,0,0,1]
                    body_name=part+'_'+str(jnt_idx)
                    # jnt_idx==0: should be connected to parent in other part
                    if jnt_idx == 0:
                        # if base_link: parent==ground
                        # else: connecting point is in parent body
                        if self.ref_data[part]=="base_link":
                            parent_name = "base_link"
                            from_pos=np.array([0.0,0.0,0.0])
                        else:
                            data_part = self.ref_data[part] # finding parent
                            data_ref_idx = self.ref_index[part]
                            parent_name = data_part+'_'+str(self.ref_index[part])
                            # the parent's q exists only for screw (6D). DH is drawn locally, so leave it as 0.
                            parent_row = np.asarray(self.sorted_data[data_part][data_ref_idx])
                            from_pos = parent_row[3:6] if repr_kind=="screw" else np.array([0.0,0.0,0.0])
                        parent_body = self.env.spec.body(parent_name)
                    # jnt_idx!=0: should be connected to parent in same part   
                    else:
                        data_part = part
                        parent_name = data_part+'_'+str(jnt_idx-1)
                        parent_body = self.env.spec.body(parent_name)
                        prev_row = np.asarray(self.sorted_data[data_part][jnt_idx-1])
                        from_pos = prev_row[3:6] if prev_row.shape[-1] >= 6 else np.array([0.0,0.0,0.0])
                    if parent_body is None:
                        raise ValueError(f"Parent body '{parent_name}' not found in spec. current: {body_name}")

                    if repr_kind=="screw":
                        # ===== screw (6D): original path unchanged =====
                        p_joint = np.asarray(group_data[jnt_idx][3:6])  # world
                        axis    = np.asarray(group_data[jnt_idx][0:3])  # world
                        body = parent_body.add_body(
                            name=body_name,
                            pos = p_joint - from_pos,
                        )
                        if np.linalg.norm(axis) >= self._AXIS_EPS:  # valid axis
                            body.add_geom(
                                name=body_name+"_geom",
                                type=mj.mjtGeom.mjGEOM_CYLINDER,
                                size=[0.05, 0.05, 0.05],
                                rgba=[1,1,1,0],
                                quat = self.axis_to_quat(axis)  # z -> axis rotation
                            )
                            body.add_joint(
                                name=body_name+'_joint',
                                type=mj.mjtJoint.mjJNT_HINGE,
                                axis=axis,
                            )
                        fr2to = p_joint - from_pos
                        if np.linalg.norm(fr2to) >= self._AXIS_EPS:
                            parent_body.add_geom(
                                name=parent_name+"_link_to_"+body_name,
                                type=mj.mjtGeom.mjGEOM_CYLINDER,
                                size=[0.01, 0.01, 0.01],
                                rgba=joint_color,
                                fromto=np.concatenate([np.array([0.0,0.0,0.0]), fr2to]),
                            )
                    
                    elif repr_kind=="dh":
                        row = np.asarray(group_data[jnt_idx], float)
                        theta, d, a, alpha = map(float, row[:4].tolist())
                        axis_offset = float(row[4]) if row.size >= 5 else 0.0

                        # rotation matrix (parent -> child)
                        c, s = math.cos(theta), math.sin(theta)
                        Rz = np.array([[c,-s,0.0],[s,c,0.0],[0.0,0.0,1.0]], dtype=float)
                        sa, ca = math.sin(alpha), math.cos(alpha)
                        Rx = np.array([[1.0,0.0,0.0],[0.0,ca,-sa],[0.0,sa,ca]], dtype=float)

                        Rchild = Rz.dot(Rx)                          # the child rotation is defined only by this
                        child_quat = r2quat(Rchild)          # take the quaternion from here as well

                        # position (Q_i relative to the parent)
                        child_pos  = Rz.dot(np.array([a, 0.0, d], dtype=float))
                        z_child_in_parent = Rchild[:, 2]
                        child_tip_parent  = child_pos + axis_offset * z_child_in_parent
                        
                        # ---- child body (joint frame): keep the kinematics ----
                        body = parent_body.add_body(name=body_name, pos=child_tip_parent, quat=child_quat)

                        # ---- joint & geom: at the child origin (Q_i), along the child z axis ----
                        is_valid_row = not np.allclose(row[:4], 0.0, atol=1e-12)
                        if is_valid_row and not part.endswith('_tcp'):
                            axis_local      = np.array([0.0, 0.0, 1.0], dtype=float)
                            joint_pos_local = np.array([0.0, 0.0, 0.0], dtype=float)
                            body.add_geom(
                                name=body_name+"_geom",
                                type=mj.mjtGeom.mjGEOM_CYLINDER,
                                size=[0.05, 0.05, 0.05],
                                rgba=[1,1,1,0],
                                pos = joint_pos_local,   # to draw at the 'tip', change only this to pos=[0,0,axis_offset]
                            )
                            body.add_joint(
                                name=body_name+'_joint',
                                type=mj.mjtJoint.mjJNT_HINGE,
                                axis=axis_local,
                                pos = joint_pos_local,
                            )

                        # ---- link as a 'single line': parent tip -> child tip ----
                        if np.linalg.norm(child_tip_parent) >= self._AXIS_EPS:
                            parent_body.add_geom(
                                name=parent_name+"_link_to_"+body_name,
                                type=mj.mjtGeom.mjGEOM_CYLINDER,
                                size=[0.01, 0.01, 0.01],
                                rgba=joint_color,
                                fromto=np.concatenate([[0.0,0.0,0.0],child_tip_parent]),
                            )
                    else:
                        raise ValueError(f"Unsupported feature dim: {feat_dim}")

                    self.env.spec.compile()

        self.env.model = self.env.spec.compile()
        self.env.data = mj.MjData(self.env.model)
        self.env.forward(increase_tick=False)
        self.env._parse_xml(update=True)
    
    def _is_valid_row(self, row: np.ndarray) -> bool:
        """Validity per representation: screw (||w|| >= eps), dh (invalid if the whole row is 0)."""
        d = row.shape[-1]
        if d == 6:  # screw
            return float(np.linalg.norm(row[:3])) >= self._AXIS_EPS
        elif d in (4,5):  # dh
            return not np.allclose(row, 0.0, atol=1e-12)
        else:
            return False
    
    def is_valid(self, joint) -> bool:
        """
        Validity per representation:
          - screw(6D): ‖w‖>=_AXIS_EPS
          - SDH (4D)  : invalid if the whole row is 0
        """
        row = np.asarray(joint)
        return self._is_valid_row(row)

    def get_group(self, group, group_type: str = 'valid'):
        """
        Split valid/invalid rows within a group. (uses is_valid per representation)
        - group: (N, 6) [screw] or (N, 4) [dh] numpy array / list
        """
        if group is None:
            return []
        rows = list(group)  # iterable whether numpy or list
        if group_type == 'valid':
            return [r for r in rows if self.is_valid(r)]
        elif group_type == 'invalid':
            return [r for r in rows if not self.is_valid(r)]
        else:
            raise ValueError(f"unknown group_type: {group_type}")

    def axis_to_quat(self,axis):
        """Return the quaternion that rotates the z axis onto the given axis."""
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        # quaternion that rotates the z-axis onto axis
        quat = np.zeros(4)
        mj.mju_quatZ2Vec(quat, axis)
        return quat
    
    def _feature_dim_of_group(self, group_rows: np.ndarray) -> int:
        """Return the feature dimension of a group array (6: screw, 4: dh)."""
        if group_rows is None or len(group_rows) == 0:
            return 0
        return int(group_rows.shape[-1])

    

    def _quat_rx(self, alpha: float) -> np.ndarray:
        """Quaternion [w,x,y,z] of the rotation Rx(alpha)."""
        half = 0.5 * float(alpha)
        return np.array([math.cos(half), math.sin(half), 0.0, 0.0], dtype=float)

    def visualize_data_to_classify(self,data_to_classify,
                                   save_video=True, 
                                   vis=True, 
                                   test=False,
                                   xml=True,
                                   repr_kind = "screw",
                                   verbose=False
                                   ):
        """Build a MuJoCo model for every robot in data_to_classify.

        Args:
            data_to_classify: {name: {group: (N, 6) or (N, 5) array}}, e.g. the output of
                convert_flat_interp_to_dict().
            save_video: also render a short clip of each robot.
            vis: open an interactive viewer.
            xml: store the generated MJCF string of each robot in self.xml[name].
            repr_kind: 'screw' or 'dh'.

        Returns:
            a moviepy clip when save_video is True, otherwise None. The MJCF strings are
            available in self.xml.
        """
        frames=[]
        self.xml={}
        # Loop
        self._initialize_env() # load mujoco_parser


        for name, robot_data in data_to_classify.items(): # load robot_name, robot_data
            if vis:
                self.env.init_viewer(
                    title             = name,
                    transparent       = False,
                    black_sky         = True,
                    azimuth           = 180, # None,
                    distance          = 3.0,
                    elevation         = 0, # None,
                    lookat            = [0.0,0.0,0.3], # None,
                    use_rgb_overlay   = False,
                    loc_rgb_overlay   = 'top right',
                    fontscale         = mj.mjtFontScale.mjFONTSCALE_50.value,
                    )
            
            self.reconstruct_kinematic_chain(name,robot_data,repr_kind)

            self.env.forward(increase_tick=False)
            # add base link
            self.build_spec(robot_data,repr_kind)
            if vis:
                self.env.plot_contact_info()
                self.env.viewer.model=self.env.model
                self.env.viewer.data = self.env.data
            if xml:
                self.xml[name]=self.env.spec.to_xml()

            if vis:
                if verbose:
                    self.env.plot_T()
                self.env.forward()
                self.env.render()
                if save_video:
                    image=self.env.grab_image()
                    frames.append(image)
                if not test:
                    self.env.close_viewer()
                else:
                    while self.env.is_viewer_alive():
                        if verbose:
                            self.env.plot_bodies_T(plot_name=True)
                            self.env.plot_joints(plot_joint_names=True)
                            self.env.plot_T()
                        self.env.forward()
                        self.env.render()


            body = self.env.spec.worldbody.first_body() # initialize mjSpec
            delete_list = []
            while body: # to ensure initialization
                delete_list.append(body)
                body = self.env.spec.worldbody.next_body(body)
            for body in delete_list:
                self.env.spec.detach_body(body)
        
        if frames:
            fps = 5
            timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
            video_filename = f"../../result/simulation_{timestamp}.mp4"

            # 2) create the clip (frames is a list of RGB numpy arrays)
            clip = ImageSequenceClip(frames, fps=fps)

            # 3) save as mp4 (libx264 codec)
            clip.write_videofile(video_filename, codec="libx264", audio=False)

            # 4) embed in the notebook
            return clip

    def visualize_skeleton(self,data_to_classify,
                                   save_video=True, 
                                   vis=True, 
                                   test=False,
                                   xml=True,
                                   repr_kind = "screw",
                                   ):
        """Like visualize_data_to_classify(), but draws only the skeleton (lines between joints)."""
        frames=[]
        self.xml={}
        # Loop
        self._initialize_env() # load mujoco_parser


        for name, robot_data in data_to_classify.items(): # load robot_name, robot_data
            if vis:
                self.env.init_viewer(
                    title             = name,
                    transparent       = True,
                    black_sky         = False,
                    azimuth           = 180, # None,
                    distance          = 3.0,
                    elevation         = 0, # None,
                    lookat            = [0.0,0.0,0.3], # None,
                    use_rgb_overlay   = False,
                    loc_rgb_overlay   = 'top right',
                    fontscale         = mj.mjtFontScale.mjFONTSCALE_50.value,
                    )
            
            self.reconstruct_kinematic_chain(name,robot_data,repr_kind)

            self.env.forward(increase_tick=False)
            # add base link
            self.build_spec_wo_cylinder(robot_data,repr_kind)
            if vis:
                self.env.plot_contact_info()
                self.env.viewer.model=self.env.model
                self.env.viewer.data = self.env.data
            if xml:
                self.xml[name]=self.env.spec.to_xml()

            if vis:
                self.env.forward()
                self.env.render()
                if save_video:
                    image=self.env.grab_image()
                    frames.append(image)
                if not test:
                    self.env.close_viewer()
                else:
                    while self.env.is_viewer_alive():
                        self.env.forward()
                        self.env.render()


            body = self.env.spec.worldbody.first_body() # initialize mjSpec
            delete_list = []
            while body: # to ensure initialization
                delete_list.append(body)
                body = self.env.spec.worldbody.next_body(body)
            for body in delete_list:
                self.env.spec.detach_body(body)
        
        if frames:
            fps = 5
            timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
            video_filename = f"../../result/simulation_{timestamp}.mp4"

            # 2) create the clip (frames is a list of RGB numpy arrays)
            clip = ImageSequenceClip(frames, fps=fps)

            # 3) save as mp4 (libx264 codec)
            clip.write_videofile(video_filename, codec="libx264", audio=False)

            # 4) embed in the notebook
            return clip

    def get_slider_range(self,data_np, model, ratio=0.1, device="cpu"):
        """
        Encodes data using the model and returns min/max values for each latent axis
        with a buffer defined by `ratio`.

        :param data_np: (N, D) numpy array of input data
        :param model: Trained AutoEncoderClass with .encode()
        :param ratio: float, extra padding ratio on both sides
        :param device: computation device
        :return: (2, z_dim) numpy array with [max values, min values]
        """
        model.eval()
        data_tensor = torch.tensor(data_np, dtype=torch.float32).to(device)

        with torch.no_grad():
            z = model.encode(data_tensor).cpu().numpy()  # shape: (N, z_dim)

        z_min = np.min(z, axis=0)
        z_max = np.max(z, axis=0)
        buffer = (z_max - z_min) * ratio

        # compute the extended range
        z_min_with_buf = z_min - buffer
        z_max_with_buf = z_max + buffer

        # flip so that max is on top for readability
        return np.vstack([z_max_with_buf, z_min_with_buf])


    # ---- global threshold estimation ----
    def thresholds_from_quantiles(self,dataset, p_low=30, p_high=60, floors=(0.0, 0.0), alpha_col=0, length_col=-1):
        """Estimate (low, high) thresholds for the DH alpha and length columns from percentiles over a dataset."""
        alphas, lengths = [], []
        for rd in dataset:
            for arr in rd.values():
                alphas.append(np.abs(arr[:, alpha_col]).ravel())
                lengths.append(np.abs(arr[:, length_col]).ravel())
        alphas = np.concatenate(alphas) if alphas else np.array([])
        lengths = np.concatenate(lengths) if lengths else np.array([])

        t_low_a  = np.percentile(alphas,  p_low) if alphas.size  else 0.0
        t_high_a = np.percentile(alphas,  p_high) if alphas.size else 0.0
        t_low_l  = np.percentile(lengths, p_low) if lengths.size else 0.0
        t_high_l = np.percentile(lengths, p_high) if lengths.size else 0.0

        fa, fl = floors
        return {"alpha": (max(t_low_a,  fa), max(t_high_a, fa)),
                "length":(max(t_low_l,  fl), max(t_high_l, fl))}

    # (alternative) MAD-based threshold
    def thresholds_from_mad(self,dataset, k_low=0.25, k_high=0.75, floors=(0.0, 0.0), alpha_col=0, length_col=-1):
        """Estimate (low, high) thresholds for the DH alpha and length columns from the median absolute deviation."""
        alphas, lengths = [], []
        for rd in dataset:
            for arr in rd.values():
                alphas.append(np.abs(arr[:, alpha_col]).ravel())
                lengths.append(np.abs(arr[:, length_col]).ravel())
        alphas = np.concatenate(alphas) if alphas else np.array([])
        lengths = np.concatenate(lengths) if lengths else np.array([])

        def mad_thr(x, k):
            if x.size == 0: return 0.0
            med = np.median(x)
            mad = np.median(np.abs(x - med))
            sigma = 1.4826 * mad
            return med + k * sigma

        fa, fl = floors
        return {"alpha": (max(mad_thr(alphas,  k_low),  fa), max(mad_thr(alphas,  k_high), fa)),
                "length":(max(mad_thr(lengths, k_low),  fl), max(mad_thr(lengths, k_high), fl))}

    # ---- persistence estimation (weak/strong frequency over many samples) ----
    def build_persistence(self,dataset, thresholds, persist_on="weak", alpha_col=0, length_col=-1):
        """Count how often each joint slot is weak/strong over a dataset, for prune_no_budget()."""
        t_low_a,  t_high_a  = thresholds['alpha']
        t_low_l,  t_high_l  = thresholds['length']
        parts = {p for rd in dataset for p in rd}
        maxJ  = {p: max(rd[p].shape[0] for rd in dataset if p in rd) for p in parts}

        stats = {p: {"count": np.zeros(maxJ[p], dtype=int),
                    "count_strong": np.zeros(maxJ[p], dtype=int),
                    "N": 0} for p in parts}

        for rd in dataset:
            for p in parts:
                arr = rd.get(p)
                if arr is None: continue
                a = np.abs(arr[:, alpha_col]); l = np.abs(arr[:, length_col])
                weak   = (a >= t_low_a)  | (l >= t_low_l)
                strong = (a >= t_high_a) | (l >= t_high_l)
                stats[p]["count"][:arr.shape[0]]        += weak.astype(int)
                stats[p]["count_strong"][:arr.shape[0]] += strong.astype(int)
            for p in parts:
                stats[p]["N"] += 1

        for p in parts:
            N = max(stats[p]["N"], 1)
            stats[p]["freq"]         = stats[p]["count"]        / N
            stats[p]["freq_strong"]  = stats[p]["count_strong"] / N
        return stats

   

    def prune_no_budget(
        self, robot_data, thresholds, persistence,
        persist_min=0.4, 
        link_across_parts=True, 
        anchors=True, 
        symmetry="intersection", # union | intersection
        mode="drop", alpha_col=0, length_col=-1
    ):
        """Prune weak DH joints of one robot with hysteresis, connectivity and left/right symmetry.

        Returns:
            (pruned robot_data, keep mask per group).
        """
        t_low_a,  t_high_a  = thresholds['alpha']
        t_low_l,  t_high_l  = thresholds['length']
         # ---- hysteresis + connectivity + (optional) left/right sync ----
        CHAIN_SIDES = {
            "left":  ["left_shoulder_girdle","left_shoulder","left_upperarm","left_elbow","left_forearm","left_wrist","left_tcp"],
            "right": ["right_shoulder_girdle","right_shoulder","right_upperarm","right_elbow","right_forearm","right_wrist","right_tcp"],
        }
        keep, strong = {}, {}
        # 1) strong/weak + persistence + in-part proximity hysteresis
        for p, arr in robot_data.items():
            a = np.abs(arr[:, alpha_col]); l = np.abs(arr[:, length_col])
            strong[p] = (a >= t_high_a) | (l >= t_high_l)
            weak      = (a >= t_low_a)  | (l >= t_low_l)

            if p in persistence:
                freq = persistence[p]['freq'][:arr.shape[0]]
                weak = weak & (freq >= persist_min)

            keep[p] = strong[p] | weak

            if arr.shape[0] > 1:  # propagate to neighbors within the part
                neigh = np.zeros_like(keep[p], dtype=bool)
                idx = np.where(strong[p])[0]
                for i in idx:
                    if i-1 >= 0:                  neigh[i-1] = True
                    if i+1 < arr.shape[0]:        neigh[i+1] = True
                keep[p] |= (weak & neigh)

        # 2) propagate strong-joint connections at chain boundaries
        if link_across_parts:
            for side, parts in CHAIN_SIDES.items():
                for i, p in enumerate(parts):
                    arr = robot_data.get(p)
                    if arr is None: continue
                    J = arr.shape[0]
                    # previous part <-> current part
                    if i > 0:
                        prevp = parts[i-1]
                        if prevp in robot_data and keep[p].size:
                            if strong.get(prevp, np.array([],bool)).shape[0] and strong[prevp][-1]:
                                keep[p][0] = True
                            if strong.get(p, np.array([],bool)).shape[0] and strong[p][0] and keep.get(prevp, np.array([],bool)).shape[0]:
                                keep[prevp][-1] = True
                    # current part <-> next part
                    if i < len(parts)-1:
                        nextp = parts[i+1]
                        if nextp in robot_data and keep[p].size:
                            if strong[p].shape[0] and strong[p][-1] and robot_data[nextp].shape[0]:
                                keep[nextp][0] = True
                            if strong.get(nextp, np.array([],bool)).shape[0] and strong[nextp][0]:
                                keep[p][-1] = True
                # anchors (optional): both ends of the chain
                if anchors:
                    firstp, lastp = parts[0], parts[-1]
                    if firstp in keep and keep[firstp].size: keep[firstp][0]  = True
                    if lastp  in keep and keep[lastp].size:  keep[lastp][-1]  = True

        # 3) left/right sync (optional)
        if symmetry in ("union", "intersection"):
            for lp, rp in zip(CHAIN_SIDES["left"], CHAIN_SIDES["right"]):
                if lp in keep and rp in keep and keep[lp].shape == keep[rp].shape:
                    if symmetry == "union":
                        u = keep[lp] | keep[rp]
                        keep[lp], keep[rp] = u.copy(), u.copy()
                    else:
                        inter = keep[lp] & keep[rp]
                        keep[lp], keep[rp] = inter.copy(), inter.copy()

        # 4) apply (drop the row, or keep it with only alpha/length set to 0)
        out = {}
        for p, arr in robot_data.items():
            m = keep[p]
            if mode == "drop":
                out[p] = arr[m]
            else:  # mode == "zero"
                out[p] = arr.copy()
                nz = ~m
                out[p][nz, alpha_col]  = 0.0
                out[p][nz, length_col] = 0.0

        return out, keep

    def build_full_interp_from_symmetric(
        self,
        interp_data_sym: np.ndarray,
        threshold: float = 0.0,
        # feature dimension per group (treat all as screw=6 if missing)
        repr_kind="screw",
        # axis indices to flip when mirroring screws (y-mirror: (1,))
        screw_flip_axes: tuple = (1,),
    ):
        """
        interp_data_sym : (N, F_sym)
        - interpolation result with only the symmetric joint groups concatenated
        featdim_by_group : {group_name: feat_dim}, feat_dim ∈ {6(screw), 4(DH), 5(DH+tau)}
        - if not given, all groups are assumed to be 6
        """
        N = interp_data_sym.shape[0]
        if repr_kind=="screw":
            classifying_all_joints = self.classifying_all_joints
            classifying_symmetric_joints = self.classifying_symmetric_joints
        elif repr_kind=="dh":
            classifying_all_joints = self.classifying_all_joints_dh
            classifying_symmetric_joints = self.classifying_symmetric_joints_dh
        all_group_names = [j[0] for j in classifying_all_joints]
        sym_group_names = [j[0] for j in classifying_symmetric_joints]
        njnt_all        = [int(x) for x in self.max_joints_per_group]
        # --- 0) get the per-group dimension table ---
        def _dim_of():
            if repr_kind == "screw":
                return 6
            elif repr_kind == "dh":
                return 5

        featdim_all = [_dim_of() for g in all_group_names]
        featdim_sym = [_dim_of() for g in sym_group_names]
        # --- 1) offsets (per-group dimensions) ---
        def _calc_offsets(njnts, dims):
            offs = []
            start = 0
            for c, d in zip(njnts, dims):
                span = c * d
                offs.append((start, start + span))
                start += span
            return offs, start

        all_offsets, F_all = _calc_offsets(njnt_all, featdim_all)

        # get the sym indices (positions in all) and the corresponding njnt/featdim
        sym_indices = [all_group_names.index(name) for name in sym_group_names]
     
        njnt_sym    = [njnt_all[i] for i in sym_indices]
        # assume the symmetric block is concatenated in the order of the symmetric groups
        sym_offsets, F_sym = _calc_offsets(njnt_sym, featdim_sym)

        assert interp_data_sym.shape[1] == F_sym, \
            f"Symmetric data shape mismatch: got {interp_data_sym.shape[1]}, expected {F_sym}"

        # --- 2) result buffer ---
        full_data = np.zeros((N, F_all), dtype=float)

        # --- 3) helper: mirroring rule ---
        def _mirror_screw(chunk_reshaped: np.ndarray) -> np.ndarray:
            # chunk_reshaped: (N, J, 6) = [axis(3), pos(3)]
            out = chunk_reshaped.copy()
            # for axes that are almost 0 under the threshold, zero only the axis (keep the position to preserve LEFT/RIGHT symmetry).
            # this used to zero the whole row, but then the virtual joint / anchor position on the LEFT side
            # that got destroyed, and when the wrist hack fired, a phantom body was pinned at the world origin, creating a long capsule (bug).
            # convert_flat_interp_to_dict only zeroes the axis anyway, so this restores consistency between the two.
            axis = out[:, :, 0:3]
            mask_zero = np.linalg.norm(axis, axis=2) < threshold
            out[mask_zero, 0:3] = 0.0
            # y-mirror: flip the sign of the given axes
            for idx in screw_flip_axes:
                out[:, :, idx]   *= -1.0       # axis.y, etc.
                out[:, :, 3+idx] *= -1.0       # pos.y, etc.
            return out

        def _mirror_dh(chunk_reshaped: np.ndarray) -> np.ndarray:
            # (N, J, 4 or 5): [theta, d, a, alpha, (tau)]
            out = chunk_reshaped.copy()
            if out.shape[2] < 4:
                return out
            out[:, :, 0] *= -1.0   # theta
            out[:, :, 3] *= -1.0   # alpha
            # d(1), a(2), tau(4) are kept as is
            return out

        # --- 4) main loop: fill sym -> full, restore the other side by mirroring ---
        sym_cursor = 0
        for sym_block_idx, group_name in enumerate(sym_group_names):
            all_idx   = sym_indices[sym_block_idx]
            a_start, a_end = all_offsets[all_idx]     # position in full
            s_start, s_end = sym_offsets[sym_block_idx]  # position in sym
            J = njnt_all[all_idx]
            D = featdim_all[all_idx]
            chunk = interp_data_sym[:, s_start:s_end]
            chunk_reshaped = chunk.reshape(N, J, D)

            # 4-1) first fill the block (e.g. right_arm)
            full_data[:, a_start:a_end] = chunk

            # 4-2) fill the opposite side (right_* <-> left_*)
            if group_name.startswith("right_"):
                left_name = group_name.replace("right_", "left_")
            elif group_name.startswith("left_"):
                left_name = group_name.replace("left_", "right_")
            else:
                left_name = None

            if left_name and left_name in all_group_names:
                left_all_idx = all_group_names.index(left_name)
                # the other side must have the same dimension / number of joints
                if (njnt_all[left_all_idx] == J) and (featdim_all[left_all_idx] == D):
                    l_start, l_end = all_offsets[left_all_idx]

                    if D == 6:
                        mirrored = _mirror_screw(chunk_reshaped)
                    elif D in (4, 5):
                        mirrored = _mirror_dh(chunk_reshaped)
                    else:
                        raise ValueError(f"Unsupported feature dim {D} for group {group_name}")

                    full_data[:, l_start:l_end] = mirrored.reshape(N, -1)
        return full_data

    def convert_flat_interp_to_dict(self,flat_interp_data, threshold=0.5,repr_kind="screw"):
        """
        flat_interp_data: shape (N, F_all)
        returns: dict[str index] = { group_name: (njoint, 6) np.array }
        """
        n_samples = flat_interp_data.shape[0]
        if repr_kind=="screw":
            classifying_all_joints = self.classifying_all_joints
            param_num = 6
        elif repr_kind=="dh":
            classifying_all_joints = self.classifying_all_joints_dh
            param_num = 5
        all_names = [j[0] for j in classifying_all_joints]
        njnt_all = [int(x) for x in self.max_joints_per_group]
        # compute offsets
        group_offsets = []
        start = 0
        for count in njnt_all:
            group_offsets.append((start, start + count * param_num))
            start += count * param_num

        result_dict = {}

        for i in range(n_samples):
            row = flat_interp_data[i]
            group_dict = {}

            for group_name, (start, end), nj in zip(all_names, group_offsets, njnt_all):
                chunk = row[start:end].reshape(nj, param_num)
                # apply the threshold
                for idx in range(nj):
                    if repr_kind =="screw":
                        axis = chunk[idx, 0:3]
                        if np.linalg.norm(axis) < threshold:
                            chunk[idx, 0:3] = np.array([0.0, 0.0, 0.0])
                    elif repr_kind=="dh":
                        if chunk[idx][1]<=0.01 and chunk[idx][2]<=0.01:
                            chunk[idx]=np.array([0.0,0.0,0.0,0.0,0.0])
                group_dict[group_name] = chunk

            result_dict[str(i)] = group_dict
        return result_dict

    def explore_latent_dimension(self,data_np,model,ratio=0.1,repr_kind="screw"):
        """Open an interactive slider window to move through the latent space of model and see the decoded robot live."""
        self._initialize_env()
        slider_range = self.get_slider_range(data_np, model, ratio=ratio)
        if is_macos():
            sliders = MultiSliderQtClass(
                n_slider      = np.shape(slider_range)[1],
                title         = 'Sliders for [%s] Control'%(self.env.name),
                window_width  = 600,
                window_height = 800,
                x_offset      = 50,
                y_offset      = 100,
                slider_width  = 350,
                label_texts   = [str(f"axis{i+1}") for i in range(np.shape(slider_range)[1])],
                slider_mins   = slider_range[1],
                slider_maxs   = slider_range[0],
                slider_vals   = np.zeros(np.shape(slider_range)[1]),
                resolutions    = np.full(np.shape(slider_range)[1],0.001),
                verbose       = False,
            )
        elif is_ubuntu():
            sliders = MultiSliderClass(
                n_slider      = np.shape(slider_range)[1],
                title         = 'Sliders for [%s] Control'%(self.env.name),
                window_width  = 600,
                window_height = 800,
                x_offset      = 50,
                y_offset      = 100,
                slider_width  = 350,
                label_texts   = [str(f"axis{i+1}") for i in range(np.shape(slider_range)[1])],
                slider_mins   = slider_range[1],
                slider_maxs   = slider_range[0],
                slider_vals   = np.zeros(np.shape(slider_range)[1]),
                resolutions    = np.full(np.shape(slider_range)[1],0.001),
                verbose       = False,
            )
        
        self.env.init_viewer(
            transparent       = True,
            black_sky         = False,
            azimuth           = 180, # None,
            distance          = 4.0,
            elevation         = 0, # None,
            lookat            = [0.0,0.0,0.3], # None,
            use_rgb_overlay   = False,
            loc_rgb_overlay   = 'top right',
            fontscale         = mj.mjtFontScale.mjFONTSCALE_50.value,
            )
        past_slider_val = np.zeros(np.shape(slider_range)[1])
        while self.env.is_viewer_alive():
            sliders.update()
            self.env.render()
            slider_val = sliders.get_slider_values()
            if (slider_val != past_slider_val).any():
                half_data= model.decode(torch.from_numpy(slider_val[None,:]).float()).detach().numpy()
                full_interp= self.build_full_interp_from_symmetric(interp_data_sym=half_data,threshold=0.0,repr_kind=repr_kind)
                robot_data= self.convert_flat_interp_to_dict(full_interp,threshold=0.5,repr_kind=repr_kind)['0']
                body = self.env.spec.worldbody.first_body() # initialize mjSpec
                delete_list = []
                while body: # to ensure initialization
                    delete_list.append(body)
                    body = self.env.spec.worldbody.next_body(body)
                for body in delete_list:
                    self.env.spec.detach_body(body)
                self.reconstruct_kinematic_chain(name='trial',robot_data=robot_data,repr_kind=repr_kind)
                self.env.forward(increase_tick=False)
                self.build_spec(robot_data,repr_kind=repr_kind)
                self.env.plot_contact_info()
                
                self.env.viewer.model=self.env.model
                self.env.viewer.data = self.env.data
                print(f"njnt: {self.env.model.njnt}")
                past_slider_val = deepcopy(slider_val)
                slider_val=None

            self.env.forward()
        if not self.env.is_viewer_alive():
            sliders.close()


    def connect_all_joints_full(self, robot_data, spec=False):

        """Draw lines between consecutive valid joints of robot_data in the viewer (debug visualization)."""
        def draw_line(p1, p2):
            self.env.plot_line_fr2to(
                p_fr=p1.astype(np.float64),
                p_to=p2.astype(np.float64),
                rgba=[1, 1, 1, 1]
            )

        all_joint_names = [j[0] for j in self.classifying_all_joints]
        sorted_data = deepcopy(robot_data)

        # === 1. sort valid joints within each group ===
        for i, part in enumerate(all_joint_names):
            group = robot_data.get(part, [])
            valid_group = self.get_group(group,'valid')

            if len(valid_group) == 0:
                sorted_data[part] = []
                continue

            # if sorting relative to the neck is needed -> apply only to the shoulder
            if "shoulder" in part:
                neck_group = self.get_group(robot_data.get("neck", []),'valid')
                if len(neck_group) > 0:
                    ref = neck_group[-1][3:6]
                    valid_group = sorted(valid_group, key=lambda j: np.linalg.norm(j[3:6] - ref))

            sorted_data[part] = np.array(valid_group)

        # === 2. connect girdle -> neck or shoulder -> neck ===
        neck_group = sorted_data.get("neck", [])
        if len(neck_group) > 0:
            neck_base = neck_group[0][3:6]

            for side in ["left", "right"]:
                girdle_name = f"{side}_shoulder_girdle"
                shoulder_name = f"{side}_shoulder"

                girdle_group = sorted_data.get(girdle_name, [])
                shoulder_group = sorted_data.get(shoulder_name, [])

                if len(girdle_group) > 0:
                    draw_line(girdle_group[0][3:6], neck_base)
                elif len(shoulder_group) > 0:
                    draw_line(shoulder_group[0][3:6], neck_base)

        # === 3. connect valid joints in order within each group ===
        for i, part in enumerate(all_joint_names):
            group = sorted_data.get(part, [])
            if len(group) < 1:
                continue

            for j in range(len(group) - 1):
                draw_line(group[j][3:6], group[j + 1][3:6])

            # torso → origin
            if part == "torso":
                draw_line(np.array([0.0, 0.0, 0.0]), group[0][3:6])
            if part == "neck":
                continue
            # connect between groups: find the next valid group and connect (no left -> right)
            for j in range(i + 1, len(all_joint_names)):
                next_part = all_joint_names[j]

                # prevent connections that jump from left to right
                if part.startswith("left") and next_part.startswith("right"):
                    continue

                next_group = sorted_data.get(next_part, [])
                if len(next_group) > 0:
                    draw_line(group[-1][3:6], next_group[0][3:6])
                    break

    def pca_data_to_classify(self,n_components=2):
        """Run PCA on the flattened representation of all robots and store it in self.pca_result."""
        total_data = []
        for robot_name, joint_info in self.classifying_joint_info.items():
            data_to_classify = joint_info['data_to_classify']
            data_to_classify = np.array(data_to_classify).flatten()
            total_data.append(data_to_classify)
        total_data = np.array(total_data)
        print(f"total data to pca: {total_data.shape}")
        pca = PCA(n_components=n_components)
        pca.fit(total_data)
        result = pca.transform(total_data)
        self.pca_result = result

    def visualize_pca_result(self):
        """Visualize PCA results with robot names."""
        if self.pca_result is None:
            print("No PCA result. Run pca_data_to_classify() first.")
            return
        
        plt.figure(figsize=(12, 8))
        
        # draw scatter plot
        scatter = plt.scatter(self.pca_result[:, 0], self.pca_result[:, 1], c='b', alpha=0.6)
        
        # label each point with the robot name
        for idx, robot_name in enumerate(self.robot_list):
            plt.annotate(robot_name, 
                        (self.pca_result[idx, 0], self.pca_result[idx, 1]),
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.8)
        
        # style the plot
        plt.title('Robot Joint Configuration PCA Visualization')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.grid(True, alpha=0.3)
        
        # show origin
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def visualize_pca_result_3d(self):
        """Visualize PCA results in 3D with robot names."""
        if self.pca_result is None or self.pca_result.shape[1] < 3:
            print("No 3D PCA result. Run pca_data_to_classify(n_components=3) first.")
            return
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # draw 3D scatter plot
        scatter = ax.scatter(self.pca_result[:, 0], 
                            self.pca_result[:, 1], 
                            self.pca_result[:, 2], 
                            c='b', alpha=0.6)
        
        # label each point with the robot name
        for idx, robot_name in enumerate(self.robot_list):
            ax.text(self.pca_result[idx, 0], 
                    self.pca_result[idx, 1], 
                    self.pca_result[idx, 2],
                    robot_name,
                    fontsize=8)
        
        # style the plot
        ax.set_title('Robot Joint Configuration PCA Visualization (3D)')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
        
        # show grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

