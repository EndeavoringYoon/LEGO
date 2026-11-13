import numpy as np
from typing import Optional, Union, Tuple

def normalize(vec):
    """
    Compute the unit vector of the given vector.
    
    Parameters:
        vec (array): Input vector.
    
    Returns:
        np.array: Unit vector.
    """
    x = np.array(vec)
    len = np.linalg.norm(x)
    if len <= 1e-6:
        return np.array([0,0,1])
    else:
        return x/len    

def t2pr(T):
    """
    Convert a 4x4 transformation matrix to its position vector and rotation matrix.
    
    Parameters:
        T (np.array): 4x4 transformation matrix.
        
    Returns:
        p (np.array): 3-element position vector.
        R (np.array): 3x3 rotation matrix.
    """ 
    p = T[:3,3]
    R = T[:3,:3]
    return p,R

def t2p(T):
    """
    Extract the position vector from a 4x4 transformation matrix.
    
    Parameters:
        T (np.array): 4x4 transformation matrix.
        
    Returns:
        p (np.array): 3-element position vector.
    """
    p = T[:3,3]
    return p

def t2r(T):
    """
    Extract the rotation matrix from a 4x4 transformation matrix.
    
    Parameters:
        T (np.array): 4x4 transformation matrix.
        
    Returns:
        R (np.array): 3x3 rotation matrix.
    """
    R = T[:3,:3]
    return R

def rpy2r(rpy_rad):
    """
    Convert roll, pitch, and yaw angles (in radians) to a 3x3 rotation matrix.
    
    Parameters:
        rpy_rad (np.array): Array of [roll, pitch, yaw] in radians.
        
    Returns:
        R (np.array): 3x3 rotation matrix.
    """
    roll  = rpy_rad[0]
    pitch = rpy_rad[1]
    yaw   = rpy_rad[2]
    Cphi  = np.cos(roll)
    Sphi  = np.sin(roll)
    Cthe  = np.cos(pitch)
    Sthe  = np.sin(pitch)
    Cpsi  = np.cos(yaw)
    Spsi  = np.sin(yaw)
    R     = np.array([
        [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
        [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
        [-Sthe, Cthe * Sphi, Cthe * Cphi]
    ])
    assert R.shape == (3, 3)
    return R

def rpy2r_order(r0, order=[0,1,2]):
    """
    Convert roll, pitch, and yaw angles (in radians) to a 3x3 rotation matrix with a specified rotation order.
    
    Parameters:
        r0 (np.array): Array of [roll, pitch, yaw] in radians.
        order (list): List specifying the order of rotations (default is [0, 1, 2]).
        
    Returns:
        R (np.array): 3x3 rotation matrix.
    """
    c1 = np.cos(r0[0]); c2 = np.cos(r0[1]); c3 = np.cos(r0[2])
    s1 = np.sin(r0[0]); s2 = np.sin(r0[1]); s3 = np.sin(r0[2])
    a1 = np.array([[1,0,0],[0,c1,-s1],[0,s1,c1]])
    a2 = np.array([[c2,0,s2],[0,1,0],[-s2,0,c2]])
    a3 = np.array([[c3,-s3,0],[s3,c3,0],[0,0,1]])
    a_list = [a1,a2,a3]
    a = np.matmul(np.matmul(a_list[order[0]],a_list[order[1]]),a_list[order[2]])
    assert a.shape == (3,3)
    return a

def r2rpy(R,unit='rad'):
    """
    Convert a 3x3 rotation matrix to roll, pitch, and yaw angles.
    
    Parameters:
        R (np.array): 3x3 rotation matrix.
        unit (str): Unit for output angles ('rad' for radians or 'deg' for degrees).
        
    Returns:
        np.array: Array of [roll, pitch, yaw] angles.
    """
    roll  = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], (np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    if unit == 'rad':
        out = np.array([roll, pitch, yaw])
    elif unit == 'deg':
        out = np.array([roll, pitch, yaw])*180/np.pi
    else:
        out = None
        raise Exception("[r2rpy] Unknown unit:[%s]"%(unit))
    return out

def rs2rpys(Rs,unit='rad'):
    """
    Convert a list of 3x3 rotation matrices to roll, pitch, and yaw angles.

    Parameters:
        Rs (np.array): List of 3x3 rotation matrices.
        unit (str): Unit for output angles ('rad' for radians or 'deg' for degrees).

    Returns:
        np.array: Array of [roll, pitch, yaw] angles.
    """
    rpy_list = []
    for R in Rs:
        rpy_list.append(r2rpy(R,unit=unit))
    rpys = np.array(rpy_list) # [N x 3]
    return rpys

# r2quat is adapted from mat2quat in OpenAI Gym (gym/envs/robotics/rotations.py), which is
# based on transforms3d. Copyright (c) 2009-2017 Matthew Brett and Christoph Gohlke (BSD);
# Copyright (c) 2016 OpenAI (MIT). See THIRD_PARTY_NOTICES.md.
def r2quat(R):
    """
    Convert a 3x3 rotation matrix to a quaternion.
    
    Parameters:
        R (np.array): 3x3 rotation matrix.
        
    Returns:
        q (np.array): Quaternion in the form [w, x, y, z].
    """
    R = np.asarray(R, dtype=np.float64)
    Qxx, Qyx, Qzx = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    Qxy, Qyy, Qzy = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    Qxz, Qyz, Qzz = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q

def pr2t(p=(0,0,0),R=np.eye(3)):
    """
    Convert a position vector and rotation matrix into a 4x4 transformation matrix.
    
    Parameters:
        p (np.array): 3-element position vector.
        R (np.array): 3x3 rotation matrix.
        
    Returns:
        T (np.array): 4x4 transformation matrix.
    """
    p = np.asarray(p, dtype=np.float64)
    p0 = p.ravel() # flatten
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

def r2t(R=np.eye(3)):
    """
    Convert a rotation matrix into a 4x4 transformation matrix.
    
    Parameters:
        R (np.array): 3x3 rotation matrix.
        
    Returns:
        T (np.array): 4x4 transformation matrix.
    """
    p = np.asarray((0,0,0), dtype=np.float64)
    p0 = p.ravel() # flatten
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

def r2w(R):
    """
    Compute the angular velocity vector (axis-angle representation) from a 3x3 rotation matrix.
    
    Parameters:
        R (np.array): 3x3 rotation matrix.
        
    Returns:
        w (np.array): Angular velocity vector.
    """
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]], 
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()

def r2sixd(R):
    """
    Convert a 3x3 rotation matrix to a 6D representation.
    """
    return R[:, :2].flatten()

def sixd2r(d6):
    """
    Convert a 6D representation to a 3x3 rotation matrix.
    """
    d6 = d6.reshape(3, 2)
    a1, a2 = d6[:, 0], d6[:, 1]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])

def meters2xyz(depth_img,cam_matrix):
    """
    Convert a scaled depth image into a 3D point cloud.
    
    Parameters:
        depth_img (np.array): Depth image with depth values.
        cam_matrix (np.array): Camera intrinsic matrix.
        
    Returns:
        xyz_img (np.array): 3D point cloud of shape [H, W, 3].
    """
    fx = cam_matrix[0][0]
    cx = cam_matrix[0][2]
    fy = cam_matrix[1][1]
    cy = cam_matrix[1][2]
    
    height = depth_img.shape[0]
    width = depth_img.shape[1]
    indices = np.indices((height, width),dtype=np.float32).transpose(1,2,0)
    
    z_e = depth_img
    x_e = (indices[..., 1] - cx) * z_e / fx
    y_e = (indices[..., 0] - cy) * z_e / fy
    
    # Order of y_ e is reversed !
    xyz_img = np.stack([z_e, -x_e, -y_e], axis=-1) # [H x W x 3] 
    return xyz_img # [H x W x 3]

def get_rotation_matrix_from_two_points(p_fr,p_to):
    """
    Compute a rotation matrix that aligns a reference direction with the direction from one point to another.
    
    Parameters:
        p_fr (np.array): Starting point.
        p_to (np.array): Ending point.
        
    Returns:
        R (np.array): 3x3 rotation matrix.
    """
    p_a  = np.copy(np.array([1e-10,-1e-10,1.0]))
    if np.linalg.norm(p_to-p_fr) < 1e-8: # if two points are too close
        return np.eye(3)
    p_b  = (p_to-p_fr)/np.linalg.norm(p_to-p_fr)
    v    = np.cross(p_a,p_b)
    S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    if np.linalg.norm(v) == 0:
        R = np.eye(3,3)
    else:
        R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))
    return R

def skew(x):
    """
    Compute the skew-symmetric matrix of a 3-element vector.
    
    Parameters:
        x (np.array): 3-element vector.
        
    Returns:
        x_hat (np.array): 3x3 skew-symmetric matrix.
    """
    x_hat = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    return x_hat

def unskew(x_hat):
    """
    Compute the vector from a skew-symmetric matrix.
    
    Parameters:
        x_hat (np.array): 3x3 skew-symmetric matrix.
        
    Returns:
        x (np.array): 3-element vector.
    """
    x = np.array([x_hat[2,1], x_hat[0,2], x_hat[1,0]])
    return x

def screw_to_se3(w, q):
    """
    Convert screw axis (w, q) to a 4x4 Lie algebra se(3) matrix.
    Note that the output is an element of the Lie algebra se(3), not a full transformation,
    so this is used in exponential maps like expm(se3).

    Parameters:
        w (np.array): 3-element vector representing the screw axis.
        q (np.array): 3-element vector representing the position along the screw axis.

    Returns:
        se3 (np.array): 4x4 Lie algebra se(3) matrix.
    """
    w_hat = skew(w) # (3x3)
    v     = -np.cross(w, q) # (3,)
    se3   = np.zeros((4, 4)) # (4x4)
    se3[:3,:3] = w_hat
    se3[:3,3]  = v
    se3[3, 3] = 0.0  # stays 0, because this is se(3), not SE(3)
    return se3 # (4x4)

def rodrigues(a=np.array([1,0,0]),q_rad=0.0):
    """
    Compute the rotation matrix from an axis-angle representation using Rodrigues' formula.
    
    Parameters:
        a (np.array): 3-element unit vector representing the rotation axis.
        q_rad (float): Rotation angle in radians.
        
    Returns:
        R (np.array): 3x3 rotation matrix.
    """
    a_norm = np.linalg.norm(a)
    if abs(a_norm-1) > 1e-6:
        print ("[rodrigues] norm of a should be 1.0 not [%.2e]."%(a_norm))
        return np.eye(3)
    
    a = a / a_norm
    q_rad = q_rad * a_norm
    a_hat = skew(a)
    
    R = np.eye(3) + a_hat*np.sin(q_rad) + a_hat@a_hat*(1-np.cos(q_rad))
    return R


def rodrigues_batch(a=np.array([1, 0, 0]), q_rad=np.array([0.0])):
    """
    Compute rotation matrices for a batch of axis-angle representations using Rodrigues' formula.
    
    Parameters:
        a (np.array): Array of shape [B, 3] representing unit rotation axes.
        q_rad (np.array): Array of shape [B] representing rotation angles in radians.
        
    Returns:
        R (np.array): Array of shape [B, 3, 3] containing rotation matrices.
    """
    a = np.asarray(a)
    q_rad = np.asarray(q_rad)
    B = a.shape[0]

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(np.abs(a_norm - 1) > 1e-6):
        print("[rodrigues] Norm of each vector in a should be 1.0.")
        return np.tile(np.eye(3), (B, 1, 1))

    # Vectorized skew-symmetric matrix computation
    a_hat = np.zeros((B, 3, 3))  # shape [B, 3, 3]
    a_hat[:, 0, 1] = -a[:, 2]
    a_hat[:, 0, 2] = a[:, 1]
    a_hat[:, 1, 0] = a[:, 2]
    a_hat[:, 1, 2] = -a[:, 0]
    a_hat[:, 2, 0] = -a[:, 1]
    a_hat[:, 2, 1] = a[:, 0]

    sin_q = np.sin(q_rad).reshape(-1, 1, 1)  # shape [B, 1, 1]
    cos_q = np.cos(q_rad).reshape(-1, 1, 1)  # shape [B, 1, 1]

    R = (
        np.eye(3).reshape(1, 3, 3) +
        a_hat * sin_q +
        np.matmul(a_hat, a_hat) * (1 - cos_q)
    )  # shape [B, 3, 3]
    return R

def R_yuzf2zuxf(R):
    """
    Convert a rotation matrix from a Y-up Z-front coordinate system to a Z-up X-front coordinate system.
    
    Parameters:
        R (np.array): 3x3 rotation matrix in Y-up Z-front format.
        
    Returns:
        R_converted (np.array): 3x3 rotation matrix in Z-up X-front format.
    """
    R_offset = rpy2r(np.radians([-90,0,-90]))
    return R_offset@R

def T_yuzf2zuxf(T):
    """
    Convert a transformation matrix from a Y-up Z-front coordinate system to a Z-up X-front coordinate system.
    
    Parameters:
        T (np.array): 4x4 transformation matrix in Y-up Z-front format.
        
    Returns:
        T_converted (np.array): 4x4 transformation matrix in Z-up X-front format.
    """
    p,R = t2pr(T)
    T = pr2t(p=p,R=R_yuzf2zuxf(R))
    return T

def quat2r(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    
    Parameters:
        q (np.array): Quaternion in the form [w, x, y, z].
        
    Returns:
        R (np.array): 3x3 rotation matrix.
    """    
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
def align_z_axis(R):
    """
    Align the z-axis of a 3x3 rotation matrix with the global z-axis.
    
    Parameters:
        R (np.array): 3x3 rotation matrix.
        
    Returns:
        R_aligned (np.array): 3x3 rotation matrix with its z-axis aligned to [0, 0, 1].
    """
    q = r2quat(R)
    z_axis = R[:, 2]
    
    # Compute the rotation axis and angle
    rotation_axis = np.cross(z_axis, [0, 0, 1])
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    
    if rotation_axis_norm < 1e-15:  # z_axis is already [0,0,1] or [0,0,-1]
        if z_axis[2] < 0:  # [0,0,-1] case
            return R @ quat2r([0, 1, 0, 0])  # 180 degree rotation around x-axis
        else:
            return R
    
    rotation_axis /= rotation_axis_norm
    cos_theta = np.dot(z_axis, [0, 0, 1])
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    
    # Compute the rotation quaternion
    q_rot = np.array([np.cos(theta/2)] + list(np.sin(theta/2) * rotation_axis))
    
    # Apply the rotation
    q_result = np.array([
        q_rot[0]*q[0] - q_rot[1]*q[1] - q_rot[2]*q[2] - q_rot[3]*q[3],
        q_rot[0]*q[1] + q_rot[1]*q[0] + q_rot[2]*q[3] - q_rot[3]*q[2],
        q_rot[0]*q[2] - q_rot[1]*q[3] + q_rot[2]*q[0] + q_rot[3]*q[1],
        q_rot[0]*q[3] + q_rot[1]*q[2] - q_rot[2]*q[1] + q_rot[3]*q[0]
    ])
    
    return quat2r(q_result)

def get_transformed_pcd(pcd, p, R):
    """
        transformation pcd using p, R

        pcd : point cloud       np.ndarray [B, 3]
        p   : transition pose   np.ndarray [3,]
        R   : rotation matrix   np.ndarray [3,3]

        return transformed_pcd  np.ndarray [B, 3]
    """
    ones = np.ones((pcd.shape[0], 1))
    point_cloud_homogeneous = np.hstack((pcd, ones))  # Shape (2426, 4)
    
    # Create the 4x4 homogeneous transformation matrix
    homogeneous_matrix = np.eye(4)  # Start with an identity matrix
    homogeneous_matrix[:3, :3] = R  # Set the rotation matrix
    homogeneous_matrix[:3, 3] = p   # Set the translation vector

    # Apply the transformation matrix
    transformed_pcd_homogeneous = point_cloud_homogeneous @ homogeneous_matrix.T

    # Convert back to (x, y, z) by removing the homogeneous coordinate
    transformed_pcd = transformed_pcd_homogeneous[:, :3]
    return transformed_pcd

def project_onto_plane(v, normal):
    """
    Project vector v onto the plane orthogonal to the given normal vector.

    Parameters:
        v (np.ndarray): Input vector.
        normal (np.ndarray): Plane normal vector (not necessarily unit).

    Returns:
        np.ndarray: Projected vector on the plane.
    """
    return v - np.dot(v, normal) * normal

def quat_between_vectors(uv_fr, uv_to):
    """
    Compute the quaternion that rotates 'uv_fr' to 'uv_to'.

    Parameters:
        uv_fr (np.ndarray): Source unit vector.
        uv_to (np.ndarray): Target unit vector.

    Returns:
        np.ndarray: Quaternion [w, x, y, z] representing the rotation.
    """
    uv_fr = normalize(uv_fr)
    uv_to = normalize(uv_to)

    dot = np.clip(np.dot(uv_fr, uv_to), -1.0, 1.0)

    # Check if vectors are parallel (same direction)
    if dot > 0.99999:
        return np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion

    # Check if vectors are anti-parallel (opposite direction)
    if dot < -0.99999:
        # Find a perpendicular axis
        axis = np.cross(uv_fr, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(uv_fr, np.array([0.0, 1.0, 0.0]))
        axis = normalize(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])  # 180-degree rotation

    # General case
    cross = np.cross(uv_fr, uv_to)
    w = 1.0 + dot
    quat = np.array([w, cross[0], cross[1], cross[2]])
    return normalize(quat)

def angle_about_axis(uv_fr, uv_to, axis):
    """
    Compute the signed angle (in radians) to rotate 'uv_fr' to 'uv_to' about a given axis.

    Parameters:
        uv_fr (np.ndarray): Source unit vector.
        uv_to (np.ndarray): Target unit vector.
        axis (np.ndarray): Rotation axis (unit or not).

    Returns:
        float: Signed rotation angle in radians in range [-π, π].
    """
    axis = normalize(axis)
    v1 = normalize(project_onto_plane(uv_fr, axis))
    v2 = normalize(project_onto_plane(uv_to, axis))

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    magnitude = np.arccos(dot)
    sign = np.sign(np.dot(np.cross(v1, v2), axis))
    angle = sign*magnitude # angle in [-pi, pi]
    return angle

def axis_angle_from_quat(quat):
    """
    Extract the rotation axis and angle from a quaternion.

    Parameters:
        quat (np.ndarray): Quaternion [w, x, y, z].

    Returns:
        tuple: (axis, angle) where:
            - axis (np.ndarray): Rotation axis (unit vector).
            - angle (float): Rotation angle in radians in range [0, π].
    """
    quat = normalize(quat)
    w, x, y, z = quat

    # Ensure w is positive to get the shorter rotation
    if w < 0:
        w, x, y, z = -w, -x, -y, -z

    # Compute angle
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))

    # Handle near-zero rotation case
    if angle < 1e-6:
        return np.array([1.0, 0.0, 0.0]), 0.0  # arbitrary axis, zero angle

    # Compute axis
    sin_half_angle = np.sin(angle / 2)
    axis = np.array([x, y, z]) / sin_half_angle
    axis = normalize(axis)

    return axis, angle

def quat_from_axis_angle(axis, angle):
    """
    Create a quaternion representing a rotation around a given axis by angle (radians).

    Parameters:
        axis (np.ndarray): Rotation axis (unit vector).
        angle (float): Rotation angle in radians.

    Returns:
        np.ndarray: Quaternion [w, x, y, z].
    """
    axis = normalize(axis)
    half = angle / 2
    return np.array([
        np.cos(half),
        np.sin(half) * axis[0],
        np.sin(half) * axis[1],
        np.sin(half) * axis[2],
    ])

def R_from_axis_angle(axis, angle):
    """
    Create a rotation matrix from an axis and angle (in radians).

    Parameters:
        axis (np.ndarray): Rotation axis (unit vector).
        angle (float): Rotation angle in radians.

    Returns:
        R (np.ndarray): 3x3 rotation matrix.
    """
    quat = quat_from_axis_angle(axis, angle) # (4)
    R = quat2r(quat) # (3,3)
    return R

def R_from_axis(w,a=np.array([1,0,0])):
    """
    Create a rotation matrix from an angular velocity vector (axis-angle representation).

    Parameters:
        w (np.ndarray): Axis (3-element array).
        a (np.ndarray): Reference vector to determine the rotation axis (default is [1, 0, 0]).

    Returns:
        R (np.ndarray): 3x3 rotation matrix.
    """
    if np.linalg.norm(w) < 1e-6:
        return np.eye(3)  # Return identity if a is zero vector

    # z-axis of the rotation
    w = np.array(w, dtype=np.float64)
    ez = w / np.linalg.norm(w) # w to be the z-axis of the rotation

    # x-axis of the rotation
    u = a - ez * (a.dot(ez))
    ex = u / np.linalg.norm(u)

    # y-axis of the rotation
    ey = np.cross(ez, ex)

    # Return the rotation matrix
    R = np.stack([ex, ey, ez], axis=1)
    return R

def quat_from_vectors_about_axis(vec_from, vec_to, axis):
    """
    Create a quaternion representing the rotation from vec_from to vec_to about a given axis.

    Parameters:
        vec_from (np.ndarray): Source unit vector.
        vec_to (np.ndarray): Target unit vector.
        axis (np.ndarray): Rotation axis (unit or not).

    Returns:
        np.ndarray: Quaternion [w, x, y, z].
    """
    angle = angle_about_axis(vec_from, vec_to, axis)
    quat = quat_from_axis_angle(axis, angle)
    return quat

def quat_conjugate(quat):
    """
    Compute the conjugate of a quaternion.

    Parameters:
        quat (np.ndarray): Quaternion [w, x, y, z].

    Returns:
        np.ndarray: Conjugate quaternion [w, -x, -y, -z].
    """
    return np.array([quat[0],-quat[1],-quat[2],-quat[3]])

def quat_multiply(quat1,quat2):
    """
    Perform the Hamilton product (quaternion multiplication) of two quaternions.

    Parameters:
        q1, q2 (np.ndarray): Quaternions [w, x, y, z].

    Returns:
        np.ndarray: Resulting quaternion.
    """
    w1, x1, y1, z1 = quat1
    w2, x2, y2, z2 = quat2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rotate_vector_by_quat(vec, quat):
    """
    Rotate a vector v by quaternion quat.

    Parameters:
        vec (np.ndarray): 3D vector to rotate.
        quat (np.ndarray): Quaternion [w, x, y, z].

    Returns:
        np.ndarray: Rotated 3D vector.
    """
    v_quat = np.array([0, *vec])
    v_rot = quat_multiply(quat_multiply(quat,v_quat),quat_conjugate(quat))[1:] # v_rot = q*v*q*
    return v_rot

def rotate_uv_rate(uv_fr, uv_to, rate=1.0):
    """
    Rotate a unit vector (uv_fr) toward another (uv_to) using the shortest arc,
    with a scalar rate that supports interpolation (0 ≤ rate ≤ 1) and extrapolation (rate < 0 or rate > 1).
    
    Parameters:
        uv_fr (np.ndarray): Starting unit vector (shape: (3,))
        uv_to (np.ndarray): Target unit vector (shape: (3,))
        rate (float): Interpolation/extrapolation factor.

    Returns:
        np.ndarray: Resulting unit vector after rotation.
    """
    # Compute rotation quaternion from uv_fr to uv_to
    quat_fr2to = quat_between_vectors(uv_fr, uv_to)
    axis, angle = axis_angle_from_quat(quat_fr2to)

    # Compose scaled rotation quaternion
    quat_scaled = quat_from_axis_angle(axis, rate * angle)
    quat_inc = quat_from_axis_angle(axis, (rate-1.0)*angle)
    uv_rotated = rotate_vector_by_quat(uv_fr, quat_scaled)
    uv_rotated = normalize(uv_rotated)

    # Return
    info = {'quat_inc':quat_inc}
    return uv_rotated,info

def rotate_uv_by_axis_rate(uv_fr, uv_to, axis_fixed, rate=1.0):
    """
    Rotate a unit vector (uv_fr) toward another (uv_to) along a fixed axis,
    using a scalar rate (t) that supports both interpolation (t in [0,1]) and extrapolation (t < 0 or t > 1).
    If rate is 1.0, then uv_rotated will be exactly uv_to.
    
    Parameters:
        uv_fr (np.ndarray): Starting unit vector.
        uv_to (np.ndarray): Target unit vector.
        axis_fixed (np.ndarray): Fixed axis of rotation.
        rate (float): Interpolation/extrapolation factor.

    Returns:
        np.ndarray: Resulting unit vector after rotation.
    """
    # Get angle between vectors
    quat_fr2to = quat_between_vectors(uv_fr, uv_to)
    axis_fr2to, angle = axis_angle_from_quat(quat_fr2to)

    # Compute modified starting vector that lands on uv_to after rotating by +angle
    quat_back = quat_from_axis_angle(axis_fixed, -angle)
    uv_fr_mod = rotate_vector_by_quat(uv_to, quat_back)

    # Rotate that modified vector by rate*angle
    quat_rot = quat_from_axis_angle(axis_fixed, rate*angle)
    quat_inc = quat_from_axis_angle(axis_fixed, (rate-1.0)*angle)
    uv_rotated = rotate_vector_by_quat(uv_fr_mod, quat_rot)
    uv_rotated = normalize(uv_rotated)

    # Return
    info = {'axis_fr2to':axis_fr2to,'quat_rot':quat_rot,'quat_inc':quat_inc,'uv_fr_mod':uv_fr_mod}
    return uv_rotated,info

def get_plane_basis(normal):
    """
    Get two orthonormal basis vectors on the plane orthogonal to a given normal.

    Parameters:
        normal (np.ndarray): Plane normal vector.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two orthogonal unit vectors in the plane.
    """
    normal = normalize(normal)
    temp = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    v1 = normalize(np.cross(normal, temp))
    v2 = normalize(np.cross(normal, v1))
    return v1, v2

def generate_circle_points(center, normal, radius=0.2, n=32):
    """
    Generate 3D points forming a circle in a plane.

    Parameters:
        center (np.ndarray): Center of the circle.
        normal (np.ndarray): Plane normal vector.
        radius (float): Radius of the circle.
        n (int): Number of points.

    Returns:
        np.ndarray: Array of 3D points along the circle.
    """
    v1, v2 = get_plane_basis(normal)
    angles = np.linspace(0, 2 * np.pi, n)
    return np.array([center + radius * (np.cos(a) * v1 + np.sin(a) * v2) for a in angles])

def draw_rotation_plane(
        env, 
        center, 
        axis, 
        radius   = 0.3, 
        n_circle = 32, 
        n_spoke  = 12, 
        n_rings  = 3, 
        r_line   = 0.004,
        rgba     = (0.7,0.7,0.9,0.4),
        label    = None,
    ):
    """
    Draw a wheel-like rotation plane: outer circle + spokes + concentric rings.

    Parameters:
        env: Environment with rendering API.
        center (np.ndarray): Center of the plane.
        axis (np.ndarray): Normal vector of the plane.
        radius (float): Size of the plane.
        n_circle (int): Number of points forming the outer circle.
        n_spoke (int): Number of radial lines (spokes).
        n_rings (int): Number of concentric rings (excluding outermost).
        rgba (tuple): RGBA color for all plotted components.        
    """
    axis = normalize(axis)
    v1,v2 = get_plane_basis(axis)

    # Outer circle
    angles = np.linspace(0,2*np.pi,n_circle)
    circle_pts = [center + radius*(np.cos(a)*v1 + np.sin(a)*v2) for a in angles]
    for i in range(len(circle_pts)-1):
        env.plot_cylinder_fr2to(circle_pts[i],circle_pts[i+1],r=r_line, rgba=rgba)
    env.plot_cylinder_fr2to(circle_pts[-1], circle_pts[0], r=r_line, rgba=rgba)

    # Spokes
    spoke_angles = np.linspace(0, 2 * np.pi, n_spoke, endpoint=False)
    for a in spoke_angles:
        end = center + radius*(np.cos(a)*v1 + np.sin(a)*v2)
        env.plot_cylinder_fr2to(center, end, r=r_line, rgba=rgba)

    # Inner rings
    for i in range(1, n_rings):
        r_ring = radius * i / n_rings
        ring_pts = [center + r_ring * (np.cos(a) * v1 + np.sin(a) * v2) for a in angles]
        for j in range(len(ring_pts) - 1):
            env.plot_cylinder_fr2to(ring_pts[j], ring_pts[j + 1], r=r_line, rgba=rgba)
        env.plot_cylinder_fr2to(ring_pts[-1], ring_pts[0], r=r_line, rgba=rgba)

    # Label
    if label is not None:
        env.plot_text(center, label)

def draw_quat_rotation_plane(
        env, 
        center, 
        quat, 
        radius   = 0.3, 
        n_circle = 32, 
        n_spoke  = 12, 
        n_rings  = 3, 
        rgba     = (0.7,0.7,0.9,0.4),
    ):
    """
    Draw a wheel-like rotation plane: outer circle + spokes + concentric rings.

    Parameters:
        env: Environment with rendering API.
        center (np.ndarray): Center of the plane.
        quat (np.ndarray): Quaternion representing the rotation.
        radius (float): Size of the plane.
        n_circle (int): Number of points forming the outer circle.
        n_spoke (int): Number of radial lines (spokes).
        n_rings (int): Number of concentric rings (excluding outermost).
        rgba (tuple): RGBA color for all plotted components.        
    """
    axis = normalize(quat[1:])  # [x, y, z] part of quaternion
    draw_rotation_plane(
        env      = env,
        center   = center,
        axis     = axis,
        radius   = radius,
        n_circle = n_circle,
        n_spoke  = n_spoke,
        n_rings  = n_rings,
        rgba     = rgba,
    )

def draw_rotation_arc(
        env,
        center,
        axis,
        vec_fr,
        angle,
        n_arc       = 32,
        r_arc       = 0.01,
        rgba_fr     = (1.0, 0.5, 0.0, 0.8),  # start color
        rgba_to     = (0.2, 0.5, 1.0, 0.8),  # end color
        alpha       = None,
        n_spoke     = 0,
        r_spoke     = 0.005,
        alpha_spoke = None,
    ):
    """
    Draw a gradient arc that rotates vec_from around axis by angle (rad), from center.

    Parameters:
        env: Rendering environment with .plot_cylinder_fr2to
        center (np.ndarray): Origin point of rotation.
        axis (np.ndarray): Rotation axis (not necessarily unit).
        vec_fr (np.ndarray): Vector to rotate (with magnitude).
        angle (float): Rotation angle (radian, can be signed).
        n (int): Number of segments.
        rgba_fr (tuple): RGBA at start of arc.
        rgba_to (tuple): RGBA at end of arc.
    """
    axis = normalize(axis)
    vec_fr = np.array(vec_fr)
    arc_points = []
    arc_angles = np.linspace(0,angle,n_arc)

    for theta in arc_angles:
        q = quat_from_axis_angle(axis, theta)
        rotated = rotate_vector_by_quat(vec_fr,q)
        arc_points.append(center + rotated)

    rgba_fr = np.array(rgba_fr)
    rgba_to = np.array(rgba_to)

    for i in range(len(arc_points) - 1):
        t = i / (len(arc_points) - 1)
        color = (1 - t) * rgba_fr + t * rgba_to
        if alpha is not None:
            color[3] = alpha
        env.plot_cylinder_fr2to(arc_points[i],arc_points[i+1],r=r_arc,rgba=color)

    # Optionally draw spokes
    if n_spoke > 0:
        spoke_indices = np.linspace(0, len(arc_points) - 1, n_spoke, dtype=int)
        for idx in spoke_indices:
            t = idx / (len(spoke_indices) - 1)
            color = (1 - t) * rgba_fr + t * rgba_to
            if alpha_spoke is not None:
                color[3] = alpha_spoke
            env.plot_cylinder_fr2to(center, arc_points[idx], r=r_spoke, rgba=color)

def draw_rotation_arc_fr_to(
        env,
        center,
        vec_fr,
        vec_to,
        n       = 32,
        r       = 0.01,
        rgba_fr = (1.0, 0.5, 0.0, 0.8),  # start color
        rgba_to = (0.2, 0.5, 1.0, 0.8),  # end color
        alpha   = None,
    ):
    """
    Draw a gradient arc that rotates vec_from to vec_to around the inferred axis from center.

    Parameters:
        env: Rendering environment with .plot_cylinder_fr2to
        center (np.ndarray): Origin point of rotation.
        vec_from (np.ndarray): Start vector (with magnitude).
        vec_to (np.ndarray): Target vector (with same origin and scale).
        n (int): Number of segments in the arc.
        rgba_fr (tuple): RGBA at start of arc.
        rgba_to (tuple): RGBA at end of arc.
        alpha (float, optional): Override alpha channel.
    """
    vec_fr = np.array(vec_fr)
    vec_to = np.array(vec_to)

    # Compute rotation axis and angle
    axis  = np.cross(vec_fr, vec_to)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return  # Vectors are parallel or zero; skip drawing
    axis  = axis / axis_norm
    angle = angle_about_axis(vec_fr, vec_to, axis)

    # Generate intermediate arc points
    arc_points = []
    arc_angles = np.linspace(0, angle, n)
    for theta in arc_angles:
        q = quat_from_axis_angle(axis, theta)
        rotated = rotate_vector_by_quat(vec_fr, q)
        arc_points.append(center + rotated)

    # Draw arc with gradient
    rgba_fr = np.array(rgba_fr)
    rgba_to = np.array(rgba_to)
    for i in range(len(arc_points) - 1):
        t = i / (len(arc_points) - 1)
        color = (1 - t) * rgba_fr + t * rgba_to
        if alpha is not None:
            color[3] = alpha
        env.plot_cylinder_fr2to(arc_points[i], arc_points[i+1], r=r, rgba=color.tolist())

def tf_view_fr(p: np.ndarray, 
               R: Optional[np.ndarray] = None, 
               T_view_fr: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Transform points and rotations from global coordinate to the view from T_view_fr.
    
    This function applies the inverse transformation of T_view_fr to see how the points
    and rotations would appear from the perspective of T_view_fr coordinate frame.
    
    Parameters:
        p (np.ndarray): Points in global coordinate
                       - Shape: (3,) or (n,3) or (n,m,3) or higher dimensions ending with 3
        R (np.ndarray, optional): Rotation matrices in global coordinate
                                 - Shape: (3,3) or (n,3,3) or (n,m,3,3) or higher dimensions ending with (3,3)
                                 - Default: None
        T_view_fr (np.ndarray): 4x4 transformation matrix representing the viewpoint frame
                               - Shape: (4,4)
    
    Returns:
        If R is None:
            np.ndarray: Transformed points with same shape as input p
        If R is provided:
            Tuple[np.ndarray, np.ndarray]: (transformed_points, transformed_rotations)
    
    Example:
        # Transform single point
        p = np.array([1, 2, 3])
        T_view_fr = np.eye(4)
        p_transformed = tf_view_fr(p, T_view_fr=T_view_fr)
        
        # Transform multiple points with rotations
        p = np.random.rand(5, 3)
        R = np.random.rand(5, 3, 3)
        T_view_fr = np.eye(4)
        p_trans, R_trans = tf_view_fr(p, R, T_view_fr)
    """
    
    if T_view_fr is None:
        raise ValueError("T_view_fr must be provided")
    
    if T_view_fr.shape != (4, 4):
        raise ValueError(f"T_view_fr must be (4,4) matrix, got shape {T_view_fr.shape}")
    
    # Convert inputs to numpy arrays
    p = np.asarray(p)
    if R is not None:
        R = np.asarray(R)
    
    # Validate point dimensions
    if p.shape[-1] != 3:
        raise ValueError(f"Points must have last dimension of 3, got shape {p.shape}")
    
    # Validate rotation dimensions if provided
    if R is not None:
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"Rotations must have last two dimensions of (3,3), got shape {R.shape}")
    
    # Get inverse transformation matrix
    T_inv = np.linalg.inv(T_view_fr)
    R_inv = T_inv[:3, :3]  # Rotation part
    t_inv = T_inv[:3, 3]   # Translation part
    
    # Store original shape for reshaping later
    original_p_shape = p.shape
    
    # Reshape points to (..., 3) for easier processing
    p_flat = p.reshape(-1, 3)
    
    # Transform points: p_new = R_inv @ (p - t) = R_inv @ p - R_inv @ t
    # Vectorized computation: (N, 3) @ (3, 3).T = (N, 3)
    p_transformed_flat = (p_flat @ R_inv.T) + t_inv #(t_inv @ R_inv.T)
    
    # Reshape back to original shape
    p_transformed = p_transformed_flat.reshape(original_p_shape)
    
    # Transform rotations if provided
    if R is not None:
        # Store original rotation shape
        original_R_shape = R.shape
        
        # Reshape rotations to (..., 3, 3)
        R_flat = R.reshape(-1, 3, 3)
        
        # Transform rotations: R_new = R_inv @ R
        # Vectorized computation using einsum
        R_transformed_flat = np.einsum('ij,njk->nik', R_inv, R_flat)
        
        # Reshape back to original shape
        R_transformed = R_transformed_flat.reshape(original_R_shape)
        
        # Return both transformed points and rotations
        return p_transformed, R_transformed
    
    # Return transformed points only
    return p_transformed

def tf_view_to(p: np.ndarray, 
               R: Optional[np.ndarray] = None, 
               T_view_to: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Transform points defined at origin when the origin moves to T_view_to.
    
    This maintains the relative positions of points with respect to the new origin.
    If a point was at (x,y,z) relative to the old origin, it will still be at 
    (x,y,z) relative to the new origin at T_view_to.
    
    Parameters:
        p (np.ndarray): Points defined at origin
        R (np.ndarray, optional): Rotation matrices at origin
        T_view_to (np.ndarray): 4x4 transformation matrix where the new origin will be
    
    Returns:
        Transformed points/rotations in global coordinate system
    
    Example:
        p = np.array([1, 2, 3])  # Point at (1,2,3) from origin
        T_view_to = np.eye(4)
        T_view_to[:3, 3] = [5, 6, 0]  # New origin at (5,6,0)
        p_global = tf_view_to(p, T_view_to=T_view_to)
        # p_global will be [6, 8, 3] = [5+1, 6+2, 0+3]
    """
    
    if T_view_to is None:
        raise ValueError("T_view_to must be provided")
    
    if T_view_to.shape != (4, 4):
        raise ValueError(f"T_view_to must be (4,4) matrix, got shape {T_view_to.shape}")
    
    # Convert inputs to numpy arrays
    p = np.asarray(p)
    if R is not None:
        R = np.asarray(R)
    
    # Validate point dimensions
    if p.shape[-1] != 3:
        raise ValueError(f"Points must have last dimension of 3, got shape {p.shape}")
    
    # Validate rotation dimensions if provided
    if R is not None:
        if R.shape[-2:] != (3, 3):
            raise ValueError(f"Rotations must have last two dimensions of (3,3), got shape {R.shape}")
    
    # Extract rotation and translation from T_view_to
    R_new = T_view_to[:3, :3]  # Rotation part
    t_new = T_view_to[:3, 3]   # Translation part
    
    # Store original shape for reshaping later
    original_p_shape = p.shape
    
    # Reshape points to (..., 3) for easier processing
    p_flat = p.reshape(-1, 3)
    
    # Transform points: p_global = R_new @ p_local + t_new
    # This moves the local coordinate system to T_view_to
    p_transformed_flat = (p_flat @ R_new.T) + t_new
    
    # Reshape back to original shape
    p_transformed = p_transformed_flat.reshape(original_p_shape)
    
    # Transform rotations if provided
    if R is not None:
        # Store original rotation shape
        original_R_shape = R.shape
        
        # Reshape rotations to (..., 3, 3)
        R_flat = R.reshape(-1, 3, 3)
        
        # Transform rotations: R_global = R_new @ R_local
        # Vectorized computation using einsum
        R_transformed_flat = np.einsum('ij,njk->nik', R_new, R_flat)
        
        # Reshape back to original shape
        R_transformed = R_transformed_flat.reshape(original_R_shape)
        
        # Return both transformed points and rotations
        return p_transformed, R_transformed
    
    # Return transformed points only
    return p_transformed

def tf_global(T_in, p_global=(0, 0, 0), R_global=np.eye(3), T_global=None):
    """
    Applies a global transformation to the given transformation matrix T_in.

    This function applies an additional transformation defined in the global coordinate frame
    (via p_global and R_global) to the transformation matrix T_in.

    Parameters:
        T_in (np.ndarray): A 4x4 transformation matrix.
        p_global (tuple, optional): Translation to apply in the global coordinate frame (default: (0, 0, 0)).
        R_global (np.ndarray, optional): Rotation to apply in the global coordinate frame (default: identity).
        T_global (np.ndarray, optional): A 4x4 transformation matrix defining the global transform.
                                         If provided, it overrides p_global and R_global.

    Returns:
        np.ndarray: The resulting transformation matrix after applying the global transform.
                    Computed as T_global @ T_in.
    """
    if T_global is None:
        T_global = pr2t(p=p_global, R=R_global)
    return T_global @ T_in

def tf_local(T_in, p_local=(0, 0, 0), R_local=np.eye(3), T_local=None):
    """
    Applies a local transformation to the given transformation matrix T_in.

    This function applies an additional transformation defined in T_in's local coordinate frame
    (via p_local and R_local) to T_in.

    Parameters:
        T_in (np.ndarray): A 4x4 transformation matrix.
        p_local (tuple, optional): Translation to apply in the local coordinate frame (default: (0, 0, 0)).
        R_local (np.ndarray, optional): Rotation to apply in the local coordinate frame (default: identity).
        T_local (np.ndarray, optional): A 4x4 transformation matrix defining the local transform.
                                        If provided, it overrides p_local and R_local.

    Returns:
        np.ndarray: The resulting transformation matrix after applying the local transform.
                    Computed as T_in @ T_local.
    """
    if T_local is None:
        T_local = pr2t(p=p_local, R=R_local)
    return T_in @ T_local

def get_p_offset(p_fr: np.ndarray, p_to: np.ndarray) -> np.ndarray:
    """
    Compute the translation offset from one position to another.

    Given a source position p_fr and a target position p_to, returns the
    vector p_offset such that:
        p_to == p_fr + p_offset

    Parameters:
        p_fr (np.ndarray): Source position, shape (3,) or (N, 3).
        p_to (np.ndarray): Target position, shape (3,) or (N, 3).

    Returns:
        np.ndarray: The offset vector(s), same shape as inputs.
    """
    p_fr = np.asarray(p_fr, dtype=float)
    p_to = np.asarray(p_to, dtype=float)
    return p_to - p_fr


def get_R_offset(R_fr: np.ndarray, R_to: np.ndarray) -> np.ndarray:
    """
    Compute the rotation offset from one orientation to another.

    Given a source rotation matrix R_fr and a target rotation matrix R_to,
    returns the rotation R_offset such that:
        R_to == R_offset @ R_fr

    Parameters:
        R_fr (np.ndarray): Source rotation matrix, shape (3, 3).
        R_to (np.ndarray): Target rotation matrix, shape (3, 3).

    Returns:
        np.ndarray: The rotation offset matrix, shape (3, 3).
    """
    R_fr = np.asarray(R_fr, dtype=float)
    R_to = np.asarray(R_to, dtype=float)
    # For a proper rotation matrix, inverse is its transpose
    return R_to @ R_fr.T

def apply_p_offset(p_fr: np.ndarray, p_offset: np.ndarray) -> np.ndarray:
    """
    Apply a translation offset to positions.

    Broadcasts over any leading batch dimensions.

    Parameters:
        p_fr (np.ndarray): Original position(s), shape (..., 3).
        p_offset (np.ndarray): Offset vector(s), shape (..., 3), broadcastable to p_fr.

    Returns:
        np.ndarray: Translated position(s), shape (..., 3), where:
            result == p_fr + p_offset
    """
    p_fr = np.asarray(p_fr, dtype=float)
    p_offset = np.asarray(p_offset, dtype=float)
    return p_fr + p_offset

def apply_R_offset(R_fr: np.ndarray, R_offset: np.ndarray) -> np.ndarray:
    """
    Apply a rotation offset to orientation matrices.

    Broadcasts over any leading batch dimensions.

    Parameters:
        R_fr (np.ndarray): Original rotation(s), shape (..., 3, 3).
        R_offset (np.ndarray): Rotation offset(s), shape (..., 3, 3), broadcastable to R_fr.

    Returns:
        np.ndarray: New rotation matrix/matrices, shape (..., 3, 3), where:
            result == R_offset @ R_fr
    """
    R_fr = np.asarray(R_fr, dtype=float)
    R_offset = np.asarray(R_offset, dtype=float)
    # batch matmul: (...,3,3) @ (...,3,3)
    return np.matmul(R_offset, R_fr)

def get_R_offset_local(R_fr: np.ndarray, R_to: np.ndarray) -> np.ndarray:
    """
    Compute the rotation offset from one orientation to another in a local frame.
    """
    R_fr = np.asarray(R_fr, dtype=float)
    R_to = np.asarray(R_to, dtype=float)
    # For a proper rotation matrix, inverse is its transpose
    return R_fr.T @ R_to

def apply_R_offset_local(R_fr: np.ndarray, R_offset: np.ndarray) -> np.ndarray:
    """
    Apply a rotation offset to orientation matrices in a local frame.
    """
    R_fr = np.asarray(R_fr, dtype=float)
    R_offset = np.asarray(R_offset, dtype=float)
    # batch matmul: (...,3,3) @ (...,3,3)
    return np.matmul(R_fr, R_offset)


