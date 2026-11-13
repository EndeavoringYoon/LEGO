# SMPLModel is adapted from CalciferZh/SMPL (https://github.com/CalciferZh/SMPL), smpl_np.py.
# Copyright (c) 2018 CalciferZh. MIT License; see THIRD_PARTY_NOTICES.md.
import numpy as np
import pickle

"""
    We assume that the below code has been called. 
    sys.path.append('../../package/helper/')
"""
from transforms import t2p,t2r,pr2t,rpy2r

class SMPLModel():
    """
        (Deprecated)
        Minimal implementation of SMPL 
        https://github.com/CalciferZh/SMPL
    """
    def __init__(self, model_path):
        """
            SMPL model.

            Parameter:
            ---------
            model_path: Path to the SMPL model parameters, pre-processed by
            `preprocess.py`.
        """
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
        self.J_regressor   = params['J_regressor']
        self.weights       = params['weights']
        self.posedirs      = params['posedirs']
        self.v_template    = params['v_template']
        self.shapedirs     = params['shapedirs']
        self.faces         = params['f']
        self.kintree_table = params['kintree_table']

        id_to_col = {
            self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
        }
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }

        self.pose_shape  = [24, 3]
        self.beta_shape  = [10]
        self.trans_shape = [3]

        self.pose  = np.zeros(self.pose_shape)
        self.beta  = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)

        self.verts = None
        self.J     = None
        self.R     = None
        
        # Joint names
        self.joint_names = [
            'base',
            'lp',
            'rp',
            'spine1',
            'lk',
            'rk',
            'spine2',
            'la',
            'ra',
            'spine3',
            'lf',
            'rf',
            'neck',
            'lc',
            'rc',
            'head',
            'ls',
            'rs',
            'le',
            're',
            'lw',
            'rw',
            'lh',
            'rh',
        ]
        self.n_joint = len(self.joint_names )

        # Initial update
        self.update()

    def set_params(self, pose=None, beta=None, trans=None):
        """
            Set pose, shape, and/or translation parameters of SMPL model. Verices of the
            model will be updated and returned.

            Parameters:
            ---------
            pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
            relative to parent joint. For root joint it's global orientation.
            Represented in a axis-angle format.

            beta: Parameter for model shape. A vector of shape [10]. Coefficients for
            PCA component. Only 10 components were released by MPI.

            trans: Global translation of shape [3].

            Return:
            ------
            Updated vertices.
        """
        if pose is not None:
            self.pose = pose
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
        self.update()
        return self.verts

    def update(self):
        """
            Called automatically when parameters are updated.
        """
        # how beta affect body shape
        v_shaped = self.shapedirs.dot(self.beta) + self.v_template
        # joints location
        self.J = self.J_regressor.dot(v_shaped)
        pose_cube = self.pose.reshape((-1, 1, 3)) # [24 x 1 x 3]
        # rotation matrix for each joint
        self.R = self.rodrigues(pose_cube) # [24 x 3 x 3]
        I_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            (self.R.shape[0]-1, 3, 3)
        ) # [23 x 3 x 3]
        lrotmin = (self.R[1:] - I_cube).ravel()
        # how pose affect body shape in zero pose
        v_posed = v_shaped + self.posedirs.dot(lrotmin)
        # world transformation of each joint
        G = np.empty((self.kintree_table.shape[1], 4, 4))
        G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i] = G[self.parent[i]].dot(
                self.with_zeros(
                    np.hstack(
                        [self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
                    )
                )
            )
        self.G0 = G.copy()
        G = G - self.pack(
            np.matmul(
                G,
                np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
            )
        )
        self.G = G.copy()
        # transformation of each vertex
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        self.T = T
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        self.verts = v + self.trans.reshape([1, 3])

    def rodrigues(self, r):
        """
            Rodrigues' rotation formula that turns axis-angle vector into rotation
            matrix in a batch-ed manner.

            Parameter:
            ----------
            r: Axis-angle rotation vector of shape [batch_size, 1, 3].

            Return:
            -------
            Rotation matrix of shape [batch_size, 3, 3].
        """
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(r.dtype).eps)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x):
        """
            Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

            Parameter:
            ---------
            x: Matrix to be appended.

            Return:
            ------
            Matrix after appending of shape [4,4]
        """
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

    def pack(self, x):
        """
            Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
            manner.

            Parameter:
            ----------
            x: Matrices to be appended of shape [batch_size, 4, 1]

            Return:
            ------
            Matrix of shape [batch_size, 4, 4] after appending.
        """
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

    def save_to_obj(self, path):
        """
            Save the SMPL model into .obj file.

            Parameter:
            ---------
            path: Path to save.
        """
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def get_T_joints(self,T_offset=None):
        """ 
            Get joint coordinates 
            'self.set_params()' should be called in advance
        """
        
        # Trnaslate SMPL skeleton using 'self.trans'
        T_joints  = self.G0.copy() # [24 x 4 x 4]
        T_offset_trans = pr2t(p=self.trans,R=rpy2r(np.radians([0,0,0]))) # [4 x 4]
        T_joints = np.einsum('ij,kjl->kil', T_offset_trans, T_joints) # pre-multiply to globally transform
        
        # Make SMPL skeleton Z-up and X-front to be comaptible with MuJoCo
        T_offset_zup = pr2t(p=np.zeros(3),R=rpy2r(np.radians([90,0,90]))) # [4 x 4]
        T_joints = np.einsum('ij,kjl->kil', T_offset_zup, T_joints) # pre-multiply to globally transform
        
        if T_offset is not None: # apply offset
            T_joints = np.einsum('ij,kjl->kil', T_offset, T_joints) # pre-multiply to globally transform
        return T_joints # [24 x 4 x 4]
    
    def get_vertices(self,T_offset=None):
        """ 
            Get vertices
        """
        T_offset_zup = pr2t(p=np.array([0,0,0]),R=rpy2r(np.radians([90,0,90]))) # [4 x 4]
        # z-up
        vertices = self.verts.copy() # [M x 3]
        vertices = np.hstack([vertices, np.ones((vertices.shape[0],1))]) # [M x 4]
        vertices = vertices @ T_offset_zup.T  # [M x 4]
        
        if T_offset is not None: # apply offset
            vertices = vertices @ T_offset.T # [M x 4]
        
        vertices = vertices[:,:3] # [M x 3]
        return vertices
    
    def plot_in_mujoco(self,env,print_joint=False):
        """ 
            Plot SMPL skeleton in MuJoCo
        """
        # Get joint poses
        T_smpl = self.get_T_joints(T_offset=None)
        
        # Plot SMPL-Skeleton's joint
        for j_idx in range(self.n_joint): # for each joint
            p = t2p(T_smpl[j_idx,:])
            R = t2r(T_smpl[j_idx,:])
            if print_joint:
                label = ' [%d] %s'%(j_idx,self.joint_names[j_idx])
            else:
                label = ''
            env.plot_T(
                p=p,
                R=R,
                plot_axis=True,axis_len=0.05,axis_width=0.005,axis_rgba=None,
                plot_sphere=True,sphere_r=0.01,sphere_rgba=(0,0,1,0.5),
                label=label,
            )
            # Link to paraent
            if j_idx in self.parent.keys(): # if parent exists
                parent   = self.parent[j_idx]
                p_parent = t2p(T_smpl[parent,:])
                env.plot_cylinder_fr2to(
                    p_fr = p,
                    p_to = p_parent,
                    r    = 0.005,
                    rgba = (0.4,0.4,0.4,0.5))
        
class SMPLModelClass(object):
    """
        Minimal implementation of SMPL
        https://github.com/CalciferZh/SMPL
    """
    def __init__(self, model_path):
        """
            Initialize SMPL model.
            model_path: Path to the SMPL model parameters, pre-processed by
            `preprocess.py`.
        """
        # Open SMPL model path and parse items
        with open(model_path,'rb') as f:
            params = pickle.load(f)
        self.J_regressor   = params['J_regressor']
        self.weights       = params['weights']
        self.posedirs      = params['posedirs']
        self.v_template    = params['v_template']
        self.shapedirs     = params['shapedirs']
        self.faces         = params['f']
        self.kintree_table = params['kintree_table']
        id_to_col = {
            self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
        }
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }
        
        self.pose_shape  = [24,3]
        self.beta_shape  = [10]
        self.trans_shape = [3]

        self.pose  = np.zeros(self.pose_shape)
        self.beta  = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)

        self.verts = None
        self.J     = None
        self.R     = None
        
        # Joint names
        self.joint_names = [
            'base',
            'lp',
            'rp',
            'spine1',
            'lk',
            'rk',
            'spine2',
            'la',
            'ra',
            'spine3',
            'lf',
            'rf',
            'neck',
            'lc',
            'rc',
            'head',
            'ls',
            'rs',
            'le',
            're',
            'lw',
            'rw',
            'lh',
            'rh',
        ]
        self.n_joint = len(self.joint_names )

        # Initial update
        self.update()
        
    def set_params(
            self, 
            pose     = None, 
            beta     = None, 
            trans    = None,
            T_offset = None,
        ):
        """
            Set pose, shape, and/or translation parameters of SMPL model. Verices of the
            model will be updated and returned.

            Parameters:
            ---------
            pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
            relative to parent joint. For root joint it's global orientation.
            Represented in a axis-angle format.

            beta: Parameter for model shape. A vector of shape [10]. Coefficients for
            PCA component. Only 10 components were released by MPI.

            trans: Global translation of shape [3].
        """
        if pose is not None:
            self.pose = pose
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
            
        # Update SMPL skeleton 
        self.update()
        
        # Update vertices
        T_offset_zup = pr2t(p=np.array([0,0,0]),R=rpy2r(np.radians([90,0,90]))) # [4 x 4]
        # z-up
        vertices = self.verts.copy() # [M x 3]
        vertices = np.hstack([vertices, np.ones((vertices.shape[0],1))]) # [M x 4]
        vertices = vertices @ T_offset_zup.T  # [M x 4]
        
        if T_offset is not None: # apply offset
            vertices = vertices @ T_offset.T # [M x 4]
        self.vertices = vertices[:,:3] # [M x 3]
        
        # Trnaslate SMPL skeleton using 'self.trans'
        self.T_joints  = self.G0.copy() # [24 x 4 x 4]
        
        # Batch pre-multiply to globally transform (using einsum)
        T_offset_trans = pr2t(p=self.trans,R=rpy2r(np.radians([0,0,0]))) # [4 x 4]
        self.T_joints = np.einsum('ij,kjl->kil', T_offset_trans, self.T_joints) # [24 x 4 x 4]
        
        # Make SMPL skeleton Z-up and X-front to be comaptible with MuJoCo
        T_offset_zup = pr2t(p=np.zeros(3),R=rpy2r(np.radians([90,0,90]))) # [4 x 4]
        # Batch Pre-multiply to globally transform
        self.T_joints = np.einsum('ij,kjl->kil', T_offset_zup, self.T_joints) # [24 x 4 x 4]
        
        if T_offset is not None: # apply offset
            self.T_joints = np.einsum('ij,kjl->kil', T_offset, self.T_joints) # [24 x 4 x 4]
        
    
    def update(self):
        """
            Called automatically when parameters are updated.
        """
        # how beta affect body shape
        v_shaped = self.shapedirs.dot(self.beta) + self.v_template
        # joints location
        self.J = self.J_regressor.dot(v_shaped)
        pose_cube = self.pose.reshape((-1, 1, 3)) # [24 x 1 x 3]
        # rotation matrix for each joint
        self.R = self.rodrigues(pose_cube) # [24 x 3 x 3]
        I_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            (self.R.shape[0]-1, 3, 3)
        ) # [23 x 3 x 3]
        lrotmin = (self.R[1:] - I_cube).ravel()
        # how pose affect body shape in zero pose
        v_posed = v_shaped + self.posedirs.dot(lrotmin)
        
        # world transformation of each joint
        G = np.empty((self.kintree_table.shape[1], 4, 4))
        G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i] = G[self.parent[i]].dot(
                self.with_zeros(
                    np.hstack(
                        [self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
                    )
                )
            )
        self.G0 = G.copy() # <= Joints in task space
        G = G - self.pack(
            np.matmul(
                G,
                np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
            )
        )
        self.G = G.copy() 
        
        # transformation of each vertex
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        self.T = T
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        self.verts = v + self.trans.reshape([1, 3])

    def rodrigues(self, r):
        """
            Rodrigues' rotation formula that turns axis-angle vector into rotation
            matrix in a batch-ed manner.

            Parameter:
            ----------
            r: Axis-angle rotation vector of shape [batch_size, 1, 3].

            Return:
            -------
            Rotation matrix of shape [batch_size, 3, 3].
        """
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(r.dtype).eps)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x):
        """
            Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

            Parameter:
            ---------
            x: Matrix to be appended.

            Return:
            ------
            Matrix after appending of shape [4,4]
        """
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

    def pack(self, x):
        """
            Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
            manner.

            Parameter:
            ----------
            x: Matrices to be appended of shape [batch_size, 4, 1]

            Return:
            ------
            Matrix of shape [batch_size, 4, 4] after appending.
        """
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))
    
    def plot_in_mujoco(
            self,
            env,
            plot_vertices = False,
            print_joint   = False,
            link_r        = 0.005,
            link_rgba     = (0,0,0,1.0),
        ):
        """ 
            Plot SMPL skeleton in MuJoCo
        """
        # Get joint poses and vertices
        T_smpl   = self.T_joints
        vertices = self.vertices
        
        # Plot vertices
        if plot_vertices:
            for p in vertices: env.plot_sphere(p=p,r=0.0025,rgba=(1,1,1,0.5))
            
        # Plot SMPL-Skeleton's joint
        for j_idx in range(self.n_joint): # for each joint
            p = t2p(T_smpl[j_idx,:])
            R = t2r(T_smpl[j_idx,:])
            if print_joint:
                label = ' [%d] %s'%(j_idx,self.joint_names[j_idx])
            else:
                label = ''
            env.plot_T(
                p=p,
                R=R,
                plot_axis=True,axis_len=0.05,axis_width=0.005,axis_rgba=None,
                plot_sphere=True,sphere_r=0.01,sphere_rgba=(0,0,1,0.5),
                label=label,
            )
            # Link to paraent
            if j_idx in self.parent.keys(): # if parent exists
                parent   = self.parent[j_idx]
                p_parent = t2p(T_smpl[parent,:])
                env.plot_cylinder_fr2to(
                    p_fr = p,
                    p_to = p_parent,
                    r    = link_r,
                    rgba = link_rgba,
                )