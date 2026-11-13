import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.traversal.depth_first_search import dfs_edges

def pr2t_torch(p, R, dtype=torch.float32):
    """
    Convert pose to transformation matrix in PyTorch
    Args:
        p: torch.Tensor - Position vector [3] or [Bx3]
        R: torch.Tensor - Rotation matrix [3x3] or [Bx3x3]
    Returns:
        torch.Tensor - Transformation matrix [4x4] or [Bx4x4]
    """
    # Check if input is batched
    is_batched = len(p.shape) > 1
    device = p.device
    
    if is_batched:
        B = p.shape[0]
        p0 = p.reshape(B, 3)  # Ensure shape is [B, 3]
        zeros = torch.zeros(B, 1, 3, device=device, dtype=dtype)  # [B, 1, 3]
        ones = torch.ones(B, 1, 1, device=device, dtype=dtype)    # [B, 1, 1]
        
        # Stack position vector as [B, 3, 1]
        p0 = p0.unsqueeze(-1)  # [B, 3, 1]
        
        # Create transformation matrix [B, 4, 4]
        T = torch.cat([
            torch.cat([R, p0], dim=2),           # [B, 3, 4]
            torch.cat([zeros, ones], dim=2)      # [B, 1, 4]
        ], dim=1)
        
    else:
        p0 = p.reshape(3)  # Ensure shape is [3]
        zeros = torch.zeros(3, device=device, dtype=dtype)
        
        # Stack position vector as [3, 1]
        p0 = p0.unsqueeze(-1)  # [3, 1]
        
        # Create transformation matrix [4, 4]
        T = torch.cat([
            torch.cat([R, p0], dim=1),           # [3, 4]
            torch.cat([zeros, torch.tensor([1.0], device=device, dtype=dtype)]).unsqueeze(0)  # [1, 4]
        ], dim=0)
        
    return T

def t2pr_torch(T):
    """Convert transformation matrix to position and rotation in PyTorch"""
    is_batched = len(T.shape) > 2
    
    if is_batched:
        p = T[..., :3, 3]      # [B, 3]
        R = T[..., :3, :3]     # [B, 3, 3]
    else:
        p = T[:3, 3]           # [3]
        R = T[:3, :3]          # [3, 3]
    return p, R

def t2p_torch(T):
    """Convert transformation matrix to position in PyTorch"""
    is_batched = len(T.shape) > 2
    
    if is_batched:
        p = T[..., :3, 3]      # [B, 3]
    else:
        p = T[:3, 3]           # [3]
    return p

def t2r_torch(T):
    """Convert transformation matrix to rotation in PyTorch"""
    is_batched = len(T.shape) > 2
    
    if is_batched:
        R = T[..., :3, :3]     # [B, 3, 3]
    else:
        R = T[:3, :3]          # [3, 3]
    return R

def skew_torch(v, dtype=torch.float32):
    """
    Create a skew-symmetric matrix from a 3D vector in PyTorch.
    Parameters:
        v: torch.Tensor - 3D vector [3] or [B, 3]
    Returns:
        torch.Tensor - Skew-symmetric matrix [3, 3] or [B, 3, 3]
    """
    if len(v.shape) == 1:
        # Single vector input
        device = v.device
        zero = torch.zeros(1, device=device, dtype=dtype)
        skew_mat = torch.tensor([[zero, -v[2], v[1]],
                               [v[2], zero, -v[0]],
                               [-v[1], v[0], zero]], device=device, dtype=dtype)
        return skew_mat.squeeze()
    else:
        # Batched input
        device = v.device
        B = v.shape[0]
        skew_mat = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        skew_mat[:, 0, 1] = -v[:, 2]
        skew_mat[:, 0, 2] = v[:, 1]
        skew_mat[:, 1, 0] = v[:, 2]
        skew_mat[:, 1, 2] = -v[:, 0]
        skew_mat[:, 2, 0] = -v[:, 1]
        skew_mat[:, 2, 1] = v[:, 0]
        return skew_mat

def rodrigues_torch(a=None, q_rad=0.0, dtype=torch.float32):
    """
    Compute the rotation matrix from an angular velocity vector in PyTorch.
    Parameters:
        a: torch.Tensor - Angular velocity vector [3]
        q_rad: float or torch.Tensor - Rotation angle in radians
    Returns:
        torch.Tensor - Rotation matrix [3, 3]
    """
    if a is None:
        a = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=dtype)
    if not isinstance(q_rad, torch.Tensor):
        q_rad = torch.tensor(q_rad)
        
    device = a.device
    a_norm = torch.norm(a)
    
    if torch.abs(a_norm - 1) > 1e-6:
        print(f"[rodrigues] norm of a should be 1.0 not [{a_norm:.2e}].")
        return torch.eye(3, device=device, dtype=dtype)
    
    a = a / a_norm
    q_rad = q_rad * a_norm
    a_hat = skew_torch(a)
    
    R = (torch.eye(3, device=device, dtype=dtype) + 
         a_hat * torch.sin(q_rad) + 
         torch.matmul(a_hat, a_hat) * (1 - torch.cos(q_rad)))
    return R

def rodrigues_batch_torch(a=None, q_rad=None, dtype=torch.float32):
    """
    Compute the rotation matrices from angular velocity vectors in PyTorch (batch support).
    Parameters:
        a: torch.Tensor, shape [B, 3] - Angular velocity unit vectors for each batch
        q_rad: torch.Tensor, shape [B] - Rotation angles (in radians) for each batch
    Returns:
        torch.Tensor - Rotation matrices [B, 3, 3]
    """
    if a is None:
        a = torch.tensor([[1.0, 0.0, 0.0]])
    if q_rad is None:
        q_rad = torch.zeros(1)
        
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(q_rad, torch.Tensor):
        q_rad = torch.tensor(q_rad)
        
    device = a.device
    B = a.shape[0]

    # Check norms
    a_norm = torch.norm(a, dim=1, keepdim=True)
    if torch.any(torch.abs(a_norm - 1) > 1e-6):
        print("[rodrigues] Norm of each vector in a should be 1.0.")
        return torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

    # Compute skew-symmetric matrices for all vectors in batch
    a_hat = skew_torch(a)  # shape [B, 3, 3]

    # Reshape angle for broadcasting
    sin_q = torch.sin(q_rad).view(-1, 1, 1)  # shape [B, 1, 1]
    cos_q = torch.cos(q_rad).view(-1, 1, 1)  # shape [B, 1, 1]

    # Compute rotation matrices
    R = (torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1) +
         a_hat * sin_q +
         torch.matmul(a_hat, a_hat) * (1 - cos_q))

    return R

def quat2r_torch(q, dtype=torch.float32):
    """
    Convert quaternion to rotation matrix in PyTorch.
    Parameters:
        q: torch.Tensor - Quaternion [w, x, y, z] shape [4] or [B, 4]
    Returns:
        torch.Tensor - Rotation matrix [3, 3] or [B, 3, 3]
    """
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q, dtype=dtype)
    
    # Handle both single and batched inputs
    is_batched = len(q.shape) > 1
    if not is_batched:
        w, x, y, z = q
    else:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=-1),
        torch.stack([2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w], dim=-1),
        torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y], dim=-1)
    ], dim=-2)
    
    return R

class TorchChainClass(object):
    """
        Kinematic chain using Torch
    """
    def __init__(self,name='Torch Chain',device='cpu',dtype=torch.float32):
        """
        Initialize Chain Class
        Parameters:
            name: str - Name of the chain.
        """
        self.chain  = None
        self.device = device
        self.dtype = dtype
        self.init_chain(name=name) # initialize chain
        
    def init_chain(self,name=None):
        """
        Initialize chain
        Parameters:
            name: str - Name of the chain.
        """
        # Set name
        self.name = name
        
        # Clear and instantiate chain
        if self.chain is not None:
            self.chain.clear()
        self.chain = nx.DiGraph(name=name)
        
        # Clear body and joint names
        self.body_names  = []
        self.joint_names = []
        
    def get_n_body(self):
        """
        Get the number of bodies.
        Returns:
            int - Number of bodies in the chain.
        """
        return self.chain.number_of_nodes()
    
    def get_body_idx(self,body_name):
        """
        Get the index of a body.
        Parameters:
            body_name: str - Name of the body.
        Returns:
            int - Index of the body.
        """
        body_idx = self.body_names.index(body_name)
        return body_idx
    
    def get_body_idxs(self,body_names):
        """
        Get the indices of bodies.
        Parameters:
            body_names: list of str - Names of the bodies.
        Returns:
            list of int - Indices of the bodies.
        """
        body_idxs = [self.body_names.index(name) 
                     if name in self.body_names else None 
                     for name in body_names]
        return body_idxs
    
    def get_root_body_idx(self):
        """
        Get the root body index.
        Returns:
            int - Index of the root body.
        """
        for node in self.chain.nodes:
            if self.chain.in_degree(node) == 0:  # the root node is the node with in-degree 0
                return list(self.chain.nodes).index(node)
        raise ValueError("No root node found in the chain.")
    
    def get_joint_idx(self,joint_name):
        """
        Get the index of a joint.
        Parameters:
            joint_name: str - Name of the joint.
        Returns:
            int - Index of the joint.
        """
        joint_idx = self.joint_names.index(joint_name)
        return joint_idx
    
    def get_body(self,body_idx):
        """
        Get body information.
        Parameters:
            body_idx: int - Index of the body.
        Returns:
            dict - Information of the body.
        """
        body = self.chain.nodes[body_idx]
        return body
    
    def get_p_body(self,body_name=None,body_idx=None):
        """
        Get the position of a body.
        Parameters:
            body_name: str, optional - Name of the body.
            body_idx: int, optional - Index of the body.
        Returns:
            torch.Tensor - Position of the body.
        """
        if body_name is not None:
            body_idx = self.get_body_idx(body_name=body_name)
        elif body_idx is not None:
            body_idx = body_idx
        else:
            raise ValueError("Either body name or index should be given.")
        return self.chain.nodes[body_idx]['p']
    
    def get_p_batch_body(self,body_name=None,body_idx=None):
        """
        Get batched positions of a body.
        Parameters:
            body_name: str, optional - Name of the body.
            body_idx: int, optional - Index of the body.
        Returns:
            torch.Tensor - Batched positions of the body.
        """
        if body_name is not None:
            body_idx = self.get_body_idx(body_name=body_name)
        elif body_idx is not None:
            body_idx = body_idx
        else:
            raise ValueError("Either body name or index should be given.")
        return self.chain.nodes[body_idx]['p_batch']
    
    def get_R_body(self, body_name=None, body_idx=None):
        """
        Get the rotation matrix of a body.
        Parameters:
            body_name: str, optional - Name of the body.
            body_idx: int, optional - Index of the body.
        Returns:
            torch.Tensor - Rotation matrix of the body.
        """
        if body_name is not None:
            body_idx = self.get_body_idx(body_name=body_name)
        elif body_idx is not None:
            body_idx = body_idx
        else:
            raise ValueError("Either body name or index should be given.")
        return self.chain.nodes[body_idx]['R']
    
    def get_R_batch_body(self, body_name=None, body_idx=None):
        """
        Get batched rotation matrices of a body.
        Parameters:
            body_name: str, optional - Name of the body.
            body_idx: int, optional - Index of the body.
        Returns:
            torch.Tensor - Batched rotation matrices of the body.
        """
        if body_name is not None:
            body_idx = self.get_body_idx(body_name=body_name)
        elif body_idx is not None:
            body_idx = body_idx
        else:
            raise ValueError("Either body name or index should be given.")
        return self.chain.nodes[body_idx]['R_batch']
    
    def get_T_body(self, body_name=None, body_idx=None):
        """
        Get the transformation matrix of a body.
        Parameters:
            body_name: str, optional - Name of the body.
            body_idx: int, optional - Index of the body.
        Returns:
            torch.Tensor - Transformation matrix of the body.
        """
        return pr2t_torch(
            p = self.get_p_body(body_name, body_idx),
            R = self.get_R_body(body_name, body_idx),
        )
        
    def set_body_info(self,body_idx,key,value):
        """
        Set body information using a key and value.
        Parameters:
            body_idx: int - Index of the body.
            key: str - Key of the information.
            value: Any - Value to set.
        """
        self.chain.nodes[body_idx][key] = value
        
    def set_p_body(self, body_name=None, body_idx=None, p=None):
        """
        Set the position of a body.
        Parameters:
            body_name: str, optional - Name of the body.
            body_idx: int, optional - Index of the body.
            p: torch.Tensor - Position to set.
        """
        if body_name is not None:
            body_idx = self.get_body_idx(body_name=body_name)
        elif body_idx is not None:
            body_idx = body_idx
        else:
            raise ValueError("Either body name or index should be given.")
        if isinstance(p, torch.Tensor):
            p = p.to(self.device)
        self.chain.nodes[body_idx]['p'] = p
        
    def set_R_body(self, body_name=None, body_idx=None, R=None):
        """
        Set the rotation matrix of a body.
        Parameters:
            body_name: str, optional - Name of the body.
            body_idx: int, optional - Index of the body.
            R: torch.Tensor - Rotation matrix to set.
        """
        if body_name is not None:
            body_idx = self.get_body_idx(body_name=body_name)
        elif body_idx is not None:
            body_idx = body_idx
        else:
            raise ValueError("Either body name or index should be given.")
        if isinstance(R, torch.Tensor):
            R = R.to(self.device)
        self.chain.nodes[body_idx]['R'] = R
        
    def set_T_body(self, body_name=None, body_idx=None, T=None):
        """
        Set the transformation matrix of a body.
        Parameters:
            body_name: str, optional - Name of the body.
            body_idx: int, optional - Index of the body.
            T: torch.Tensor - Transformation matrix to set.
        """
        if body_name is not None:
            body_idx = self.get_body_idx(body_name=body_name)
        elif body_idx is not None:
            body_idx = body_idx
        else:
            raise ValueError("Either body name or index should be given.")
        if isinstance(T, torch.Tensor):
            T = T.to(self.device)
        self.chain.nodes[body_idx]['p'] = t2p_torch(T)
        self.chain.nodes[body_idx]['R'] = t2r_torch(T)
        
    def set_q_joint(self, joint_name, q):
        """
        Set the joint angle for a joint.
        Parameters:
            joint_name: str - Name of the joint.
            q: float or torch.Tensor - Joint angle to set.
        """
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, device=self.device, dtype=self.dtype)
        self.chain.nodes[self.get_joint_idx(joint_name)]['q'] = q.to(self.device)
        
    def set_qs_joints(self, joint_names, qs, forward=False, root_body_name=None):
        """
        Set joint angles for multiple joints.
        Parameters:
            joint_names: list of str - Names of the joints.
            qs: list or torch.Tensor - Joint angles to set.
            forward: bool, optional - Whether to perform forward kinematics after setting.
            root_body_name: str, optional - Name of the root body.
        """
        if not isinstance(qs, torch.Tensor):
            qs = torch.tensor(qs, device=self.device, dtype=self.dtype)
        for joint_name, q in zip(joint_names, qs):
            self.set_q_joint(joint_name=joint_name, q=q)
        if forward:
            self.forward(root_body_name=root_body_name)
            
    def set_qs_joints_batch(self, joint_names, q_batch):
        """
        Set batched joint angles for multiple joints.
        Parameters:
            joint_names: list of str - Names of the joints.
            q_batch: torch.Tensor, shape [B, N] - Batch of joint angles for all joints.
        """
        if not isinstance(q_batch, torch.Tensor):
            q_batch = torch.tensor(q_batch, device=self.device, dtype=self.dtype)
        B, N = q_batch.shape
        for i, joint_name in enumerate(joint_names):
            q_values = q_batch[:, i]  # shape [B]
            self.chain.nodes[self.get_joint_idx(joint_name)]['q_batch'] = q_values.to(self.device)
            
    def add_body(
            self,
            body_name='',
            parent_body_name='',
            joint_name='',
            p_offset=None,
            R_offset=None,
            a=None,
        ):
        """
        Add a body to the chain.
        Parameters:
            body_name: str - Name of the body.
            parent_body_name: str - Name of the parent body.
            joint_name: str - Name of the joint associated with the body.
            p_offset: torch.Tensor - Positional offset of the body.
            R_offset: torch.Tensor - Rotational offset of the body.
            a: torch.Tensor - Joint axis of the body.
        """
        # Set default tensor values if None
        if p_offset is None:
            p_offset = torch.zeros(3, device=self.device, dtype=self.dtype)
        if R_offset is None:
            R_offset = torch.eye(3, device=self.device, dtype=self.dtype)
        if a is None:
            a = torch.zeros(3, device=self.device, dtype=self.dtype)
            
        # Convert numpy arrays to torch tensors if necessary
        if not isinstance(p_offset, torch.Tensor):
            p_offset = torch.tensor(p_offset, device=self.device, dtype=self.dtype)
        if not isinstance(R_offset, torch.Tensor):
            R_offset = torch.tensor(R_offset, device=self.device, dtype=self.dtype)
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, device=self.device, dtype=self.dtype)
        
        # Add new body
        new_body_idx = self.get_n_body()
        self.chain.add_node(new_body_idx)
        
        # Update body information
        body_info = {
            'name': body_name,
            'parent_name': parent_body_name,
            'joint_name': joint_name,
            'p_offset': p_offset,
            'R_offset': R_offset,
            'a': a,
            'q': torch.tensor(0.0, device=self.device, dtype=self.dtype),
            'p': torch.zeros(3, device=self.device, dtype=self.dtype),
            'R': torch.eye(3, device=self.device, dtype=self.dtype),
            'parent': [],
            'childs': [],
        }
        self.chain.update(nodes=[(new_body_idx, body_info)])
        
        # Append body and joint names
        self.body_names.append(body_name)
        self.joint_names.append(joint_name)
        
        # Add parent body (if necessary)
        if (parent_body_name is not None) and (body_name != parent_body_name):
            parent_body_idx = self.get_body_idx(parent_body_name)
            self.chain.nodes[new_body_idx]['parent'] = parent_body_idx
            self.chain.add_edge(u_of_edge=parent_body_idx, v_of_edge=new_body_idx)
            
        # Append childs to the parent
        parent_body_idx = self.get_body_idx(body_name)
        parent_childs = self.chain.nodes[parent_body_idx]['childs']
        parent_childs.append(new_body_idx)
        
    def build_chain_from_mujoco_env(
            self,
            env=None,
            init_first=True,
        ):
        """
        Build a kinematic chain from a MuJoCo environment.
        Parameters:
            env: MuJoCo environment - The environment to parse.
            init_first: bool, optional - Whether to reset the chain before building.
        """
        # Reset first
        if init_first:
            self.init_chain(name=env.name)
            
        for body_idx in range(env.n_body):
            # Parse body information
            body_name = env.body_names[body_idx]
            body = env.model.body(body_name) 
            parent_body_name = env.body_names[body.parentid[0]]
            
            # Convert position and quaternion to torch tensors
            p_body_offset = torch.tensor(body.pos, device=self.device, dtype=self.dtype)
            quat_body_offset = torch.tensor(body.quat, device=self.device, dtype=self.dtype)
            
            # Parse joint information
            n_joint = body.jntnum  # number of attached joints 
            if n_joint == 1:
                joint = env.model.joint(body.jntadr[0])  # joint attached joint
                joint_name = joint.name
                # Convert joint position and axis to torch tensors
                p_joint_offset = torch.tensor(joint.pos, device=self.device, dtype=self.dtype)
                joint_axis = torch.tensor(joint.axis, device=self.device, dtype=self.dtype)
            else:
                joint_name = ''
                p_joint_offset = torch.zeros(3, device=self.device, dtype=self.dtype)
                joint_axis = torch.zeros(3, device=self.device, dtype=self.dtype)
                
            # Add body to the chain
            self.add_body(
                body_name=body_name,
                parent_body_name=parent_body_name,
                joint_name=joint_name,
                p_offset=p_body_offset,
                R_offset=quat2r_torch(quat_body_offset),  # quat2r should be torch version
                a=joint_axis,
            )
        
    def forward(self, joint_names=None, q=None, root_body_name=None):
        """
        Perform forward kinematics.
        Parameters:
            joint_names: list of str, optional - Names of the joints.
            q: torch.Tensor or np.ndarray, optional - Joint angles.
            root_body_name: str, optional - Name of the root body.
        """
        # Convert q to tensor if it's numpy array
        if isinstance(q, np.ndarray):
            q = torch.tensor(q, device=self.device, dtype=self.dtype)
            
        # Set joint angles (optional)
        if joint_names is not None:
            self.set_qs_joints(joint_names=joint_names, qs=q)
        
        # Get root body index
        if root_body_name is None:
            root_body_idx = self.get_root_body_idx()
        else:
            root_body_idx = self.get_body_idx(root_body_name)
            
        # Recursively update p and R
        for edge in dfs_edges(self.chain, source=root_body_idx):
            idx_fr = edge[0]
            idx_to = edge[1]
            body_fr = self.get_body(idx_fr)
            body_to = self.get_body(idx_to)
            
            # Get and convert tensors
            R_fr = body_fr['R']
            p_fr = body_fr['p']
            p_offset = body_to['p_offset']
            R_offset = body_to['R_offset']
            
            # Convert to tensors if numpy arrays
            if isinstance(R_fr, np.ndarray):
                R_fr = torch.tensor(R_fr, device=self.device, dtype=self.dtype)
            if isinstance(p_fr, np.ndarray):
                p_fr = torch.tensor(p_fr, device=self.device, dtype=self.dtype)
            if isinstance(p_offset, np.ndarray):
                p_offset = torch.tensor(p_offset, device=self.device, dtype=self.dtype)
            if isinstance(R_offset, np.ndarray):
                R_offset = torch.tensor(R_offset, device=self.device, dtype=self.dtype)
                
            # Ensure correct dtype
            R_fr = R_fr.to(dtype=self.dtype)
            p_fr = p_fr.to(dtype=self.dtype)
            p_offset = p_offset.to(dtype=self.dtype)
            R_offset = R_offset.to(dtype=self.dtype)
            
            # Update p
            p = torch.matmul(R_fr, p_offset) + p_fr  # [3]
            self.set_body_info(body_idx=idx_to, key='p', value=p)
            
            # Update R
            a_to = body_to['a']
            if isinstance(a_to, np.ndarray):
                a_to = torch.tensor(a_to, device=self.device, dtype=self.dtype)
            else:
                a_to = a_to.to(dtype=self.dtype)
                
            if torch.abs(torch.norm(a_to) - 1) < 1e-6:  # revolute joint
                q_to = body_to['q']  # [1]
                if isinstance(q_to, np.ndarray):
                    q_to = torch.tensor(q_to, device=self.device, dtype=self.dtype)
                else:
                    q_to = q_to.to(dtype=self.dtype)
                    
                R = torch.matmul(
                    torch.matmul(R_fr, R_offset),
                    rodrigues_torch(a=a_to, q_rad=q_to, dtype=self.dtype)
                )  # [3x3]
            else:
                R = torch.matmul(R_fr, R_offset)  # [3x3]
                
            self.set_body_info(body_idx=idx_to, key='R', value=R)

    def forward_batch(self, joint_names=None, q_batch=None, root_body_name=None):
        """
        Perform batched forward kinematics.
        Parameters:
            joint_names: list of str, optional - Names of the joints.
            q_batch: torch.Tensor or np.ndarray, shape [B, N], optional - Batch of joint angles for all joints.
            root_body_name: str, optional - Name of the root body.
        """
        # Convert q_batch to tensor if it's numpy array
        if isinstance(q_batch, np.ndarray):
            q_batch = torch.tensor(q_batch, device=self.device, dtype=self.dtype)

        if joint_names is not None:
            self.set_qs_joints_batch(joint_names=joint_names, q_batch=q_batch)

        root_body_idx = self.get_root_body_idx() if root_body_name is None else self.get_body_idx(root_body_name)

        B = q_batch.shape[0]  # Number of batches

        # Initialize root body's p and R for batch
        root_body = self.get_body(root_body_idx)
        root_p = root_body['p']  # shape [3]
        root_R = root_body['R']  # shape [3, 3]

        # Convert to tensor if numpy array
        if isinstance(root_p, np.ndarray):
            root_p = torch.tensor(root_p, device=self.device, dtype=self.dtype)
        if isinstance(root_R, np.ndarray):
            root_R = torch.tensor(root_R, device=self.device, dtype=self.dtype)

        # Convert to correct dtype if needed
        root_p = root_p.to(dtype=self.dtype)
        root_R = root_R.to(dtype=self.dtype)

        # Broadcast root position and rotation to batch size
        root_p_batch = root_p.unsqueeze(0).expand(B, -1)  # shape [B, 3]
        root_R_batch = root_R.unsqueeze(0).expand(B, -1, -1)  # shape [B, 3, 3]

        # Update batched position and rotation for the root body
        root_body['p_batch'] = root_p_batch.to(self.device)
        root_body['R_batch'] = root_R_batch.to(self.device)

        for edge in dfs_edges(self.chain, source=root_body_idx):
            idx_fr = edge[0]
            idx_to = edge[1]
            body_fr = self.get_body(idx_fr)
            body_to = self.get_body(idx_to)

            # Update p_batch
            p_fr_batch = body_fr['p_batch']  # shape [B, 3]
            R_fr_batch = body_fr['R_batch']  # shape [B, 3, 3]
            p_offset = body_to['p_offset']  # shape [3]

            # Convert p_offset to tensor if numpy array
            if isinstance(p_offset, np.ndarray):
                p_offset = torch.tensor(p_offset, device=self.device, dtype=self.dtype)
            else:
                p_offset = p_offset.to(dtype=self.dtype)

            p_to_batch = torch.matmul(R_fr_batch, p_offset.unsqueeze(-1)).squeeze(-1) + p_fr_batch  # shape [B, 3]
            body_to['p_batch'] = p_to_batch

            # Update R_batch
            a_to = body_to['a']  # shape [3]
            if isinstance(a_to, np.ndarray):
                a_to = torch.tensor(a_to, device=self.device, dtype=self.dtype)
            else:
                a_to = a_to.to(dtype=self.dtype)

            R_offset = body_to['R_offset']
            if isinstance(R_offset, np.ndarray):
                R_offset = torch.tensor(R_offset, device=self.device, dtype=self.dtype)
            else:
                R_offset = R_offset.to(dtype=self.dtype)

            if torch.abs(torch.norm(a_to) - 1) < 1e-6:  # Revolute joint
                q_to_batch = body_to['q_batch']  # shape [B]
                if isinstance(q_to_batch, np.ndarray):
                    q_to_batch = torch.tensor(q_to_batch, device=self.device, dtype=self.dtype)
                else:
                    q_to_batch = q_to_batch.to(dtype=self.dtype)

                R_to_batch = torch.matmul(
                    torch.matmul(R_fr_batch, R_offset),
                    rodrigues_batch_torch(
                        a=a_to.unsqueeze(0).expand(B, -1), 
                        q_rad=q_to_batch
                    )
                )  # shape [B, 3, 3]
            else:
                R_to_batch = torch.matmul(R_fr_batch, R_offset)  # shape [B, 3, 3]

            body_to['R_batch'] = R_to_batch
            
    def plot_graph(
            self,
            align='horizontal',
            figsize=(6,4),
            node_size=100,
            nose_font_size=6,
            node_colors=None,
            title_font_size=10,
            root_on_top=True,
        ):
        """
        Plot the kinematic chain graph.
        Parameters:
            align: str, optional - Layout alignment ('horizontal' or 'vertical').
            figsize: tuple, optional - Figure size.
            node_size: int, optional - Size of the nodes.
            nose_font_size: int, optional - Font size for node labels.
            node_colors: list, optional - Colors of the nodes.
            title_font_size: int, optional - Font size of the title.
            root_on_top: bool, optional - Whether to place the root node on top.
        """
        n_body = self.get_n_body()
        tree = self.chain
        for layer, nodes in enumerate(nx.topological_generations(tree)):
            for node in nodes:
                tree.nodes[node]['layer'] = layer
        pos = nx.multipartite_layout(
            tree,
            align=align,
            scale=1.0,
            subset_key='layer',
        )
        # Invert the tree so that the root node comes on the top
        if root_on_top:
            pos = {node: (x, -y) for node, (x, y) in pos.items()} 
            
        # Plot nodes
        fig, ax = plt.subplots(figsize=figsize)
        if node_colors is None:  # set colors
            node_colors = []
            for body_idx in range(n_body):
                a = self.get_body(body_idx)['a']
                if torch.norm(a) < 1e-6:
                    node_color = (1,1,1,0.5)
                else:
                    node_color = [0,0,0,0.5]
                    node_color[torch.argmax(a)] = 1
                    node_color = tuple(node_color)
                node_colors.append(node_color)
                
        nx.draw_networkx(
            tree,
            pos=pos,
            ax=ax,
            with_labels=True,
            node_size=node_size,
            font_size=nose_font_size,
            node_color=node_colors,
            linewidths=1,
            edgecolors='k'
        )
        ax.set_title(f'{tree.name}', fontsize=title_font_size)
        fig.tight_layout()
        plt.show()

    def plot_chain_mujoco(
            self,
            env,  # <= assumed to be instantiated from 'MuJoCoParserClass'
            plot_link=True,
            r_link=0.01,
            rgba_link=(0,0,1,0.25),
            root_body_name=None,
            plot_body=True,
            plot_body_name=False,
            plot_body_axis=True,
            axis_len=0.025,
            axis_width=0.0025,
            rate=1.0,
            plot_body_sphere=False,
            r_body=0.025,
            rgba_body=(0,0,0,0.5),
            plot_rev_axis=True,
            r_axis=0.01,
            h_axis=0.05,
            alpha_axis=0.5,
        ):
        """
        Plot the kinematic chain in a MuJoCo environment.
        Parameters:
            env: MuJoCo environment - Environment to plot.
            plot_link: bool, optional - Whether to plot links.
            r_link: float, optional - Radius of the links.
            rgba_link: tuple, optional - Color of the links (RGBA).
            root_body_name: str, optional - Name of the root body.
            plot_body: bool, optional - Whether to plot bodies.
            plot_body_name: bool, optional - Whether to display body names.
            plot_body_axis: bool, optional - Whether to plot body axes.
            axis_len: float, optional - Length of the axes.
            axis_width: float, optional - Width of the axes.
            rate: float, optional - Scaling factor.
            plot_body_sphere: bool, optional - Whether to plot bodies as spheres.
            r_body: float, optional - Radius of the body spheres.
            rgba_body: tuple, optional - Color of the bodies (RGBA).
            plot_rev_axis: bool, optional - Whether to plot revolute axes.
            r_axis: float, optional - Radius of the axes.
            h_axis: float, optional - Height of the axes.
            alpha_axis: float, optional - Transparency of the axes.
        """
        if root_body_name is None:
            root_body_idx = self.get_root_body_idx()
        else:
            root_body_idx = self.get_body_idx(body_name=root_body_name)
            
        # Plot link
        if plot_link:
            for idx, edge in enumerate(dfs_edges(self.chain, source=root_body_idx)):
                body_fr = self.get_body(edge[0])
                body_to = self.get_body(edge[1])
                p_fr = body_fr['p']
                p_to = body_to['p']
                
                # Convert to tensor if numpy array
                if isinstance(p_fr, np.ndarray):
                    p_fr = torch.tensor(p_fr, device=self.device, dtype=self.dtype)
                if isinstance(p_to, np.ndarray):
                    p_to = torch.tensor(p_to, device=self.device, dtype=self.dtype)
                    
                env.plot_cylinder_fr2to(
                    p_fr=p_fr.cpu().numpy() if isinstance(p_fr, torch.Tensor) else p_fr,
                    p_to=p_to.cpu().numpy() if isinstance(p_to, torch.Tensor) else p_to,
                    r=r_link,
                    rgba=rgba_link,
                )
                
        # Plot body
        if plot_body:
            subtree = nx.dfs_tree(self.chain, source=root_body_idx)
            for body_idx in list(subtree.nodes):
                body = self.get_body(body_idx)
                if plot_body_name:
                    body_name = body['name']
                else:
                    body_name = ''
                    
                p = body['p']
                R = body['R']
                
                # Convert to tensor if numpy array
                if isinstance(p, np.ndarray):
                    p = torch.tensor(p, device=self.device, dtype=self.dtype)
                if isinstance(R, np.ndarray):
                    R = torch.tensor(R, device=self.device, dtype=self.dtype)
                    
                env.plot_T(
                    p=p.cpu().numpy() if isinstance(p, torch.Tensor) else p,
                    R=R.cpu().numpy() if isinstance(R, torch.Tensor) else R,
                    plot_axis=plot_body_axis,
                    axis_len=rate*axis_len,
                    axis_width=rate*axis_width,
                    plot_sphere=plot_body_sphere,
                    sphere_r=r_body,
                    sphere_rgba=rgba_body,
                    label=body_name,
                )
                
        # Plot revolute axis
        if plot_rev_axis:
            subtree = nx.dfs_tree(self.chain, source=root_body_idx)
            for body_idx in list(subtree.nodes):
                body = self.get_body(body_idx)
                a = body['a']
                if torch.norm(a) > 1e-6:
                    p = body['p']
                    R = body['R']
                    
                    # Convert to tensor if numpy array
                    if isinstance(p, np.ndarray):
                        p = torch.tensor(p, device=self.device, dtype=self.dtype)
                    if isinstance(R, np.ndarray):
                        R = torch.tensor(R, device=self.device, dtype=self.dtype)
                    
                    p2 = p + torch.matmul(R, a*rate*h_axis)
                    axis_color = [0,0,0,alpha_axis]
                    axis_color[torch.argmax(a)] = 1
                    env.plot_arrow_fr2to(
                        p_fr=p.cpu().numpy() if isinstance(p, torch.Tensor) else p,
                        p_to=p2.cpu().numpy() if isinstance(p2, torch.Tensor) else p2,
                        r=rate*r_axis,
                        rgba=axis_color
                    )
                    
