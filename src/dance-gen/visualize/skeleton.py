import torch
import einops
import numpy as np
import torch.nn.functional as F
from typing import NewType
Tensor = NewType("Tensor", torch.Tensor)

from pytorch3d.transforms import axis_angle_to_matrix

smpl_joints = [
    "root",  # 0
    "lhip",  
    "rhip",  
    "belly", 
    "lknee", 
    "rknee", 
    "spine", 
    "lankle",# 7
    "rankle",# 8
    "chest", # 9
    "ltoes", # 10
    "rtoes", # 11
    "neck",  
    "linshoulder", 
    "rinshoulder", 
    "head", 
    "lshoulder", 
    "rshoulder", 
    "lelbow", 
    "relbow", 
    "lwrist", # 20
    "rwrist", # 21
    "lhand", # 22
    "rhand", # 23
]

smplh_joints = [
    'pelvis',   # 0
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',  # 21
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1', # 25
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',  # 36
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',     # 40
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3'  # 51
]

smplx_joints = [
    'pelvis',   # 0
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',   # 7
    'right_ankle',  # 8
    'spine3',   
    'left_foot',    # 10
    'right_foot',   # 11
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist', 
    'jaw',  # 22
    'left_eye_smplhf',
    'right_eye_smplhf',
    'left_index1',  # 25
    'left_index2',
    'left_index3',
    'left_middle1', # 28
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',    # 43
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3'  # 55
]

smpl_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
smplh_parents =[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 
                20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 
                21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50]
smplx_parents =[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 
                15, 15, 15, 
                20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 
                21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53]

smpl_from_smplh = list(range(22)) + [25, 40]
smpl_from_smplx = list(range(22)) + [28, 43]
smplh_from_smplx = list(range(22)) + list(range(25, 55))


@torch.no_grad()
class SMPLX_Skeleton:
    def __init__(self, smplx_joint_path: str = "assets/smplx_model/smplx_neu_J_1.npy", with_normal: bool = False):
        self.joint_t_pose = self.load_joints(smplx_joint_path, correct_direction=True, with_normal=with_normal)
    
    def load_joints(self, smplx_joint_path, correct_direction=True, with_normal=False):
        smplx_joints = np.load(smplx_joint_path)
        smplx_joints = smplx_joints - smplx_joints[0]
        if correct_direction:
            smplx_joints = smplx_joints[:, (2, 0, 1)]
        if with_normal:
            smplx_joints = np.concatenate([smplx_joints, np.array([[1., 0., 0.]], dtype=smplx_joints.dtype)])
        return smplx_joints
    
    def batch_rodrigues(self, rot_vecs: Tensor, epsilon: float = 1e-8,) -> Tensor:
        ''' Calculates the rotation matrices for a batch of rotation vectors
            Parameters
            ----------
            rot_vecs: torch.tensor (..., 3)
                array of N axis-angle vectors
            Returns
            -------
            R: torch.tensor (..., 3, 3)
                The rotation matrices for the given axis-angle parameters
        '''
        # device, dtype = rot_vecs.device, rot_vecs.dtype
        # prefix_shape = rot_vecs.shape[:-1]  # (..., )

        # angle = torch.norm(rot_vecs + epsilon, dim=-1, keepdim=True)            # (..., 1)
        # rot_dir = rot_vecs / angle                                              # (..., 3)

        # cos = einops.repeat(torch.cos(angle), '... 1 -> ... 1 1')
        # sin = einops.repeat(torch.sin(angle), '... 1 -> ... 1 1')               # (..., 1, 1)

        # # Bx1 arrays
        # rx, ry, rz = torch.split(rot_dir, 1, dim=-1)                            # (..., 1)
        # zeros = torch.zeros(prefix_shape + (1,), dtype=dtype, device=device)    # (..., 1)

        # K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1)
        # K = einops.rearrange(K, '... (h w) -> ... h w', h=3, w=3)               # (..., 3, 3)

        # ident = torch.eye(3, dtype=dtype, device=device)
        # rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)
        # return rot_mat
        return axis_angle_to_matrix(rot_vecs)


    def batch_rigid_transform(self,
        rot_mats: Tensor,
        joints: Tensor,
        parents: list[int],
        dtype=torch.float32
    ) -> Tensor:
        """
        Applies a batch of rigid transformations to the joints

        Parameters
        ----------
        rot_mats : torch.tensor (..., J, 3, 3)
            Tensor of rotation matrices
        joints : torch.tensor (..., J, 3)
            Locations of joints
        parents : list[int] 
            The kinematic tree of each object
        dtype : torch.dtype, optional:
            The data type of the created tensors, the default is torch.float32

        Returns
        -------
        posed_joints : torch.tensor (..., J, 3)
            The locations of the joints after applying the pose rotations
        """

        joints = torch.unsqueeze(joints, dim=-1)                    # (..., J, 3, 1)
        # joints_check = joints.detach().cpu().numpy()

        rel_joints = joints.clone()
        rel_joints[..., 1:, :, :] -= joints[..., parents[1:], :, :]   # (..., J, 3, 1)

        transforms_mat = self.transform_mat(rot_mats, rel_joints)   # (..., J, 4, 4)

        transform_chain = [transforms_mat[..., 0, :, :]]
        for i in range(1, len(parents)):
            # Subtract the joint location at the rest pose
            # No need for rotation, since it's identity when at rest
            curr_res = torch.matmul(transform_chain[parents[i]],
                                    transforms_mat[..., i, :, :])
            transform_chain.append(curr_res)

        transforms = torch.stack(transform_chain, dim=-3)

        posed_joints = transforms[..., :, :3, 3]                     # (..., J, 3)
        
        return posed_joints

    def transform_mat(self, R: Tensor, t: Tensor) -> Tensor:
        ''' 
        Creates a batch of transformation matrices
        
        Parameters
        ----------
        R: torch.tensor (..., 3, 3)
            batch of rotation matrices
        t: torch.tensor (..., 3, 1), broadcastable with `R`
            batch of translation vectors
        
        Returns
        -------
        T: torch.tensor (..., 4, 4) 
            Transformation matrix
        '''
        R = F.pad(R, [0, 1, 0, 1])
        t = F.pad(F.pad(t, [0, 0, 0, 1], value=1), [3, 0, 0, 0])
        return R + t

    def forward(self, rotations, root_positions, device=None):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        
        Parameters
        ----------
        rotations: torch.tensor (..., J, 3)
            Tensor of local rotation sequence
        root_positions: torch.tensor (..., 3)
            Location of root joint
        
        Returns
        -------
        J_transformed: torch.tensor (..., J, 3)
            Location of all joints
        
        Normal_transformed: torch.tensor (..., 1, 3) or (..., 0, 3)
            Location of orginal normal vector (1.0, 0.0, 0.0)
        """
        if device is None:
            device = rotations.device
        else:
            rotations = rotations.to(device)
        joint_t_pose = torch.from_numpy(self.joint_t_pose).to(device)   # (J', 3)
        
        num_joint = rotations.shape[-2]
        if num_joint == 55: # SMPL-X (55, 3)
            joint_indices = list(range(55))
            parents = list(smplx_parents)
        elif num_joint == 52:   # SMPL-H (52, 3)
            joint_indices = list(smplh_from_smplx)
            parents = list(smplh_parents)
        elif num_joint == 24:   # SMPL (24, 3)
            joint_indices = list(smpl_from_smplx)
            parents = list(smpl_parents)
        else:
            raise NotImplementedError
        
        if joint_t_pose.shape[0] == 56: # add an additional normal vector
            joint_indices += [-1]
            parents += [0]
            rotations = torch.cat([rotations, torch.zeros(rotations.shape[:-2] + (1, 3), dtype=rotations.dtype, device=rotations.device)], dim=-2)
        J = joint_t_pose[joint_indices]
        
        local_q = rotations.float()
        root_pos = root_positions.to(device).float()
        
        rot_mats = self.batch_rodrigues(local_q)    # (..., J , 3, 3)
        
        J_transformed = self.batch_rigid_transform(rot_mats, J, parents, dtype=torch.float32)
        J_transformed += root_pos.unsqueeze(dim=-2)
        # J_transformed = J_transformed.detach().cpu().numpy()

        return J_transformed[..., :num_joint, :], J_transformed[..., num_joint:, :]

