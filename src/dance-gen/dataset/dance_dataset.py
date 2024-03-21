
import torch 
import einops
import numpy as np

from torch.utils.data import Dataset
from pathlib import Path
from typing import Literal, Sequence, Optional

from .quaternion import ax_to_6v
from visualize.skeleton import SMPLX_Skeleton

from .preprocess import standarize_3d_trajectory, permute_xyz2zxy, rot_matrix_to_XOZ, compute_ground_height
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


class Sliced_Motion_Music(Dataset):
    """ 
    Args:
        data_path (str): 
    """
    def __init__(
        self, data_path: str, 
        split: Literal['train', 'test'] = 'train',
        use_cached: bool = True,
        num_joint: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.sliced_data_path = Path(data_path, split, "slice")
        
        self.names: Sequence[str]
        self.positions: torch.Tensor
        self.motions: torch.Tensor
        self.contacts: torch.Tensor
        self.musics: torch.Tensor
        
        backup_data = self.sliced_data_path.parent / "backup_data.npz"
        if use_cached and self.check_cached_data_exist(backup_data):
            print(f"Loading cached data from {backup_data}...")
            data = np.load(backup_data)
            self.names = list(data['name'])
            self.positions = torch.from_numpy(data['position']).float()
            self.motions = torch.from_numpy(data['motion']).float()
            self.contacts = torch.from_numpy(data['contact']).float()
            self.musics = torch.from_numpy(data['music']).float()
        else:   # reload data from `slice/` folder
            print(f"Reloading data from {self.sliced_data_path}...")
            names, motions, musics = self.reload_data()
            positions, motions, contacts = self.preprocess_motion(motions)
            musics = np.array(musics)
            
            # print(motions.shape, musics.shape)
            
            np.savez(backup_data, 
                    name=names, 
                    position=positions,
                    motion=motions, 
                    contact=contacts,
                    music=musics)
            print(f"Loaded data has been cached to {backup_data}.")
            
            self.names = names
            self.positions = torch.from_numpy(positions).float()
            self.motions = torch.from_numpy(motions).float()
            self.contacts = torch.from_numpy(contacts).float()
            self.musics = torch.from_numpy(musics).float()
        
        if num_joint is not None:
            if num_joint * 6 > self.motions.shape[-1]:
                raise ValueError
            elif num_joint == 24 and self.motions.shape[-1] == 312:
                select_indices = tuple(range(22*6)) + tuple(range(25*6, 26*6)) + tuple(range(40*6, 41*6))
            elif num_joint == 24 and self.motions.shape[-1] == 144:
                select_indices = tuple(range(144))
            else:
                raise NotImplementedError
            self.motions = self.motions[..., select_indices]
        
        print(f"Dataset length: {len(self)} \n   Motion tensor: {tuple(self.motions.shape[1:])} | Positon tensor: {tuple(self.positions.shape[1:])} | Contact tensor: {tuple(self.contacts.shape[1:])} \n   Music tensor: {tuple(self.musics.shape[1:])}")
    
    def __len__(self):
        return len(self.names) 
    
    def __getitem__(self, index):
        data_dict = dict(
            name=self.names[index],
            position=self.positions[index],
            motion=self.motions[index],
            contact=self.contacts[index],
            music=self.musics[index],
        )
        return data_dict

    def check_cached_data_exist(self, backup_data):
        return backup_data.exists() and backup_data.is_file()
    
    def reload_data(self):
        names = []
        motions = []
        musics = []
        for npz in sorted(self.sliced_data_path.rglob("*.npz")):
            sliced_pair = np.load(npz)
            # print(sliced_pair['motion'].shape, sliced_pair['music'].shape)    # check
            names.append(npz.stem)
            motions.append(sliced_pair['motion'])
            musics.append(sliced_pair['music'])
        return names, motions, musics
    
    def preprocess_motion(self, motions: list[np.array]) -> list[np.array]:
        motions = torch.from_numpy(np.array(motions)).float() # (B, N, 3 + J * 3)
        root_pos = motions[..., :3]
        local_q = einops.rearrange(motions[..., 3:], '... (j c) -> ... j c', c=3)
        
        # translate trajectory to original start
        root_pos = standarize_3d_trajectory(root_pos)
        # permute to standard (X+ forward, Z+ up)
        root_pos = permute_xyz2zxy(root_pos)
        local_q = permute_xyz2zxy(local_q)
        
        skeleton = SMPLX_Skeleton(with_normal=True)
        joints, normal = skeleton.forward(local_q, root_pos)  # (B, N, J, 3)
        
        # rotate root start rotations to x+ forward
        rot_mat = rot_matrix_to_XOZ(normal[..., 0, 0, :])   # (B, 3, 3)
        root_rotations = matrix_to_axis_angle(torch.matmul(rot_mat.unsqueeze(-3), axis_angle_to_matrix(local_q[..., 0, :])))
        local_q[..., 0, :] = root_rotations
        
        # translate root_pos
        root_pos = torch.matmul(rot_mat.unsqueeze(-3), root_pos.unsqueeze(-1)).squeeze(-1)
        gound = compute_ground_height(joints)   # (B, )
        root_pos[..., :, 2] -= gound.unsqueeze(-1)
        
        # compute feet contact
        feet = joints[..., (7, 8, 10, 11), :]
        feetv = torch.zeros(feet.shape[:-1], device=feet.device)
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = feetv < 0.01
        
        rotation_6d = ax_to_6v(local_q).float()
        rotation_6d = einops.rearrange(rotation_6d, '... j c -> ... (j c)')
        
        return root_pos.cpu().numpy(), rotation_6d.cpu().numpy(), contacts.cpu().numpy()


class Sliced_BodyMotion_Music(Sliced_Motion_Music):
    def __init__(self, data_path: str, split: Literal['train'] | Literal['test'] = 'train') -> None:
        super().__init__(data_path, split, use_cached=True, num_joint=24)
