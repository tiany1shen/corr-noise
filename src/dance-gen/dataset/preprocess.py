import glob
import os
import re
from pathlib import Path

import torch

from .scaler import MinMaxScaler
import pickle


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == "" else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


class Normalizer:
    def __init__(self, data):
        flat = data.reshape(-1, data.shape[-1])     # bxt , 151
        self.scaler = MinMaxScaler((-1, 1), clip=True)
        self.scaler.fit(flat)

    def normalize(self, x):
        batch, seq, ch = x.shape
        x = x.reshape(-1, ch)
        return self.scaler.transform(x).reshape((batch, seq, ch))

    def unnormalize(self, x):
        batch, seq, ch = x.shape
        x = x.reshape(-1, ch)
        x = torch.clip(x, -1, 1)  # clip to force compatibility
        return self.scaler.inverse_transform(x).reshape((batch, seq, ch))
    
    
class My_Normalizer:
    def __init__(self, data):
        if isinstance(data, str):
            self.scaler = MinMaxScaler((-1, 1), clip=True)
            with open(data, 'rb') as f:
                normalizer_state_dict = pickle.load(f)
            # normalizer_state_dict = torch.load(data)
            self.scaler.scale_ = normalizer_state_dict["scale"]
            self.scaler.min_ = normalizer_state_dict["min"]
        else:
            flat = data.reshape(-1, data.shape[-1])     # bxt , 151
            self.scaler = MinMaxScaler((-1, 1), clip=True)
            self.scaler.fit(flat)

    def normalize(self, x):
        if len(x.shape) == 3:
            batch, seq, ch = x.shape
            x = x.reshape(-1, ch)
            return self.scaler.transform(x).reshape((batch, seq, ch))
        elif len(x.shape) == 2:
            batch, ch = x.shape
            return self.scaler.transform(x)
        else:
            raise("input error!")

    def unnormalize(self, x):
        if len(x.shape) == 3:
            batch, seq, ch = x.shape
            x = x.reshape(-1, ch)
            x = torch.clip(x, -1, 1)  # clip to force compatibility
            return self.scaler.inverse_transform(x).reshape((batch, seq, ch))
        elif len(x.shape) == 2:
             x = torch.clip(x, -1, 1)
             return self.scaler.inverse_transform(x)
        else:
            raise("input error!")


def vectorize_many(data):
    # given a list of batch x seqlen x joints? x channels, flatten all to batch x seqlen x -1, concatenate
    batch_size = data[0].shape[0]
    seq_len = data[0].shape[1]

    out = [x.reshape(batch_size, seq_len, -1).contiguous() for x in data]

    global_pose_vec_gt = torch.cat(out, dim=2)
    return global_pose_vec_gt


def standarize_3d_trajectory(root_positions):
    '''
    Translate a (batched) trajectory by making it starting from the origin (0, 0, 0)
    
    Parameters
    ----------
    root_positions: torch.Tensor (..., N, 3)
        (Batched) 3d trajectory
    
    Returns
    -------
    root_positions: torch.Tensor (..., N, 3)
        (Batched) standard 3d trajectory starting from (0, 0, 0)
    '''
    return root_positions - root_positions[..., :1, :]


def permute_xyz2zxy(vector3d):
    '''
    Permute a (batched) 3d vector's last dimension by (2, 0, 1)
    '''
    return vector3d[..., (2, 0, 1)]


def rot_matrix_to_XOZ(vector3d):
    '''
    Compute a (batched) rotation matrix that can rotate a given vector along axis Z+ 
    to fall into plane +XOZ.
    
    Parameters
    ----------
    vector3d: torch.Tensor (..., 3)
    
    Returns
    -------
    rot_mat: torch.Tensor (..., 3, 3)
    
    Examples
    --------
    >>> x = torch.tensor([3.0, 4.0, 2.0])
    >>> rot_matrix_to_XOZ(x)
    
    The output is:
    >>> torch.tensor([[ 0.6, 0.8, 0.0], 
                      [-0.8, 0.6, 0.0], 
                      [ 0.0, 0.0, 1.0]])
    
    since we can check: the result's Y entries are 0.
    >>> (torch.matmul(rot_matix_to_XOZ(x), x.unsqueeze(-1))[..., 1, :] == 0).all()
    >>> True
    
    the transforms do not change Z entries.
    >>> (torch.matmul(rot_matix_to_XOZ(x), x.unsqueeze(-1))[..., 2, :] ==  x.unsqueeze(-1)[..., 2, :]).all()
    >>> True 
    
    the vectors are rigid and has same length.
    >>> (torch.matmul(rot_matix_to_XOZ(x), x.unsqueeze(-1)).norm(dim=-2) == x.unsqueeze(-1).norm(dim=-2)).all()
    >>> True
    '''
    vector3d = vector3d.float()
    rot_mat = torch.zeros(vector3d.shape[:-1] + (3, 3), device=vector3d.device)
    
    normal_xoy = vector3d[..., :2] / vector3d[..., :2].norm(dim=-1, keepdim=True)
    rot_mat[..., 0, 0] += normal_xoy[..., 0]
    rot_mat[..., 0, 1] += normal_xoy[..., 1]
    rot_mat[..., 1, 0] += normal_xoy[..., 1] * -1
    rot_mat[..., 1, 1] += normal_xoy[..., 0]
    rot_mat[..., 2, 2] += 1.0
    
    return rot_mat


def compute_ground_height(motions):
    '''
    Given a (batched) motions, recognize the (batched) gound heights
    
    Parameters
    ----------
    motions: torch.Tensor (..., N, J, 3)
    
    Returns
    -------
    gounds: torch.Tensor (..., )
        the probably ground heights of the given motions
    '''
    lowest_zs = torch.topk(motions[..., 2], k=2, dim=-1, largest=False)[0].mean(dim=-1)
    length = lowest_zs.shape[-1]
    lowest_zs = torch.topk(lowest_zs, k= length//2, dim=-1, largest=False)[0].mean(dim=-1)
    return lowest_zs
    