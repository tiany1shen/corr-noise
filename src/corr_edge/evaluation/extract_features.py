import os
import glob
import pickle
import numpy as np
import tqdm

from .features.kinetic import extract_kinetic_features
from .features.manual import extract_manual_features 

from vis import SMPLSkeleton
import torch
from pytorch3d.transforms import (RotateAxisAngle, axis_angle_to_quaternion,
                                  quaternion_multiply,
                                  quaternion_to_axis_angle)

__doc__ = r"""
FID:
    从动作坐标中抽取特征，需要数据集特征和生成样本特征，根据特征做进一步计算
Dist/Div:
    从动作坐标中抽取特征，只需要生成样本特征，根据特征做进一步计算
BeatAlign:
    从动作坐标中抽取beat特征，从音频baseline特征中挑选beat特征

需要抽取的特征包括：
1. 数据集内特征：
    - 名称：{name}_slice{no. slice}.pkl
    - 原始数据中只包含 motion 数据（旋转角）
    - 根据 sliced_motion FK 得到 motion keyjoint3d
    - 根据 keyjoint3 抽取对应特征（只需抽取一次）
2. 生成的样本特征：
    - 名称：{name}_slice{no. slice}_{no. sample}.pkl
    - 生成的样本数据中保存了 motion keyjoint3d 数据 
    - 根据保存的 keyjoint3d 抽取对应特征 
3. 音频节拍特征：
    - 从 baseline feature 内部提前，名称：{name}_slice{no. slice}.npy
"""

def extract_motion_feature(joint3d):
    roott = joint3d[:1, :1, :] # the starting point, (1, 1, 3)
    joint3d -= roott 
    # calculate feature
    kinetic_feat = extract_kinetic_features(joint3d)
    manual_feat = extract_manual_features(joint3d)
    return kinetic_feat, manual_feat

def extract_and_save_motion_feature(joint3d, save_dir, seq_name):
    # joint3d: (seq-len, 24, 3) np.array
    # save_dir: existing diretory, containing 2 sub diretories: `kinetic/` and `manual/`
    # sequence_name
    # return: kinetic and manual features of a given joint3d motion sequence
    kinetic_feat, manual_feat = extract_motion_feature(joint3d)
    # save paths
    kinetic_path = os.path.join(save_dir, "kinetic", f"{seq_name}.npy")
    manual_path = os.path.join(save_dir, "manual", f"{seq_name}.npy")
    np.save(kinetic_path, kinetic_feat)
    np.save(manual_path, manual_feat)
    
    return kinetic_feat, manual_feat

def load_cached_motion_feature(feature_dir):
    # feature_dir: directory containing 2 sub diretories: `kinetic/` and `manual/` 
    # return: a list of features in the given diretory
    kinetic_features = []
    manual_features = []
    for file in glob.glob(os.path.join(feature_dir, "kinetic", "*.npy")):
        kinetic_features.append(np.load(file))
    for file in glob.glob(os.path.join(feature_dir, "manual", "*.npy")):
        manual_features.append(np.load(file))
    return kinetic_features, manual_features

def extract_aistpp_gt(aistpp_data_dir, save_dir, save_gt):

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(os.path.join(save_dir, "kinetic"))
        os.mkdir(os.path.join(save_dir, "manual"))
    
    smpl = SMPLSkeleton()
    kinetic_feats = []
    manual_feats = []
    for motion_path in tqdm.tqdm(glob.glob(os.path.join(aistpp_data_dir, "*.pkl"))):
        name = os.path.splitext(os.path.basename(motion_path))[0]
        
        with open(motion_path, "rb") as f:
            data = pickle.load(f)
        root_pos = torch.from_numpy(np.stack([data['pos']]))
        local_q = torch.from_numpy(np.stack([data['q']])).reshape(1, -1, 24, 3)
        
        # downsample
        root_pos = root_pos[:, ::2, :]
        local_q = local_q[:, ::2, :, :]
        
        # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
        root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
        root_q_quat = axis_angle_to_quaternion(root_q)
        rotation = torch.Tensor(
            [0.7071068, 0.7071068, 0, 0]
        )  # 90 degrees about the x axis
        root_q_quat = quaternion_multiply(rotation, root_q_quat)
        root_q = quaternion_to_axis_angle(root_q_quat)
        local_q[:, :, :1, :] = root_q

        # don't forget to rotate the root position too 😩
        pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        root_pos = pos_rotation.transform_points(
            root_pos
        )  # basically (y, z) -> (-z, y), expressed as a rotation for readability

        # do FK
        joint3d = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
        joint3d = joint3d.numpy().reshape(-1, 24, 3)
        
        if save_gt:
            kinetic_feat, manual_feat = extract_and_save_motion_feature(joint3d, save_dir, name)
        else:
            kinetic_feat, manual_feat = extract_motion_feature(joint3d)
        kinetic_feats.append(kinetic_feat)
        manual_feats.append(manual_feat)
    return kinetic_feats, manual_feats