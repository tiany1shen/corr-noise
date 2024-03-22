import os
import pickle
import shutil
from pathlib import Path

from typing import Literal
import numpy as np

import torch
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix


def fileToList(f):
    out = open(f, "r").readlines()
    out = [x.strip() for x in out]
    out = [x for x in out if len(x)]
    return out 

def get_data_lists(split_dir, split_name):
    filter_list = set(fileToList(Path(split_dir, "ignore_list.txt")))
    data_list = set(fileToList(Path(split_dir, f"{split_name}_list.txt")))
    return data_list, filter_list

def get_pair_names_by_sequence(sequence_name, dataset):
    
    if dataset == "EDGE":
        return sequence_name, sequence_name
    
    if dataset == "FineDance":
        return sequence_name, sequence_name
    
    if dataset == "Aistpp":
        # Eg.: 'gBR_sBM_cAll_d04_mBR0_ch01' -> 'mBR0'
        return sequence_name, sequence_name.split("_")[4]

def split_data_generator(
    dataset_path, 
    dataset: Literal["EDGE", "FineDance", "Aistpp"], 
    split_name: Literal["train", "test"]
):
    # `dataset_path`: diretory containing the raw data folder.
    # train - test split
    # A generator yields tuples of (Sequence name, Motion file path, Wav file path)
    
    if dataset == "EDGE":
        raw_motion_path = Path(dataset_path, "edge_aistpp", "motions")
        raw_wav_path = Path(dataset_path, "edge_aistpp", "wavs")
        ext = ".pkl"
        split_dir = Path(dataset_path, "edge_aistpp", "splits")
        
    elif dataset == "FineDance":
        raw_motion_path = Path(dataset_path, "finedance", "motion")
        raw_wav_path = Path(dataset_path, "finedance", "music_wav")
        ext = ".npy"
        split_dir = Path(dataset_path, "finedance", "splits")
    
    elif dataset == "Aistpp":
        raw_motion_path = Path(dataset_path, "aistpp", "motions")
        raw_wav_path = Path(dataset_path, "aistpp", "wav")
        ext = ".pkl"
        split_dir = Path(dataset_path, "aistpp", "splits")
         
    else:
        raise ValueError(f"No dataset named '{dataset}' found! Choose one among ['EDGE', 'FineDance', 'Aistpp'].")
    data_list, filter_list = get_data_lists(split_dir, split_name)
    
    for sequence_name in sorted(data_list):
        if sequence_name in filter_list:
            continue 
        
        motion_name, wav_name = get_pair_names_by_sequence(sequence_name, dataset)
        motion_file = Path(raw_motion_path, motion_name + ext)
        wav_file = Path(raw_wav_path, wav_name + '.wav')
        
        if not (motion_file.is_file() and wav_file.is_file()):
            continue
        yield sequence_name, motion_file, wav_file


def preprocess_EDGE(motion_file):
    RAW_FPS = 60
    FPS = 30
    stride = int(RAW_FPS // FPS)
    with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)
    position = motion_data['smpl_trans']
    rotation_aa = motion_data['smpl_poses']
    scaling = motion_data['smpl_scaling'].item()
    
    return np.concatenate([position / scaling, rotation_aa], axis=-1)[::stride]  # Aist++ are under 60-fps, downsample to 30 fps


def ax_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    ax = matrix_to_axis_angle(mat)
    return ax

def preprocess_FineDance(motion_file):
    motion_data = np.load(motion_file)
    position = motion_data[:, :3]
    rotation_6d = torch.from_numpy(motion_data[:, 3:]).reshape(-1, 52, 6)
    rotation_aa = ax_from_6v(rotation_6d).reshape(-1, 52 * 3).numpy()
    
    return np.concatenate([position, rotation_aa], axis=-1)


def preprocess_Aistpp(motion_file):
    return preprocess_EDGE(motion_file)



def split_data_files(raw_data_path, dataset_name, split_name):
    """ 
    分割数据集并处理数据文件：
    - 动作文件: (length, 3+j*3) Numpy.ndarray, 3(position) + joint * 3(axis-angle)
    - 音频文件: 复制原始 wav 文件
    """
    data_path = Path(raw_data_path, split_name)
    print(f"Creating {dataset_name} {split_name} set in {data_path.absolute()}:")
    motion_path = data_path / "motion"
    wav_path = data_path / "wav"
    motion_path.mkdir(parents=True, exist_ok=True)
    wav_path.mkdir(parents=True, exist_ok=True)
    if dataset_name == "EDGE":
        preprocess_motion = preprocess_EDGE
    elif dataset_name == "FineDance":
        preprocess_motion = preprocess_FineDance
    elif dataset_name == "Aistpp":
        preprocess_motion = preprocess_Aistpp
    else:
        raise NotImplementedError
    
    for (seq_name, motion_file, wav_file) in split_data_generator(raw_data_path, dataset_name, split_name):
        
        new_motion_file = motion_path / (motion_file.stem + '.npy')
        print(f"Processing: {new_motion_file}")
        motion = preprocess_motion(motion_file)
        np.save(new_motion_file, motion)
        
        new_wav_file = wav_path / wav_file.name
        if not new_wav_file.exists():
            
            print(f"Copying   : {new_wav_file}")
            shutil.copy(wav_file, wav_path)







if __name__ == "__main__":
    split_data_files("/root/autodl-tmp/Aistpp/", "Aistpp", "test")