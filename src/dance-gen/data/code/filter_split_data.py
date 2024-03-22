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

def split_data_generator(
    dataset_path, 
    dataset: Literal["EDGE", "FineDance"], 
    split_name: Literal["train", "test"]
):
    # `dataset_path`: diretory containing the raw data folder.
    # train - test split
    # A generator yields tuples of (Sequence name, Motion file path, Wav file path)
    if dataset == "EDGE":
        raw_motion_path = Path(dataset_path, "edge_aistpp/motions")
        raw_wav_path = Path(dataset_path, "edge_aistpp/wavs")
        ext = ".pkl"
        filter_list = set(fileToList(Path(dataset_path, "edge_aistpp/splits/ignore_list.txt")))
        data_list = set(fileToList(Path(dataset_path, f"edge_aistpp/splits/{split_name}_list.txt")))
        
    elif dataset == "FineDance":
        raw_motion_path = Path(dataset_path, "finedance/motion")
        raw_wav_path = Path(dataset_path, "finedance/music_wav")
        ext = ".npy"
        filter_list = set(fileToList(Path(dataset_path, "finedance/splits/ignore_list.txt")))
        data_list = set(fileToList(Path(dataset_path, f"finedance/splits/{split_name}_list.txt")))
    else:
        raise ValueError(f"No dataset named '{dataset}' found! Choose one among ['EDGE', 'FineDance'].")
    
    for sequence_name in sorted(data_list):
        if sequence_name in filter_list:
            continue 
        motion_file = Path(raw_motion_path, sequence_name + ext)
        wav_file = Path(raw_wav_path, sequence_name + ".wav")
        
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



def split_data_files(raw_data_path, dataset_name, split_name):
    """ 
    分割数据集并处理数据文件：
    - 动作文件: (length, 3+j*3) Numpy.ndarray, 3(position) + joint * 3(axis-angle)
    - 音频文件: 复制原始 wav 文件
    """
    data_path = raw_data_path / split_name
    print(f"Creating {dataset_name} {split_name} set in {data_path.absolute()}:")
    motion_path = data_path / "motion"
    wav_path = data_path / "wav"
    motion_path.mkdir(parents=True, exist_ok=True)
    wav_path.mkdir(parents=True, exist_ok=True)
    if dataset_name == "EDGE":
        preprocess_motion = preprocess_EDGE
    elif dataset_name == "FineDance":
        preprocess_motion = preprocess_FineDance
    else:
        raise NotImplementedError
    
    for (seq_name, motion_file, wav_file) in split_data_generator(raw_data_path, dataset_name, split_name):
        new_motion_file = motion_path / f"{seq_name}.npy"
        print(f"Processing: {new_motion_file}")
        motion = preprocess_motion(motion_file)
        np.save(new_motion_file, motion)
        
        new_wav_file = wav_path / f"{seq_name}.wav"
        print(f"Copying   : {new_wav_file}")
        shutil.copy(wav_file, wav_path)







if __name__ == "__main__":
    for i, (a, b, c) in enumerate(split_data_generator("/ssd/shenty/EDGE", "EDGE", "train")):
        print("name:", a)
        print("motion:", b)
        print("wav:", c)
        print()
        if i == 2: 
            break
    
    
    # motion_path = Path(dataset_path, split_name, "motions")
    # wav_path = Path(dataset_path, split_name, "wavs")
    # motion_path.mkdir(parents=True, exist_ok=True)
    # wav_path.mkdir(parents=True, exist_ok=True)
    
    
    # for split_list, split_name in zip([train_list, test_list], ["train", "test"]):
    #     new_motion_path = Path(dataset_path, f"{split_name}/motions").mkdir(parents=True, exist_ok=True)
    #     new_wav_path = Path(dataset_path, f"{split_name}/wavs").mkdir(parents=True, exist_ok=True)
        
    #     for sequence in split_list:
    #         if sequence in filter_list:
    #             continue
    #         motion = f"{dataset_path}/motions/{sequence}.pkl"
    #         wav = f"{dataset_path}/wavs/{sequence}.wav"
    #         assert os.path.isfile(motion)
    #         assert os.path.isfile(wav)
    #         motion_data = pickle.load(open(motion, "rb"))
    #         trans = motion_data["smpl_trans"]
    #         pose = motion_data["smpl_poses"]
    #         scale = motion_data["smpl_scaling"]
    #         out_data = {"pos": trans, "q": pose, "scale": scale}
    #         pickle.dump(out_data, open(f"{split_name}/motions/{sequence}.pkl", "wb"))
    #         shutil.copyfile(wav, f"{split_name}/wavs/{sequence}.wav")
