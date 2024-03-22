import argparse
import shutil
import numpy as np

from pathlib import Path

from filter_split_data import split_data_files
from audio_extraction.baseline_features import extract_folder
from slice_data_pair import slice_folder


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=['EDGE', 'FineDance', 'Aistpp'], help="Dataset name.")
    parser.add_argument(
        "--data-path", type=str, help="Diretory where the data folder lies.")
    parser.add_argument(
        "--clip-duration", type=int, default=5, help="Clip length in time, defualt as 5 seconds.")
    parser.add_argument(
        "--slice-stride", type=float, default=0.5, help="Stride ratio when slicing the sequence, default as 0.5, which is 2.5 seconds.")
    return parser.parse_args()


def check_data_available(dataset_name, raw_data_path):
    """ 确认解压后的数据目录存在。
    """
    if dataset_name == "EDGE":
        data_path = Path(raw_data_path, "edge_aistpp")
    elif dataset_name == "FineDance":
        data_path = Path(raw_data_path, "finedance")
    elif dataset_name == "Aistpp":
        data_path = Path(raw_data_path, "aistpp")
    else:
        raise NotImplementedError
    
    assert data_path.exists() and data_path.is_dir()


def create_dataset(opt):
    FPS = 30
    
    dataset_name = opt.dataset
    raw_data_path = Path(opt.data_path)
    check_data_available(dataset_name, raw_data_path)
    
    for split_name in ['train', 'test']:
        # 1. split raw data
        # 2. extract music features 
        # 3. slice sequences
        
        split_data_files(raw_data_path, dataset_name, split_name)
        extract_folder(Path(raw_data_path, split_name, "wav"), Path(raw_data_path, split_name, "music"))
        slice_folder(Path(raw_data_path, split_name), opt.clip_duration, FPS, opt.slice_stride)


if __name__ == "__main__":
    # extract_folder(
    #     Path("/root/autodl-tmp/Aistpp/test/wav"),
    #     Path("/root/autodl-tmp/Aistpp/test/music"),
    # )
    
    create_dataset(parse_opt())