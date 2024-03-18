import argparse
import os 
import glob
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import random
import pickle 

from torch.utils.data import DataLoader, Dataset
from dataset.dance_dataset import AISTPPDataset
from model.model import DanceDecoder
from model.diffusion import GaussianDiffusion
from vis import SMPLSkeleton

from evaluation.extract_features import extract_aistpp_gt, load_cached_motion_feature
from evaluation.evaluate import evaluate_metrics

import warnings
 
warnings.filterwarnings("ignore")

def parse_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    
    # generate
    parser.add_argument("--feature-type", type=str, choices=["baseline", "jukebox"])
    parser.add_argument("--pose-type", type=str, choices=['rot6d', 'rot3d', 'joint3d'], default='rot6d')
    parser.add_argument("--normalizer-type", type=str, choices=["minmax", "standard"], default="minmax")
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument("--normal-noise", action="store_true")
    noise.add_argument("--corr-noise", action="store_true")
    noise.add_argument("--affine-noise", action="store_true")
    parser.add_argument("--guidance-weight", type=int, default=2)
    parser.add_argument("--state-dict-file", type=str)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--data-path", type=str, default="data/")
    parser.add_argument("--backup-path", type=str, default="dataset_backups/")
    # sample
    parser.add_argument("--use-default-testset", action="store_true")
    parser.add_argument("--wav-dirs", nargs="+")
    parser.add_argument("--feature-dirs", nargs="+")
    parser.add_argument("--num-sample", type=int, default=1000)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    # render
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-dir", type=str, default="outputs/renders/")
    parser.add_argument("--save-motion", action="store_true")
    parser.add_argument("--motion-save-dir", type=str, default="outputs/motions/")
    parser.add_argument("--sound", action="store_true")
    # evaluate
    parser.add_argument("--forced-re-extract-gt", action="store_true")
    parser.add_argument("--gt-motion-dir", type=str, default="test/motions_sliced/")
    parser.add_argument("--save-gt-feature", action="store_true")
    parser.add_argument("--gt-feature-dir", type=str, default="test/motions_sliced_feats/")
    parser.add_argument("--audio-feature-dir", type=str, default="test/baseline_feats/")
    
    return parser.parse_args()


class MusicFeatureDataset(Dataset):
    def __init__(self, 
        wav_dirs: list,
        feature_dirs: list,
        length: int
        ):
        super().__init__()
        assert len(wav_dirs) == len(feature_dirs)
        self.wavs = []
        self.features = []
        for wav_dir, feat_dir in zip(wav_dirs, feature_dirs):
            wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
            feats = sorted(glob.glob(os.path.join(feat_dir, "*.npy")))
            for wav_file, feat_file in zip(wavs, feats):
                wav_name = os.path.splitext(os.path.basename(wav_file))[0]
                feat_name = os.path.splitext(os.path.basename(feat_file))[0]
                assert wav_name == feat_name
                self.wavs.append(wav_file)
                self.features.append(feat_file)
        self.length = length
        if length < len(self.wavs):
            self.indices = random.sample(range(len(self.wavs)), length)
        else:
            mult = length // len(self.wavs)
            remainder = length - mult * len(self.wavs)
            self.indices = [i for i in range(len(self.wavs))] * mult + random.sample(range(len(self.wavs)), remainder)
        
    def __len__(self): return self.length
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        wav_file, feat_file = self.wavs[idx], self.features[idx]
        feature = torch.from_numpy(np.load(feat_file))
        return 0, feature, 0, wav_file

def add_suffix_idx(file_names, start_idx, size):
    """ 
    [file_a.wav, file_b.wav, file_c.wav, ...] --> [file_a_0.wav, file_b_1.wav, file_c_2.wav, ...]
    """
    new_list = []
    assert len(file_names) == size 
    for i, file_name in enumerate(file_names):
        name, ext = os.path.splitext(file_name)
        new_list.append(name + f"_{start_idx+i}" + ext)
    return new_list
        


def sample(opt):
    
    use_baseline_feats = opt.feature_type == "baseline"
    pos_dim = 3
    if opt.pose_type == 'rot6d':
        rot_dim = 24 * 6  # 24 joints, 6dof
    elif opt.pose_type in ['rot3d', 'joint3d']:
        rot_dim = 24 * 3
    else:
        raise NotImplementedError
    repr_dim = 4 + pos_dim + rot_dim # [contact, pos, rot]
    feature_dim = 35 if use_baseline_feats else 4800
    horizon_seconds = 5
    FPS = 30
    horizon = horizon_seconds * FPS
    
    clip_denoised = (opt.normalizer_type == "minmax")
    
    noise_schedule = None
    if opt.normal_noise:
        noise_schedule = "normal_noise"
    elif opt.corr_noise:
        noise_schedule = "corr_noise"
    elif opt.affine_noise:
        noise_schedule = "affine_noise"
    else:
        raise NotImplementedError
    
    
    
    device = torch.device(opt.device)
    
    noise_schedule = None
    if opt.corr_noise:
        noise_schedule = "corr_noise"
    if opt.affine_noise:
        noise_schedule = "affine_noise"
    
    model = DanceDecoder(
        nfeats=repr_dim,
        seq_len=horizon,
        latent_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        cond_feature_dim=feature_dim,
        activation=F.gelu,
    )
    smpl = SMPLSkeleton(device)
    diffusion = GaussianDiffusion(
        model,
        horizon,
        repr_dim,
        smpl,
        schedule="cosine",
        n_timestep=1000,
        predict_epsilon=False,
        loss_type="l2",
        clip_denoised=clip_denoised,
        use_p2=False,
        cond_drop_prob=0.25,
        guidance_weight=opt.guidance_weight,
        
        pose_type=opt.pose_type,
        noise_schedule=noise_schedule,
    )
    diffusion.load_state_dict(torch.load(opt.state_dict_file, map_location=device))
    diffusion.model = diffusion.master_model
    diffusion.to(device)
    diffusion.eval()
    
    data_path = opt.data_path
    backup_path = os.path.join(opt.data_path, opt.backup_path)
    
    normalizer = AISTPPDataset(
        data_path=data_path,
        backup_path=backup_path,
        train=True,
        feature_type=opt.feature_type,
        pose_type=opt.pose_type,
    ).normalizer
    
    if opt.use_default_testset:
        test_dataset = AISTPPDataset(
            data_path=data_path,
            backup_path=backup_path,
            train=False,
            feature_type=opt.feature_type,
            pose_type=opt.pose_type,
            normalizer=normalizer,
        )
    else:
        test_dataset = MusicFeatureDataset(
            wav_dirs = [os.path.join(opt.data_path, x) for x in opt.wav_dirs],
            feature_dirs= [os.path.join(opt.data_path, x) for x in opt.feature_dirs],
            length=opt.num_sample
        )
    test_dataloader = DataLoader(test_dataset, batch_size=opt.sample_batch_size)
    
    render_out = os.path.join(opt.render_dir, opt.exp_name)
    fk_out = None
    if opt.save_motion:
        fk_out = os.path.join(opt.motion_save_dir, opt.exp_name)
    start_idx = 0
    batch_size = test_dataloader.batch_size
    for _, cond, filename, wavname in test_dataloader:
        cond = cond.to(device)
        wavname = add_suffix_idx(wavname, start_idx, len(cond))
        start_idx += batch_size
        
        diffusion.render_sample(
            shape=(cond.shape[0], horizon, repr_dim),
            cond=cond,
            normalizer=normalizer,
            epoch="",
            render_out=render_out,
            fk_out=fk_out,
            name=wavname,
            sound=opt.sound,
            mode="normal",
            render=opt.render
        )


def evaluate(opt):
    # extract gt features if needed
    gt_feature_dir = os.path.join(opt.data_path, opt.gt_feature_dir)
    forced_extract_gt = opt.forced_re_extract_gt or not os.path.exists(gt_feature_dir)
    if forced_extract_gt:
        print(f"Extracting Aist++ ground truth features")
        if opt.save_gt_feature:
            print(f"Saving extracted features to {gt_feature_dir}")
        gt_motion_dir = os.path.join(opt.data_path, opt.gt_motion_dir)
        gt_features = extract_aistpp_gt(gt_motion_dir, gt_feature_dir, opt.save_gt_feature)
    else:
        print(f"Using pre-computed ground truth features from {gt_feature_dir}")
        gt_features = load_cached_motion_feature(gt_feature_dir)
    
    sample_joint3d_dir = os.path.join(opt.motion_save_dir, opt.exp_name)
    audio_feature_dir = os.path.join(opt.data_path, opt.audio_feature_dir)
    evaluate_metrics(gt_features, sample_joint3d_dir, audio_feature_dir)
    
if __name__ == '__main__':
    opt = parse_test_opt()
    if opt.sample:
        sample(opt)
    if opt.evaluate:
        evaluate(opt)