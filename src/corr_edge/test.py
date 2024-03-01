import argparse
import os 
import glob
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import random
import pickle 

from torch.utils.data import DataLoader
from dataset.dance_dataset import AISTPPDataset
from model.model import DanceDecoder
from model.diffusion import GaussianDiffusion
from vis import SMPLSkeleton


import warnings
 
warnings.filterwarnings("ignore")

def parse_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str)
    task = parser.add_mutually_exclusive_group()
    task.add_argument("--sample", action="store_true")
    task.add_argument("--evaluate", action="store_true")
    
    # generate
    parser.add_argument("--feature-type", type=str, choices=["baseline", "jukebox"])
    parser.add_argument("--use-corr-noise", action="store_true")
    parser.add_argument("--state-dict-file", type=str)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--normalizer-file", type=str, default="data/dataset_backups/normalizer.pkl")
    parser.add_argument("--data-path", type=str, default="data/")
    parser.add_argument("--backup-path", type=str, default="data/dataset_backups/")
    
    parser.add_argument("--sample-batch-size", type=int, default=64)
    
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-dir", type=str, default="outputs/renders/")
    parser.add_argument("--save-motion", action="store_true")
    parser.add_argument("--save-dir", type=str, default="outputs/motions/")
    parser.add_argument("--sound", action="store_true")
    
    return parser.parse_args()

def sample(opt):
    use_baseline_feats = opt.feature_type == "baseline"
    pos_dim = 3
    rot_dim = 24 * 6  # 24 joints, 6dof
    repr_dim = pos_dim + rot_dim + 4
    feature_dim = 35 if use_baseline_feats else 4800
    horizon_seconds = 5
    FPS = 30
    horizon = horizon_seconds * FPS
    device = torch.device(opt.device)
    
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
        use_p2=False,
        cond_drop_prob=0.25,
        guidance_weight=2,
        
        use_corr_noise=opt.use_corr_noise
    )
    diffusion.load_state_dict(torch.load(opt.state_dict_file, map_location=device))
    diffusion.model = diffusion.master_model
    diffusion.to(device)
    diffusion.eval()
    
    with open(opt.normalizer_file, "rb") as f:
        normalizer = pickle.load(f)
    
    test_dataset = AISTPPDataset(
        data_path=opt.data_path,
        backup_path=opt.backup_path,
        train=False,
        feature_type=opt.feature_type,
        normalizer=normalizer,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=opt.sample_batch_size)
    
    render_out = os.path.join(opt.render_dir, opt.exp_name)
    fk_out = None
    if opt.save_motion:
        fk_out = os.path.join(opt.save_dir, opt.exp_name)
    
    for _, cond, filename, wavname in test_dataloader:
        cond = cond.to(device)
        diffusion.render_sample(
            shape=(cond.shape[0], horizon, repr_dim),
            cond=cond,
            normalizer=test_dataset.normalizer,
            epoch="",
            render_out=render_out,
            fk_out=fk_out,
            name=wavname,
            sound=opt.sound,
            mode="normal",
            render=opt.render
        )
        
if __name__ == '__main__':
    opt = parse_test_opt()
    if opt.sample:
        sample(opt)