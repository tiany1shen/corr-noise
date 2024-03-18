import glob 
import os
import pickle
import tqdm

import numpy as np
from .metrics.fid import calc_fid 
from .metrics.dist import calc_dist
from .metrics.pfc import calc_pfc
from .metrics.beat_alignment import calc_ba
from .features.kinetic import extract_kinetic_features
from .features.manual import extract_manual_features 

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)

def evaluate_metrics(gt_feats, sample_joint3d_dir, audio_feature_dir):
    
    gt_kinetic_feats, gt_manual_feats = gt_feats
    gt_kinetic_feats = np.stack(gt_kinetic_feats)
    gt_manual_feats = np.stack(gt_manual_feats)
    
    pfc = []
    ba = []
    sample_kinetic_feats = []
    sample_manual_feats = []
    for motion_path in tqdm.tqdm(glob.glob(os.path.join(sample_joint3d_dir, "*.pkl"))):
        name = os.path.splitext(os.path.basename(motion_path))[0]
        name = "_".join(name.split("_")[:7])
        audio_path = os.path.join(audio_feature_dir, f"{name}.npy") 
        assert os.path.isfile(audio_path)
        
        with open(motion_path, "rb") as f:
            joint3d = pickle.load(f)['full_pose']
        roott = joint3d[:1, :1, :] # the starting point, (1, 1, 3)
        joint3d -= roott 
        audio_feat = np.load(audio_path)
        
        pfc.append(calc_pfc(joint3d))
        ba.append(calc_ba(joint3d, audio_feat))
        sample_kinetic_feats.append(extract_kinetic_features(joint3d))
        sample_manual_feats.append(extract_manual_features(joint3d))
    
    sample_kinetic_feats = np.stack(sample_kinetic_feats)
    sample_manual_feats = np.stack(sample_manual_feats)
    
    # normalize 
    gt_kinetic_feats, sample_kinetic_feats = normalize(gt_kinetic_feats, sample_kinetic_feats)
    gt_manual_feats, sample_manual_feats = normalize(gt_manual_feats, sample_manual_feats)
    
    fid_k = calc_fid(sample_kinetic_feats, gt_kinetic_feats)
    fid_g = calc_fid(sample_manual_feats, gt_manual_feats)
    
    dist_k = calc_dist(sample_kinetic_feats)
    dist_g = calc_dist(sample_manual_feats)
    
    print(f"Generated motions from: {sample_joint3d_dir}.")
    print("Calculated Metrics:")
    print(f"  FID_k: {fid_k:4e}\tFID_g: {fid_g:4e}")
    print(f"  DIST_k: {dist_k:4e}\tDIST_g: {dist_g:4e}")
    print(f"  PFC:\t{sum(pfc)/len(pfc)}")
    print(f"  BA:\t{sum(ba)/len(ba)}")
        
        
        