import numpy as np
from scipy.signal import argrelextrema

def extract_audio_beat(audio_feature):
    if audio_feature.shape[-1] == 35:
        return audio_feature[:, -1]

def extract_motion_beat(joints):
    """Calculate motion beats.
    Kwargs:
        joints: [nframes, njoints, 3]
    Returns:
        - peak_onhot: motion beats.
    """
    # Calculate velocity.
    velocity = np.zeros_like(joints, dtype=np.float32)
    velocity[1:] = joints[1:] - joints[:-1]
    velocity_norms = np.linalg.norm(velocity, axis=2)
    envelope = np.sum(velocity_norms, axis=1)  # (seq_len,)

    # Find local minima in velocity -- beats
    peak_idxs = argrelextrema(envelope, np.less, axis=0, order=10) # 10 for 60FPS
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1

    # # Second-derivative of the velocity shows the energy of the beats
    # peak_energy = np.gradient(np.gradient(envelope)) # (seq_len,)
    # # optimize peaks
    # peak_onehot[peak_energy<0.001] = 0
    return peak_onehot


from  scipy.ndimage import gaussian_filter as G

def extract_motion_beat_bailando(keypoints):
    keypoints = np.array(keypoints).reshape(-1, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats_idx = argrelextrema(kinetic_vel, np.less)
    motion_beats = np.zeros_like(kinetic_vel, dtype=bool)
    motion_beats[motion_beats_idx] = 1
    return motion_beats


def alignment_score(music_beats, motion_beats, sigma=3):
    """Calculate alignment score between music and motion."""
    if motion_beats.sum() == 0:
        return 0.0
    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]
    score_all = []
    for motion_beat_idx in motion_beat_idxs:
        # dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32) * 2 # calculate as 60 fps
        ind = np.argmin(dists)
        score = np.exp(- dists[ind]**2 / 2 / sigma**2)
        score_all.append(score)
    return sum(score_all) / len(score_all)

def calc_ba(joint3d, audio_feature):
    music_beats = extract_audio_beat(audio_feature)
    # motion_beats = extract_motion_beat(joint3d)
    motion_beats = extract_motion_beat_bailando(joint3d)
    return alignment_score(music_beats, motion_beats)


if __name__ == "__main__":
    import argparse 
    import os 
    import glob
    import pickle
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-dir", type=str)
    parser.add_argument("--audio-feat-dir", type=str, default="/ssd/shenty/EDGE_Aist_Plusplus/test/baseline_feats/")
    opt = parser.parse_args()
    
    ba_scores = []
    for pkl in glob.glob(os.path.join(opt.motion_dir, "*.pkl")):
        name = os.path.splitext(os.path.basename(pkl))[0]
        audio_npy = "_".join(name.split("_")[:7])
        audio_npy = os.path.join(opt.audio_feat_dir, audio_npy + ".npy")
        assert os.path.isfile(audio_npy)
        audio_feature = np.load(audio_npy)
        joint3d = pickle.load(open(pkl, "rb"))['full_pose']
        ba_scores.append(calc_ba(joint3d=joint3d, audio_feature=audio_feature))
    print("BAS:", sum(ba_scores) / len(ba_scores))
    