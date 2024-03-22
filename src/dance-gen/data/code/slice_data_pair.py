import glob
import numpy as np
from pathlib import Path 


def motion_music_pair_generator(data_dir):
    motion_path = Path(data_dir, "motion")
    music_path = Path(data_dir, "music")
    
    motion_files = sorted(motion_path.glob("*.npy"))
    music_files = sorted(music_path.glob("*.npy"))
    for motion_file, music_file in zip(motion_files, music_files):
        motion_name = Path(motion_file).stem 
        music_name = Path(music_file).stem 
        assert motion_name == music_name 
        yield motion_name, motion_file, music_file

def slice_folder(data_dir, clip_duration, fps = 30, stride=0.5):
    clip_length = int(clip_duration * fps)
    stride_length = int(clip_length * stride)
    slice_ignored = Path(data_dir, "slice_discard")
    slice_ignored.mkdir(exist_ok=True, parents=True)
    
    for seq_name, motion_file, music_file in motion_music_pair_generator(data_dir):
        slice_path = Path(data_dir, "slice", seq_name)
        print(f"Slicing   : {slice_path.absolute()}")
        slice_path.mkdir(parents=True, exist_ok=True)
        motion = np.load(motion_file)
        music = np.load(music_file)
        
        length = min(motion.shape[0], music.shape[0])
        assert length >= clip_length
        num_clip = 1 + (length - clip_length) // stride_length
        
        start_t = 0
        for i in range(num_clip):
            slice_file = slice_path / f"{seq_name}_slice_{str(i).zfill(3)}.npz"
            motion_aa_clip = motion[start_t: start_t + clip_length]
            music_feat_clip = music[start_t: start_t + clip_length]
            if motion_aa_clip.std(axis=0).mean() > 0.07:    # static motion
                print(f"Saving    : {slice_file}")
                np.savez(slice_file, 
                        motion=motion_aa_clip,
                        music=music_feat_clip)
            else:
                print(f"Skipping  : {slice_file}")
                np.savez(slice_ignored / f"{seq_name}_slice_{str(i).zfill(3)}.npz",
                        motion=motion_aa_clip,
                        music=music_feat_clip)
            start_t += stride_length


if __name__ == "__main__":
    slice_folder(
        data_dir="/ssd/shenty/EDGE/test/",
        clip_duration=5,
    )