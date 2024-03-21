import os
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa as lr
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
import soundfile as sf
from matplotlib import cm

smpl_parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
smplh_parents =[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 
                20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 
                21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50]
smplx_parents =[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 
                15, 15, 15, 
                20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 
                21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53]

smpl_from_smplh = list(range(22)) + [25, 40]
smpl_from_smplx = list(range(22)) + [28, 43]
smplh_from_smplx = list(range(22)) + list(range(25, 55))


def set_line_data_3d(line, x):
    line.set_data(x[:, :2].T)
    line.set_3d_properties(x[:, 2])

def set_scatter_data_3d(scat, x, c):
    scat.set_offsets(x[:, :2])
    scat.set_3d_properties(x[:, 2], "z")
    scat.set_facecolors([c])

def get_axrange(poses):
    pose = poses[0]
    x_min = pose[:, 0].min()
    x_max = pose[:, 0].max()

    y_min = pose[:, 1].min()
    y_max = pose[:, 1].max()

    z_min = pose[:, 2].min()
    z_max = pose[:, 2].max()

    xdiff = x_max - x_min
    ydiff = y_max - y_min
    zdiff = z_max - z_min

    biggestdiff = max([xdiff, ydiff, zdiff])
    return biggestdiff


def plot_single_pose(num, poses, lines, ax, axrange, scat, contact, ske_parents, camera_view=(None, None)):
    pose = poses[num]
    static = contact[num]
    indices = [7, 8, 10, 11]

    for i, (point, idx) in enumerate(zip(scat, indices)):
        position = pose[idx : idx + 1]
        color = "r" if static[i] else "g"
        set_scatter_data_3d(point, position, color)

    for i, (p, line) in enumerate(zip(ske_parents, lines)):
        # don't plot root
        if i == 0:
            continue
        # stack to create a line
        data = np.stack((pose[i], pose[p]), axis=0)
        set_line_data_3d(line, data)

    if num == 0:
        if isinstance(axrange, int):
            axrange = (axrange, axrange, axrange)
        xcenter, ycenter, zcenter = 0, 0, 1.5
        stepx, stepy, stepz = axrange[0] / 2, axrange[1] / 2, axrange[2] / 2

        x_min, x_max = xcenter - stepx, xcenter + stepx
        y_min, y_max = ycenter - stepy, ycenter + stepy
        z_min, z_max = zcenter - stepz, zcenter + stepz

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # ax.axis("off")
        ax.view_init(*camera_view)


def skeleton_render(
    poses,  # (N, J, 3)
    name,
    out="renders",
    sound=False,
    stitch=False,
    contact=None,
    render=True,
    smpl_mode="smpl",       # 是否渲染双手
    camera_view=(30, 30),
):
    '''
    Render Human skeleton
    
    Parameters
    ----------
    poses: (N, J, 3) numpy.array
        Joint coordinates of a motion sequence
    name: str
        Stem name of rendered output.
    smpl_mode: str, Literal['smpl', 'smplh', 'smplx']
        Skeleton mode. 'smpl' only render 24 joints, 'smplh' will additionally 
        render hands, and 'smplx' will further render facial expression. Default: 'smpl'.
    camera_view: tuple(float, float)
        Camera view of the rendered gif, a tuple of degree. Default: (30, 30)
    '''
    if render:
        num_joint = poses.shape[-2]
        if smpl_mode == "smpl":
            ske_parents = smpl_parents
            assert num_joint in [24, 52, 55]
            if num_joint == 55:
                poses = poses[..., smpl_from_smplx, :]
            elif num_joint == 52:
                poses = poses[..., smpl_from_smplh, :]
            elif num_joint == 24:
                pass 
        elif smpl_mode == "smplh":
            ske_parents = smplh_parents
            assert num_joint in [52, 55]
            if num_joint == 55:
                poses = poses[..., smplh_from_smplx, :]
            elif num_joint == 52:
                pass 
        elif smpl_mode == "smplx":
            ske_parents = smplx_parents
            assert num_joint == 55
        else:
            raise NotImplementedError
        
        # generate the pose with FK
        Path(out).mkdir(parents=True, exist_ok=True)
        num_steps = poses.shape[0]      # 
        
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        
        point = np.array([0, 0, 1])
        normal = np.array([0, 0, 1])
        d = -point.dot(normal)
        xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 2), np.linspace(-1.5, 1.5, 2))
        # z = (-normal[0] * xx - normal[1] * yy - d) * 1.0 / normal[2]
        z = (-normal[0] * xx - normal[1] * yy) * 1.0 / normal[2]
        # plot the plane
        ax.plot_surface(xx, yy, z, zorder=-11, cmap=cm.twilight)
        # Create lines initially without data
        lines = [
            ax.plot([], [], [], zorder=10, linewidth=1.5)[0]
            for _ in ske_parents
        ]
        scat = [
            # ax.scatter([], [], [], zorder=10, s=0, cmap=ListedColormap(["r", "g", "b"]))
            ax.scatter([], [], [], zorder=10, s=0)
            for _ in range(4)
        ]
        axrange = 3

        # create contact labels
        feet = poses[:, (7, 8, 10, 11)]
        feetv = np.zeros(feet.shape[:2])
        feetv[:-1] = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)
        if contact is None:
            contact = feetv < 0.01
        else:
            contact = contact > 0.95

        # Creating the Animation object
        anim = animation.FuncAnimation(
            fig,
            plot_single_pose,
            num_steps,
            fargs=(poses, lines, ax, axrange, scat, contact, ske_parents, camera_view),
            interval=1000 // 30,
        )
    if sound:
        # make a temporary directory to save the intermediate gif in
        if render:
            temp_dir = TemporaryDirectory()
            gifname = os.path.join(temp_dir.name, f"tmp.gif")
            anim.save(gifname)

        # stitch wavs
        if stitch:
            assert type(name) == list  # must be a list of names to do stitching
            name_ = [os.path.splitext(x)[0] + ".wav" for x in name]
            audio, sr = lr.load(name_[0], sr=None)
            ll, half = len(audio), len(audio) // 2
            total_wav = np.zeros(ll + half * (len(name_) - 1))
            total_wav[:ll] = audio
            idx = ll
            for n_ in name_[1:]:
                audio, sr = lr.load(n_, sr=None)
                total_wav[idx : idx + half] = audio[half:]
                idx += half
            # save a dummy spliced audio
            audioname = f"{temp_dir.name}/tempsound.wav" if render else os.path.join(out, f'{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.wav')
            sf.write(audioname, total_wav, sr)
            outname = os.path.join(
                out,
                f'{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.mp4',
            )
        else:
            assert type(name) == str
            assert name != "", "Must provide an audio filename"
            audioname = name
            outname = os.path.join(
                out, f"{os.path.splitext(os.path.basename(name))[0]}.mp4"
            )
        if render:
            print(f"ffmpeg -loglevel error -stream_loop 0 -y -i {gifname} -i {audioname} -shortest -c:v libx264 -crf 26 -c:a aac -q:a 4 {outname}")
            out = os.system(
                f"/home/lrh/Documents/ffmpeg-6.0-amd64-static/ffmpeg -loglevel error -stream_loop 0 -y -i {gifname} -i {audioname} -shortest -c:v libx264 -crf 26 -c:a aac -q:a 4 {outname}"
            )
    else:
        if render:
            # actually save the gif
            path = Path(name)
            gifname = Path(out, path.stem + ".gif")
            anim.save(gifname, savefig_kwargs={"transparent": True, "facecolor": "none"},)
    plt.close()


if __name__ == "__main__":
    import argparse
    import tqdm
    import einops 
    import random
    from skeleton import SMPLX_Skeleton
    import os, sys

    sys.path.append("/scratch/shenty/corr-noise/src/dance-gen/")

    from dataset.dance_dataset import Sliced_BodyMotion_Music
    from dataset.quaternion import ax_from_6v
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--save-joint", action="store_true")
    parser.add_argument(
        "--render", action="store_true")
    parser.add_argument(
        "--num-to-render", type=int, default=-1)
    parser.add_argument(
        "--device", type=str, default="cpu")
    opt = parser.parse_args()
    
    
    print("Loading SMPL-X model...")
    skeleton_model = SMPLX_Skeleton()
    
    def render_joints(
        dataset: Sliced_BodyMotion_Music, 
    ):
        folder = dataset.sliced_data_path
        render_folder = folder.parent / f"{folder.stem}_render"
        joint_folder = folder.parent / f"{folder.stem}_joint"
        
        print(f"Rendering data from folder {folder}.")
        if opt.save_joint:
            print(f"Computed joint coordinates will be save to {joint_folder} as '.npy' files.")
        if opt.render:
            print(f"Rendered animations will be saved to {render_folder} as '.gif' files.")
        
        positions = dataset.positions
        rotation_6ds = dataset.motions
        
        rotations = ax_from_6v(einops.rearrange(rotation_6ds, '... (j c) -> ... j c', c=6))
        
        joints, _ = skeleton_model.forward(rotations, positions, device=opt.device)
        joints = joints.cpu().numpy()
        
        rendered_gifs = []
        total = len(dataset)
        
        for slice_name, joints in tqdm.tqdm(zip(dataset.names, joints)):
            name = slice_name[:-10]
            
            if opt.save_joint:
                slice_joint_folder = joint_folder / name
                slice_joint_folder.mkdir(parents=True, exist_ok=True)
                np.save(slice_joint_folder / f"{slice_name}.npy", joints)
            
            if opt.render:
                slice_render_folder = render_folder / name 
                # random render 
                if opt.num_to_render > 0:
                    if random.random() > 2 * opt.num_to_render / total:
                        continue
                    if len(rendered_gifs) == opt.num_to_render:
                        continue
                slice_render_folder.mkdir(parents=True, exist_ok=True)
                skeleton_render(joints, name=slice_name, out=slice_render_folder)
                rendered_gifs.append(slice_name)
        
        print("Rendered '.gif' files include:")
        for gif in rendered_gifs:
            print("  " + gif)

    
    render_joints(Sliced_BodyMotion_Music("/ssd/shenty/FineDance/", "test"))
    render_joints(Sliced_BodyMotion_Music("/ssd/shenty/FineDance/", "train"))
    render_joints(Sliced_BodyMotion_Music("/ssd/shenty/EDGE/", "test"))
    render_joints(Sliced_BodyMotion_Music("/ssd/shenty/EDGE/", "train"))