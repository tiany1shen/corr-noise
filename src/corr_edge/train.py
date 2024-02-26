import torch 
import torch.nn.functional as F


from core import Trainer, EpochSavePlugin, LossLoggerPlugin
from dataset.dance_dataset import AISTPPDataset
from model.model import DanceDecoder
from model.diffusion import GaussianDiffusion
from vis import SMPLSkeleton
from model.adan import Adan

#trainer
exp_name = "edge"
epoch = 2000
batch_size = 512
device = torch.device("cuda:1")
feature_type = "jukebox"

from pathlib import Path

#dataset
data_path = "../../data/"
backup_path = "../../data/dataset_backups/"

#network
use_baseline_feats = feature_type == "baseline"
pos_dim = 3
rot_dim = 24 * 6  # 24 joints, 6dof
repr_dim = pos_dim + rot_dim + 4
feature_dim = 35 if use_baseline_feats else 4800
horizon_seconds = 5
FPS = 30
horizon = horizon_seconds * FPS

#optim
learning_rate=4e-4
weight_decay=0.02

#modules
trainer = Trainer(
    exp_name=exp_name,
    epoch=epoch,
    batch_size=batch_size,
    device=device,
    plugin_debug=True
)

dataset = AISTPPDataset(
    data_path=data_path,
    backup_path=backup_path,
    train=True,
    feature_type="jukebox",
)

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
)

def network_call(diffusion, batch):
    x, cond, filename, wavnames = batch 
    total_loss, (loss, v_loss, fk_loss, foot_loss) = diffusion(
        x, cond, t_override=None
    )
    return loss, v_loss, fk_loss, foot_loss

losses = {
    "network_call": network_call,
    "losses": {
        "reconstruct": lambda out, batch: out[0],
        "velocity": lambda out, batch: out[1],
        "forward-kinetic": lambda out, batch: out[2],
        "foot-contact": lambda out, batch: out[3],
    }
}

def optim_fn(diffusion):
    return Adan(diffusion.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# train
trainer.extend_plugins([
    EpochSavePlugin(period=100),
    LossLoggerPlugin(period=5)
]).train(
    dataset=dataset,
    network=diffusion,
    losses=losses,
    optim_fn=optim_fn
)