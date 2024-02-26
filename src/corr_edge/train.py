import argparse

import torch 
import torch.nn.functional as F

from core import Trainer, EpochSavePlugin, LossLoggerPlugin
from dataset.dance_dataset import AISTPPDataset
from model.model import DanceDecoder
from model.diffusion import GaussianDiffusion
from vis import SMPLSkeleton
from model.adan import Adan

def parse_train_opt():
    arg = argparse.ArgumentParser()
    arg.add_argument("--exp-name", default="runs")
    arg.add_argument("--epoch", type=int, default=2000)
    arg.add_argument("--batch-size", type=int, default=64)
    arg.add_argument("--gradient-accumulation-step", type=int, default=1)
    arg.add_argument("--device", type=int, default=0)
    arg.add_argument("--log-console", action="store_true")
    
    arg.add_argument("--data-path", type=str, default="data/")
    arg.add_argument("--backup-path", type=str, default="data/dataset_backups/")
    arg.add_argument("--feature-type", type=str, default="jukebox")
    
    arg.add_argument("--learning-rate", type=float, default=4e-4)
    arg.add_argument("--weight-decay", type=float, default=0.02)
    
    arg.add_argument("--save-period", type=int, default=100)
    arg.add_argument("--log-period", type=int, default=5)
    return arg.parse_args()


"""
baseline train command:
python train.py --exp-name=edge-baseline --batch-size=512 --gradient-accumulation-step=2 --device={device} --log-console --data-path={data-path} --backup-path={backup-path} --feature-type=baseline

jukebox train command:
python train.py --exp-name=edge-jukebox --batch-size=512 --gradient-accumulation-step=2 --device={device} --log-console --data-path={data-path} --backup-path={backup-path} --feature-type=jukebox
"""

if __name__ == "__main__":
    opt = parse_train_opt()

    trainer = Trainer(
        exp_name=opt.exp_name,
        epoch=opt.epoch,
        batch_size=opt.batch_size,
        gradient_accumulation_step=opt.gradient_accumulation_step,
        device=torch.device(opt.device),
        plugin_debug=opt.log_console
    )
    
    train_dataset = AISTPPDataset(
        data_path=opt.data_path,
        backup_path=opt.backup_path,
        train=True,
        feature_type=opt.feature_type,
    )
    
    use_baseline_feats = opt.feature_type == "baseline"
    pos_dim = 3
    rot_dim = 24 * 6  # 24 joints, 6dof
    repr_dim = pos_dim + rot_dim + 4
    feature_dim = 35 if use_baseline_feats else 4800
    horizon_seconds = 5
    FPS = 30
    horizon = horizon_seconds * FPS
    
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
    smpl = SMPLSkeleton(trainer.device)
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
        return Adan(diffusion.model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)

    trainer.extend_plugins([
        EpochSavePlugin(period=opt.save_period),
        LossLoggerPlugin(period=opt.log_period)
    ]).train(
        dataset=train_dataset,
        network=diffusion,
        losses=losses,
        optim_fn=optim_fn
    )
    
    