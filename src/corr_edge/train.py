import argparse

import torch 
import torch.nn.functional as F

from core import Trainer, BasePlugin, EpochSavePlugin, LossLoggerPlugin, LoadTrainerStatePlugin
from dataset.dance_dataset import AISTPPDataset
from model.model import DanceDecoder
from model.diffusion import GaussianDiffusion
from vis import SMPLSkeleton
from model.adan import Adan

def parse_train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="runs")
    parser.add_argument("--total-epoch", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-step", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-console", action="store_true")
    
    parser.add_argument("--data-path", type=str, default="data/")
    parser.add_argument("--backup-path", type=str, default="data/dataset_backups/")
    parser.add_argument("--feature-type", type=str, choices=["baseline","jukebox"])
    
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    
    parser.add_argument("--save-period", type=int, default=100)
    parser.add_argument("--log-period", type=int, default=5)
    
    parser.add_argument("--load-checkpoint-dir", type=str, default="")
    parser.add_argument("--enable-ema", action="store_true")
    parser.add_argument("--new-ema", action="store_true")
    
    parser.add_argument("--use-corr-noise", action="store_true")
    return parser.parse_args()



if __name__ == "__main__":
    opt = parse_train_opt()
    remain_epoch = opt.total_epoch
    if opt.load_checkpoint_dir != "":
        remain_epoch = opt.total_epoch - int(opt.load_checkpoint_dir.split('-')[1])

    seed_suffix = ""
    if opt.seed is not None:
        seed_suffix = f"_seed_{opt.seed}"
    trainer = Trainer(
        exp_name=opt.exp_name + seed_suffix,
        epoch=remain_epoch,
        batch_size=opt.batch_size,
        gradient_accumulation_step=opt.gradient_accumulation_step,
        device=torch.device(opt.device),
        plugin_debug=opt.log_console,
        init_random_seed=opt.seed
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
        
        use_corr_noise=opt.use_corr_noise
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
    
    if opt.load_checkpoint_dir != "":
        checkpoint_path = f"outputs/checkpoints/{opt.exp_name}/{opt.load_checkpoint_dir}/"
        trainer.append_plugin(
            LoadTrainerStatePlugin(checkpoint_path)
        )
    
    if opt.enable_ema:
        import copy
        class EdgeEmaPlugin(BasePlugin):
            def __init__(self, ema_update_period: int = 1, new_ema: bool = True) -> None:
                hook_info = {
                    "loop_beg": {
                        "priority": 3,
                        "description": "(optional) Copy EMA shadow weights."
                    },
                    "step_end": {
                        "priority": 5,
                        "description": "Update EMA."
                    }
                }
                super().__init__(hook_info)
                self.update_period = ema_update_period
                self.new_ema = new_ema
            
            def is_enable_step(self):
                return self.trainer.step % self.update_period == 0
            
            def loop_beg_func(self, network, *args, **kwargs):
                if self.new_ema:
                    diffusion = network.unwrap_model
                    diffusion.master_model = copy.deepcopy(diffusion.model)
            
            def step_end_func(self, network, *args, **kwargs):
                diffusion = network.unwrap_model
                if self.is_enable_step():
                    diffusion.ema.update_model_average(
                        diffusion.master_model, diffusion.model
                    )
            
            @property
            def state(self): return {}
        
        trainer.append_plugin(EdgeEmaPlugin(new_ema=opt.new_ema))
        

    trainer.extend_plugins([
        EpochSavePlugin(period=opt.save_period),
        LossLoggerPlugin(period=opt.log_period)
    ]).train(
        dataset=train_dataset,
        network=diffusion,
        losses=losses,
        optim_fn=optim_fn
    )
    

"""
python train.py --exp-name=edge-jukebox --batch-size=512 --gradient-accumulation-step=2 --device=1 --log-console --data-path=../../data --backup-path=../../data/dataset_backups --feature-type=jukebox --load-checkpoint-dir epoch-600 --enable-ema --new-ema --save-period=50
"""