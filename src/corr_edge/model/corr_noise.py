# ------------------------------------------ corr noise ------------------------------------------#
import torch 
import torch.nn as nn
from typing import Optional
from einops import rearrange

class CorrNoise(nn.Module):
    def __init__(self, size: int, dim: int, eps: float = 0.00001, momentum: float = 0.1, 
                device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, track_running_stats: bool = True):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.dim = dim
        self.size = size 
        self.eps = eps 
        self.momentum = momentum 
        self.track_running_stats = track_running_stats
        self.register_buffer('running_corr_matrices', 
                             torch.zeros(dim, size, size, **factory_kwargs) + torch.eye(n=size, **factory_kwargs))
        self.running_corr_matrices: torch.Tensor
        self.register_buffer('num_batches_tracked', 
                             torch.tensor(0, dtype=torch.long,
                                          **{k: v for k, v in factory_kwargs.items() if k != 'dtype'}))
        self.num_batches_tracked: torch.Tensor
    
    def forward(self, standard_noise: torch.Tensor, ref: Optional[torch.Tensor] = None):
        # standard_noise: [b, seqlen, dim] standard noise
        # ref: [b, seqlen, dim]
        cn_training: bool = False
        if self.training and ref is not None:
            cn_training = True 
        if cn_training:
            self.num_batches_tracked.add_(1)
            corr_matrices = self.compute_corr_matrices(ref)
            # correct diagonal ones
            correct_diag = (corr_matrices - 1.0) * torch.eye(self.size, dtype=corr_matrices.dtype, device=corr_matrices.device)
            corr_matrices -= correct_diag
            if self.track_running_stats:
                self.update_corr_matrices(corr_matrices)
        else:
            corr_matrices = self.running_corr_matrices
        return self.transform(corr_matrices, standard_noise)
    
    def compute_corr_matrices(self, x):
        x -= x.mean(dim=0, keepdim=True)
        x /= x.norm(dim=0, keepdim=True) + self.eps 
        x = rearrange(x, 'b l d -> d l b')
        return torch.matmul(x, x.mT)    # (dim, size, size)
    
    def update_corr_matrices(self, corr_matrices):
        if self.momentum is not None:
            exponential_avg_factor = self.momentum
        else:
            exponential_avg_factor = float(self.num_batches_tracked)
        # momentum update
        self.running_corr_matrices = exponential_avg_factor * corr_matrices + \
                (1.0 - exponential_avg_factor) * self.running_corr_matrices
    
    def transform(self, corr_matrices, standard_noise):
        is_half = corr_matrices.dtype == torch.half 
        if is_half:
            corr_matrices = corr_matrices.float()
        Ds, Qs = torch.linalg.eigh(corr_matrices)   # Ds: (dim, size); Qs: (dim, size, size)
        if is_half:
            Ds = Ds.half()
            Qs = Qs.half()
        Ds[Ds < 0.] = 0.
        Ds = rearrange(Ds, 'b l -> b 1 l') ** 0.5   # cause NaN in float16
        Qs = Ds * Qs
        noise = rearrange(standard_noise, 'b l d -> d l b')
        noise = torch.matmul(Qs, noise) # (d l b)
        return rearrange(noise, 'd l b -> b l d')