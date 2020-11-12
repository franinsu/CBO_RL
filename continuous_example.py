import numpy as np
import torch
from RL_Optimization import *

def π(a, s):
    return torch.tensor([1 / 2.0, 1 / 2.0]) * torch.ones_like(s)

def r(s):
    return torch.sin(s) + 1

def get_parameters():
    return {
    "σ": 0.2,
    "ϵ": 2.0 * np.pi / 32.0,
    "s_0": torch.tensor([0.0]),
    "a_s": torch.tensor([-1.0, 1.0]),
    }
def sampler():
    params = get_parameters()
    σ, ϵ, a_s = [params[p] for p in ["σ", "ϵ", "a_s"]]
    return lambda s: sample_policy(s, π, σ, ϵ, a_s, 2)[0][-1]

def resample_save_policy(suffix="", l=int(1e6),l_long=int(1e7)):
    params = get_parameters()
    s_0, σ, ϵ, a_s = [params[p] for p in ["s_0", "σ", "ϵ", "a_s"]]
    S, A_idx = sample_policy(s_0, π, σ, ϵ, a_s,l,verbose=True)
    R = r(S)
    S_long, A_idx_long = sample_policy(s_0, π, σ, ϵ, a_s, l_long,verbose=True)
    R_long = r(S_long)
    torch.save(S_long, f"cache/S_long_{suffix}.pt")
    torch.save(A_idx_long, f"cache/A_idx_long_{suffix}.pt")
    torch.save(R_long, f"cache/R_long_{suffix}.pt")
    torch.save(S, f"cache/S_{suffix}.pt")
    torch.save(A_idx, f"cache/A_idx_{suffix}.pt")
    torch.save(R, f"cache/R_{suffix}.pt")
