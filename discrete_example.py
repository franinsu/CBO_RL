import numpy as np
import torch
from RL_Optimization import *


def π(a, s):
    return torch.tensor([0.5, 0.5]) + a * torch.sin(s)/5


def r(s):
    return torch.sin(s) + 1


def get_parameters():
    n_s = 32
    return {
        "s_0": torch.tensor([int(n_s/2)], dtype=np.int),
        "σ": 1.,
        "ϵ": 1.,
        "a_s": torch.tensor([-1.0, 1.0]),
        "n_a": 2,
        "n_s": n_s,
        "s_s": torch.arange(n_s)*2*np.pi/n_s,
        "x_ls": torch.arange(n_s, dtype=np.int)
    }


def sampler():
    params = get_parameters()
    σ, ϵ, a_s, s_s, n_s = [params[p] for p in ["σ", "ϵ", "a_s", "s_s", "n_s"]]
    return lambda s: sample_policy_discrete(s, π, σ, ϵ, a_s, 2, s_s, n_s)[0][-1]


def resample_save_policy(suffix="tabular", l=int(1e6), l_long=int(1e7)):
    params = get_parameters()
    s_0, σ, ϵ, a_s, s_s, n_s = [params[p]
                                for p in ["s_0", "σ", "ϵ", "a_s", "s_s", "n_s"]]
    S, A_idx = sample_policy_discrete(
        s_0, π, σ, ϵ, a_s, l, s_s, n_s, verbose=True)
    R = r(s_s[S])
    S_long, A_idx_long = sample_policy_discrete(
        s_0, π, σ, ϵ, a_s, l_long, s_s, n_s, verbose=True)
    R_long = r(s_s[S_long])
    torch.save(S_long, f"cache/S_long_{suffix}.pt")
    torch.save(A_idx_long, f"cache/A_idx_long_{suffix}.pt")
    torch.save(R_long, f"cache/R_long_{suffix}.pt")
    torch.save(S, f"cache/S_{suffix}.pt")
    torch.save(A_idx, f"cache/A_idx_{suffix}.pt")
    torch.save(R, f"cache/R_{suffix}.pt")
