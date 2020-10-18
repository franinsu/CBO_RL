# %%
import torch
from helper import *
from IPython import get_ipython
import torch.nn as nn
import pandas as pd
import seaborn as sns

# %%


def π(a, s):
    return torch.tensor([0.5, 0.5]) + a * torch.sin(s)/5


def r(s):
    return torch.sin(s) + 1


σ = 1.
ϵ = 1.
a_s = torch.tensor([-1.0, 1.0])
n_a = len(a_s)
a_2_idx = {x: i for i, x in enumerate(a_s)}
n_s = 32
s_s = torch.arange(n_s)*2*np.pi/n_s
s_0 = torch.tensor([int(n_s/2)], dtype=int)
#%%
# %%
# S, A_idx = sample_policy_discrete(s_0, π, σ, ϵ, a_s, int(1e6), s_s, n_s)
# R = r(s_s[S])
# S_long, A_idx_long = sample_policy_discrete(s_0, π, σ, ϵ, a_s, int(1e7), s_s, n_s)
# R_long = r(s_s[S_long])

# %%
# torch.save(S_long, 'cache/S_long_tabular.pt')
# torch.save(A_idx_long, 'cache/A_idx_long_tabular.pt')
# torch.save(R_long, 'cache/R_long_tabular.pt')
# torch.save(S, 'cache/S_tabular.pt')
# torch.save(A_idx, 'cache/A_idx_tabular.pt')
# torch.save(R, 'cache/R_tabular.pt')


# %%
S_long = torch.load("cache/S_long_tabular.pt")
R_long = torch.load("cache/R_long_tabular.pt")
A_idx_long = torch.load("cache/A_idx_long_tabular.pt")
S = torch.load("cache/S_tabular.pt")
R = torch.load("cache/R_tabular.pt")
A_idx = torch.load("cache/A_idx_tabular.pt")

# %%

def sample(s): return sample_policy_discrete(
    s, π, σ, ϵ, a_s, 2, s_s, n_s)[0][-1]

x_ls = torch.arange(n_s, dtype=np.int)
common_args = {"new_Q_net": lambda: Q_Tabular(
    n_a, n_s), "epochs": 5, "x_ls": x_ls}
sgd_args = {"M": 1000, "τ_k": lambda k: 0.2*0.999**k}
cbo_args = {"N": 30, "m": 1000, "γ": 0.9, "δ": 1e-3,
            "τ_k": lambda k: 0.2*0.999**k, "η_k": lambda k: 0.2, "β_k": lambda k: 10*1.001**k}
# %%
# Q_ctrl_UR_SGD_star = Q_SGD_gen(Q_ctrl_UR_SGD_update_step)(
#     S_long, A_idx_long, R_long, a_s, π, sample, **common_args, **sgd_args
# )
# %%
# torch.save(Q_ctrl_UR_SGD_star, "cache/Q_ctrl_UR_SGD_star_tabular.pt")
Q_ctrl_UR_SGD_star = torch.load("cache/Q_ctrl_UR_SGD_star_tabular.pt")
# %%
Q_ctrl_UR_SGD, e_ctrl_UR_SGD = Q_SGD_gen(Q_ctrl_UR_SGD_update_step)(
    S, A_idx, R, a_s, π, sample, **common_args, **sgd_args,  Q_net_comp=Q_ctrl_UR_SGD_star
)
Q_ctrl_DS_SGD, e_ctrl_DS_SGD = Q_SGD_gen(Q_ctrl_DS_SGD_update_step)(
    S, A_idx, R, a_s, π, sample, **common_args, **sgd_args,  Q_net_comp=Q_ctrl_UR_SGD_star
)
Q_ctrl_BFF_SGD, e_ctrl_BFF_SGD = Q_SGD_gen(Q_ctrl_BFF_SGD_update_step)(
    S, A_idx, R, a_s, π, sample, **common_args, **sgd_args,  Q_net_comp=Q_ctrl_UR_SGD_star
)
# %%
Q_ctrl_UR_CBO, e_ctrl_UR_CBO = Q_CBO_gen(Q_ctrl_UR_CBO_L)(
    S, A_idx, R, a_s, π, sample, **common_args, **cbo_args,  Q_net_comp=Q_ctrl_UR_SGD_star
)
Q_ctrl_DS_CBO, e_ctrl_DS_CBO = Q_CBO_gen(Q_ctrl_DS_CBO_L)(
    S, A_idx, R, a_s, π, sample, **common_args, **cbo_args, Q_net_comp=Q_ctrl_UR_SGD_star
)
Q_ctrl_BFF_CBO, e_ctrl_BFF_CBO = Q_CBO_gen(Q_ctrl_BFF_CBO_L)(
    S, A_idx, R, a_s, π, sample, **common_args, **cbo_args, Q_net_comp=Q_ctrl_UR_SGD_star
)
# %%
Q_dict = {
    "SGD": [
        [Q_ctrl_UR_SGD, Q_ctrl_DS_SGD, Q_ctrl_BFF_SGD],
        [e_ctrl_UR_SGD, e_ctrl_DS_SGD, e_ctrl_BFF_SGD],
        ["UR", "DS", "BFF"],
        ["C0", "C1", "C2"],
        ["solid", "solid"]
    ],
    "BFF": [
        [Q_ctrl_UR_CBO, Q_ctrl_DS_CBO, Q_ctrl_BFF_CBO],
        [e_ctrl_UR_CBO, e_ctrl_DS_CBO, e_ctrl_BFF_CBO],
        ["UR", "DS", "BFF"],
        ["C0", "C1", "C2"],
        ["solid", "solid"]
    ]
}
plotQ2(Q_dict, Q_ctrl_UR_SGD_star, "UR * SGD", a_s, x_s=x_ls)
plt.savefig("figs/Q_ctrl_discrete_.png")
# %%

# %%
