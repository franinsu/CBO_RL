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
a_2_idx = {x:i for i,x in enumerate(a_s)}
n_s = 32
s_s = torch.arange(n_s)*2*np.pi/n_s
s_0 = torch.tensor([int(n_s/2)], dtype=int)
#%%
def sample_policy_discrete(s_0, π, σ, ϵ, a_s, m, s_s, n_s, verbose=False):
    # S returned will represent indices
    s_0 = s_0.view(s_0.size()[0], -1)
    s_0_size = s_0.size()
    S = torch.zeros(m, *s_0_size, dtype=int)
    S[0] = s_0
    A_idx = torch.zeros(m, s_0_size[0])
    z = torch.zeros(s_0_size, dtype=float)
    if verbose:
        i_range = trange(m-1,leave=False, position=0, desc="Epoch")
    else:
        i_range = range(m-1)
    for i in i_range:
        for j, s in enumerate(S[i]):
            A_idx[i, j] = Categorical(probs=π(a_s, s_s[s])).sample()
            a = a_s[int(A_idx[i, j].item())]
            s̃ = s_s[s]+(2*np.pi/n_s)*a*ϵ+σ*np.sqrt(ϵ)*torch.normal(mean=z)
            S[i+1, j] = torch.argmin(torch.abs(s̃ - s_s))
    return S.squeeze(2), A_idx
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

#%%
class Q_Tabular(nn.Module):
    def __init__(self, n_a, n_s):
        super(Q_Tabular, self).__init__()
        self.Q_matrix = torch.normal(mean=torch.zeros(n_s,n_a))
        self.Q_matrix.requires_grad = True
        self.n_s = n_s
    def forward(self, s):
        return self.Q_matrix[torch.remainder(torch.tensor(s),self.n_s),:].squeeze()

    def parameters(self):
        return [self.Q_matrix]
    def zero_grad(self):
        if self.Q_matrix.grad!=None:
            self.Q_matrix.grad.data.zero_()
# %%
sample = lambda s: sample_policy_discrete(s, π, σ, ϵ, a_s, 2, s_s, n_s)[0][-1]
x_ls = torch.arange(n_s, dtype=np.int)
common_args = {"new_Q_net": lambda: Q_Tabular(n_a, n_s),"epochs": 2,"x_ls":x_ls}
sgd_args = {"M":1000, "τ_k":lambda k:0.1}
cbo_args = {"N":30, "m":1000, "γ":0.9, "δ":1e-3, "τ_k":lambda k: 0.1, "η_k":lambda k: 0.5, "β_k":lambda k: 10}
# %%
Q_ctrl_UR_SGD_star = Q_SGD_gen(Q_ctrl_UR_SGD_update_step)(
    S, A_idx, R, a_s, π, sample, **common_args, **sgd_args
)
# %%
Q_ctrl_UR_SGD, e_ctrl_UR_SGD = Q_SGD_gen(Q_ctrl_UR_SGD_update_step)(
    S, A_idx, R, a_s, π, sample, **common_args, **sgd_args,  Q_net_comp=Q_ctrl_UR_SGD_star
)
Q_ctrl_DS_SGD, e_ctrl_DS_SGD = Q_SGD_gen(Q_ctrl_DS_SGD_update_step)(
    s, A_idx, r, a_s, π, sample, **common_args, **sgd_args,  Q_net_comp=Q_ctrl_UR_SGD_star
)
Q_ctrl_BFF_SGD, e_ctrl_BFF_SGD = Q_SGD_gen(Q_ctrl_BFF_SGD_update_step)(
    S, A_idx, R, a_s, π, sample, **common_args, **sgd_args,  Q_net_comp=Q_ctrl_UR_SGD_star
)
# %%
Q_ctrl_UR_CBO, e_ctrl_UR_CBO = Q_CBO_gen(Q_ctrl_UR_CBO_L)(
    S, A_idx, R, a_s, π, sample,**common_args, **cbo_args,  Q_net_comp=Q_ctrl_UR_SGD_star
)
Q_ctrl_DS_CBO, e_ctrl_DS_CBO = Q_CBO_gen(Q_ctrl_DS_CBO_L)(
    S, A_idx, R, a_s, π, sample,**common_args, **cbo_args, Q_net_comp=Q_ctrl_UR_SGD_star
)
Q_ctrl_BFF_CBO, e_ctrl_BFF_CBO = Q_CBO_gen(Q_ctrl_BFF_CBO_L)(
    S, A_idx, R, a_s, π, sample,**common_args, **cbo_args, Q_net_comp=Q_ctrl_UR_SGD_star
)
# %%
plotQ(
    [Q_ctrl_UR_SGD,Q_ctrl_DS_SGD,Q_ctrl_BFF_SGD],
    [e_ctrl_UR_SGD,e_ctrl_DS_SGD,e_ctrl_BFF_SGD],
    ["UR", "DS", "BFF"], Q_ctrl_UR_SGD_star, "UR*", a_s, x_ls)
plt.savefig("figs/Q_ctrl_SGD_discrete_.png")
# %%
Q_dict = {
    "SGD": [
        [Q_ctrl_UR_SGD,Q_ctrl_DS_SGD,Q_ctrl_BFF_SGD],
        [e_ctrl_UR_SGD,e_ctrl_DS_SGD,e_ctrl_BFF_SGD],
        ["UR", "DS", "BFF"],
        ["C0", "C1", "C2"],
        ["solid","solid"]
    ],
    "BFF": [
        [Q_ctrl_UR_CBO,Q_ctrl_DS_CBO,Q_ctrl_BFF_CBO],
        [e_ctrl_UR_CBO,e_ctrl_DS_CBO,e_ctrl_BFF_CBO],
        ["UR", "DS", "BFF"],
        ["C0", "C1", "C2"],
        ["solid","solid"]
    ]
}
plotQ2(Q_dict, Q_ctrl_UR_SGD_star, "UR * SGD", a_s, x_s=x_ls)

# %%
