# %%
import torch
from RL_Optimization import *
import pandas as pd
import seaborn as sns
import pickle
# %%
problem_suffix= "continuous"
model_suffix = "resnet"
# %%
if problem_suffix=="continuous":
    from continuous_example import *
    if model_suffix=="resnet":
        def new_Q_net(): return Q_ResNet()
    else:
        def new_Q_net(): return Q_Net()
else:
    from discrete_example import *
    def new_Q_net(): return Q_Tabular()
# %%
S_long = torch.load(f"cache/S_long_{problem_suffix}.pt")
R_long = torch.load(f"cache/R_long_{problem_suffix}.pt")
A_idx_long = torch.load(f"cache/A_idx_long_{problem_suffix}.pt")
S = torch.load(f"cache/S_{problem_suffix}.pt")
R = torch.load(f"cache/R_{problem_suffix}.pt")
A_idx = torch.load(f"cache/A_idx_{problem_suffix}.pt")
# %%
sample = sampler()
a_s = get_parameters()["a_s"]
x_ls = get_parameters()["x_ls"]
M = 1000
epochs = 1
args_0 = [S, A_idx, R, a_s, π, sample]
Q_ctrl_UR_SGD_star = torch.load(f"cache/Q_ctrl_UR_SGD_star_{problem_suffix}_{model_suffix}.pt")
# %%
common_args = {"new_Q_net": new_Q_net, "Q_net_comp": Q_ctrl_UR_SGD_star,  "epochs": epochs, "x_ls": x_ls}
# %%
τ_i, τ_f, τ_r = 0.5, 0.075, 0.9992

def τ_k(k):
    return max(τ_i * τ_r ** k, τ_f*τ_i)

sgd_args = {"τ_k": τ_k, "M": M}
# %%
def run_SGD_all():
    sgd_u_s = [Q_ctrl_UR_SGD_update_step,
               Q_ctrl_DS_SGD_update_step, Q_ctrl_BFF_SGD_update_step]
    Qs, es = [], []
    for u_s in sgd_u_s:
        q, e = Q_SGD_gen(u_s)(
            *args_0, **common_args, **sgd_args)
        Qs.append(q)
        es.append(e)
    return (Qs, es)

# %%
η_i, η_f, η_r, τ_i, τ_f, τ_r, β_i, β_f, β_r = 0.1,0.075,0.9992, 0.7, 0.2, 0.9992, 8, 20, 1.002
N = 90
m = 1000
δ = 1e-5
early_stop = 1000
def η_k(k): return max(η_i * η_r ** k, η_f*η_i)
def τ_k(k): return max(τ_i * τ_r ** k, τ_f*τ_i)
def β_k(k): return min(β_i * β_r ** k, β_f*β_i)


cbo_args = {"N": N, "m": m, "τ_k": τ_k, "η_k": η_k,
            "β_k": β_k, "δ": δ, "early_stop": early_stop}
# %%

def run_CBO_all():
    cbo_u_s = [Q_ctrl_UR_CBO_L, Q_ctrl_DS_CBO_L, Q_ctrl_BFF_CBO_L]
    Qs, es = [], []
    for u_s in cbo_u_s:
        q, e = Q_CBO_gen(u_s)(
            *args_0, **common_args, **cbo_args)
        Qs.append(q)
        es.append(e)
    return (Qs, es)
# %%
all_data = pd.DataFrame(columns=["i", "x", "y", "sampling", "algo", "plot"])
x_s = torch.linspace(0, 2 * np.pi, 100)
y_star = Q_ctrl_UR_SGD_star(x_s.view(-1, 1))
a_n = len(a_s)
early_stop = 1000
n_runs = 1
for j, lb in enumerate(["UR", "DS", "BFF"]):
    for k in range(a_n):
        all_data = all_data.append(
            pd.DataFrame(
                {
                    "i": 0,
                    "x": x_s.detach().numpy(),
                    "y": y_star[:, k].detach().numpy(),
                    "sampling": lb,
                    "algo": "UR SGD *",
                    "plot": f"$Q(a_{k},s)$",
                }
            )
        )
for i in range(n_runs):
    for r, algo in [(run_SGD_all, "SGD"), (run_CBO_all, "CBO")]:
        Q_s, e_s = r()
        y_s = torch.stack([Q(x_s.view(-1, 1)) for Q in Q_s])
        y_s -= torch.unsqueeze(torch.mean(y_s - y_star, axis=1), 1)
        for j, lb in enumerate(["UR", "DS", "BFF"]):
            for k in range(a_n):
                all_data = all_data.append(
                    pd.DataFrame(
                        {
                            "i": i,
                            "x": x_s.detach().numpy(),
                            "y": y_s[j, :, k].detach().numpy(),
                            "sampling": lb,
                            "algo": algo,
                            "plot": f"$Q(a_{k},s)$",
                        }
                    )
                )
            all_data = all_data.append(
                pd.DataFrame(
                    {
                        "i": i,
                        "x": np.arange(len(e_s[j])),
                        "y": e_s[j] / e_s[j][0],
                        "sampling": lb,
                        "algo": algo,
                        "plot": r"$e_k/e_0$",
                    }
                )
            )

# %%
g = sns.FacetGrid(
    data=all_data,
    col="plot",
    row="sampling",
    hue="algo",
    sharex=False,
    sharey=False,
    palette=["black", "C0", "C1"],
    margin_titles=True,
)
g.map(sns.lineplot, "x", "y")
g.set_axis_labels(x_var=f"$s$")
for ax in g.axes[:, 2]:
    ax.set_yscale("log")
g.axes[-1, -1].set_xlabel(f"$k$")
[plt.setp(ax.texts, text="") for ax in g.axes.flat]
g.set_ylabels("")
g.set_titles(col_template="{col_name}", row_template="{row_name}")
plt.tight_layout()
plt.subplots_adjust(right=0.9)
g.add_legend(title="")
plt.savefig(f"figs/Q_ctrl_SGD_vs_CBO_summary_{problem_suffix}_{model_suffix}.png")
# %%
