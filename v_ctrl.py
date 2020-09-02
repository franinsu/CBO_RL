# %%
import torch
from helper import *
from IPython import get_ipython
import torch.nn as nn
import pandas as pd
import seaborn as sns

# %%
def π(a, s):
    return torch.tensor([1 / 2.0, 1 / 2.0]) * torch.ones_like(s)

def r(s):
    return torch.sin(s) + 1

σ = 0.2
ϵ = 2.0 * np.pi / 32.0
s_0 = torch.tensor([0.0])
a_s = torch.tensor([-1.0, 1.0])
def new_Q_net(): return Q_ResNet()
# %%
# %%
# S, A_idx = sample_policy(s_0, π, σ, ϵ, a_s, int(1e6))
# R = r(S)
# S_long, A_idx_long = sample_policy(s_0, π, σ, ϵ, a_s, int(1e7))
# R_long = r(S_long)


# %%
# torch.save(S_long, 'cache/S_long_2.pt')
# torch.save(A_idx_long, 'cache/A_idx_long_2.pt')
# torch.save(R_long, 'cache/R_long_2.pt')
# torch.save(S, 'cache/S_2.pt')
# torch.save(A_idx, 'cache/A_idx_2.pt')
# torch.save(R, 'cache/R_2.pt')


# %%
S_long = torch.load("cache/S_long_2.pt")
R_long = torch.load("cache/R_long_2.pt")
A_idx_long = torch.load("cache/A_idx_long_2.pt")
S = torch.load("cache/S_2.pt")
R = torch.load("cache/R_2.pt")
A_idx = torch.load("cache/A_idx_2.pt")

# %%
# Q_ctrl_UR_SGD_star = Q_SGD_gen(Q_ctN = 90
# m = 1000
# epochs = 1
# δ = 1e-5

# def η_k(k): return max(0.1*0.9992**k, 0.075)
# def τ_k(k): return max(0.8*0.9992**k, 0.3)
# def β_k(k): return min(8*1.002**k,20)
# early_stop = 1000rl_UR_SGD_update_step)(
#     S_long, A_idx_long, R_long, a_s, π, σ, ϵ, M=1000, epochs=10, τ_k=lambda k:0.01
# )
# torch.save(Q_ctrl_UR_SGD_star, "cache/Q_ctrl_UR_SGD_star.pt")
Q_ctrl_UR_SGD_star = torch.load("cache/Q_ctrl_UR_SGD_star.pt")
# %%
# %%
M = 1000
epochs = 1
def τ_k(k):
    return max(0.5 * 0.9992 ** k, 0.075)
#%%
k_s = np.arange(M)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_s, [τ_k(k) for k in k_s])
ax.set_title(f"$τ_k$")
# %%
def run_SGD_all():
    Q_ctrl_UR_SGD, e_ctrl_UR_SGD = Q_SGD_gen(Q_ctrl_UR_SGD_update_step)(
        S, A_idx, R, a_s, π, σ, ϵ,new_Q_net=new_Q_net,
        M=M, epochs=epochs, τ_k=τ_k, Q_net_comp=Q_ctrl_UR_SGD_star
    )
    Q_ctrl_DS_SGD, e_ctrl_DS_SGD = Q_SGD_gen(Q_ctrl_DS_SGD_update_step)(
        S, A_idx, R, a_s, π, σ, ϵ,new_Q_net=new_Q_net,
        M=M, epochs=epochs, τ_k=τ_k, Q_net_comp=Q_ctrl_UR_SGD_star
    )
    Q_ctrl_BFF_SGD, e_ctrl_BFF_SGD = Q_SGD_gen(Q_ctrl_BFF_SGD_update_step)(
        S, A_idx, R, a_s, π, σ, ϵ, new_Q_net=new_Q_net,
        M=M, epochs=epochs, τ_k=τ_k, Q_net_comp=Q_ctrl_UR_SGD_star
    )
    return (Q_ctrl_UR_SGD, Q_ctrl_DS_SGD, Q_ctrl_BFF_SGD), (e_ctrl_UR_SGD, e_ctrl_DS_SGD, e_ctrl_BFF_SGD)
# %%
# Q_s, e_s = run_SGD_all()
# lb_s = ["UR", "DS", "BFF"]
# plotQ(Q_s, e_s, lb_s, Q_ctrl_UR_SGD_star, "UR *", a_s)

# %%
# N = 90
# m = 1000
# epochs = 1
# δ = 1e-5

# def η_k(k): return max(0.1*0.9992**k, 0.075)
# def τ_k(k): return max(0.8*0.9992**k, 0.3)
# def β_k(k): return min(8*1.002**k,20)
# early_stop = 1000
# %%
N = 90
m = 1000
epochs = 1
δ = 1e-5


def η_k(k):
    return max(0.1 * 0.9992 ** k, 0.075)


def τ_k(k):
    return max(0.7 * 0.9992 ** k, 0.2)


def β_k(k):
    return min(8 * 1.002 ** k, 20)


early_stop = 1000

# %%
k_s = np.arange(early_stop)
fig, axs = plt.subplots(figsize=(10, 5), ncols=3)
for i, (f, s) in enumerate(zip([η_k, τ_k, β_k], ["η_k", "τ_k", "β_k"])):
    axs[i].plot(k_s, [f(k) for k in k_s])
    axs[i].set_title(s)

# %%


def run_CBO_all():
    Q_ctrl_UR_CBO, e_ctrl_UR_CBO = Q_CBO_gen(Q_ctrl_UR_CBO_L)(
        S,
        A_idx,
        R,
        a_s,
        π,
        σ,
        ϵ,
        new_Q_net=new_Q_net,
        N=N,
        m=m,
        epochs=epochs,
        τ_k=τ_k,
        η_k=η_k,
        β_k=β_k,
        δ=δ,
        Q_net_comp=Q_ctrl_UR_SGD_star,
        early_stop=early_stop,
    )
    Q_ctrl_DS_CBO, e_ctrl_DS_CBO = Q_CBO_gen(Q_ctrl_DS_CBO_L)(
        S,
        A_idx,
        R,
        a_s,
        π,
        σ,
        ϵ,
        new_Q_net=new_Q_net,
        N=N,
        m=m,
        epochs=epochs,
        τ_k=τ_k,
        η_k=η_k,
        β_k=β_k,
        δ=δ,
        Q_net_comp=Q_ctrl_UR_SGD_star,
        early_stop=early_stop,
    )
    Q_ctrl_BFF_CBO, e_ctrl_BFF_CBO = Q_CBO_gen(Q_ctrl_BFF_CBO_L)(
        S,
        A_idx,
        R,
        a_s,
        π,
        σ,
        ϵ,
        new_Q_net=new_Q_net,
        N=N,
        m=m,
        epochs=epochs,
        τ_k=τ_k,
        η_k=η_k,
        β_k=β_k,
        δ=δ,
        Q_net_comp=Q_ctrl_UR_SGD_star,
        early_stop=early_stop,
    )
    return (
        (Q_ctrl_UR_CBO, Q_ctrl_DS_CBO, Q_ctrl_BFF_CBO),
        (e_ctrl_UR_CBO, e_ctrl_DS_CBO, e_ctrl_BFF_CBO),
    )


# %%

# Q_s, e_s = run_all()
# #%%
# (Q_ctrl_UR_CBO, Q_ctrl_DS_CBO, Q_ctrl_BFF_CBO) = Q_s
# (e_ctrl_UR_CBO, e_ctrl_DS_CBO, e_ctrl_BFF_CBO) = e_s
# #%%
# lb_s = ["UR","DS","BFF"]
# plotQ(Q_s, e_s, lb_s, Q_ctrl_UR_SGD_star, "UR *", a_s)

# Q_dict = {
#     "UR": [
#         [Q_ctrl_UR_SGD, Q_ctrl_UR_CBO],
#         [e_ctrl_UR_SGD, e_ctrl_UR_CBO],
#         ["SGD", "CBO"],
#         ["C0", "C1"],
#         ["solid", "solid"]
#     ],
#     "DS": [
#         [Q_ctrl_DS_SGD, Q_ctrl_DS_CBO],
#         [e_ctrl_DS_SGD, e_ctrl_DS_CBO],
#         ["SGD", "CBO"],
#         ["C0", "C1"],
#         ["solid", "solid"]
#     ],
#     "BFF": [
#         [Q_ctrl_BFF_SGD, Q_ctrl_BFF_CBO],
#         [e_ctrl_BFF_SGD, e_ctrl_BFF_CBO],
#         ["SGD", "CBO"],
#         ["C0", "C1"],
#         ["solid", "solid"]
#     ]
# }
# plotQ2(Q_dict, Q_ctrl_UR_SGD_star, "UR * SGD", a_s)
# plt.savefig("figs/Q_ctrl_SGD_vs_CBO_2.png")

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
plt.savefig("figs/Q_ctrl_SGD_vs_CBO_summary_ResNet.png")
# %%
# N = 90
# m = 1000
# epochs = 1
# δ = 1e-5
# early_stop = 1000
#%%
# def test(N=90):
#     Q_ctrl_BFF_CBO, e_ctrl_BFF_CBO = Q_CBO_gen(Q_ctrl_BFF_CBO_L)(
#         S, A_idx, R, a_s, π, σ, ϵ,
#         N=N, m=m, epochs=1, τ_k=τ_k, η_k=η_k, β_k=β_k, δ=1e-5, Q_net_comp=Q_ctrl_UR_SGD_star,early_stop=early_stop
#     )
#     Q_s = [Q_ctrl_BFF_CBO]
#     e_s = [e_ctrl_BFF_CBO]
#     lb_s = ["BFF"]
#     plotQ(Q_s, e_s, lb_s, Q_ctrl_UR_SGD_star, "UR *", a_s)
#     plt.show()

# #%%
# for x in [0.3,0.4,0.5]:
#     def η_k(k): return max(0.1*0.9992**k, 0.075)
#     def τ_k(k): return max(0.8*0.9992**k, x)
#     def β_k(k): return min(8*1.002**k,20)
#     test()
#     print(f"x={x}")

# # %%
# def η_k(k): return max(0.1*0.9992**k, 0.08)
# def τ_k(k): return max(0.75*0.9992**k, 0.3)
# def β_k(k): return min(8*1.002**k,20)
# Q_s = []
# e_s = []
# lb_s = []
# for _ in range(3):
#     Q_ctrl_BFF_CBO, e_ctrl_BFF_CBO = Q_CBO_gen(Q_ctrl_BFF_CBO_L)(
#         S, A_idx, R, a_s, π, σ, ϵ,
#         N=N, m=m, epochs=1, τ_k=τ_k, η_k=η_k, β_k=β_k, δ=1e-5, Q_net_comp=Q_ctrl_UR_SGD_star,early_stop=early_stop
#     )
#     Q_s.append(Q_ctrl_BFF_CBO)
#     e_s.append(e_ctrl_BFF_CBO)
#     lb_s.append("BFF")
# plotQ(Q_s, e_s, lb_s, Q_ctrl_UR_SGD_star, "UR *", a_s)


# %%
