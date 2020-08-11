# %%
import torch
from helper import *
from IPython import get_ipython
import torch.nn as nn

# %%

def π(a, s): return torch.tensor([1/2., 1/2.])*torch.ones_like(s)


def r(s): return torch.sin(s)+1


σ = 0.2
ϵ = 2.*np.pi/32.
s_0 = torch.tensor([0.])
a_s = torch.tensor([-1., 1.])


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
S_long = torch.load('cache/S_long_2.pt')
R_long = torch.load('cache/R_long_2.pt')
A_idx_long = torch.load('cache/A_idx_long_2.pt')
S = torch.load('cache/S_2.pt')
R = torch.load('cache/R_2.pt')
A_idx = torch.load('cache/A_idx_2.pt')

# %%
# Q_ctrl_UR_SGD_star = Q_SGD_gen(Q_ctrl_UR_SGD_update_step)(
#     S_long, A_idx_long, R_long, a_s, π, σ, ϵ, M=1000, epochs=10, τ_k=lambda k:0.01
# )
# torch.save(Q_ctrl_UR_SGD_star, "cache/Q_ctrl_UR_SGD_star.pt")
Q_ctrl_UR_SGD_star = torch.load("cache/Q_ctrl_UR_SGD_star.pt")


# %%
Q_ctrl_UR_SGD, e_ctrl_UR_SGD = Q_SGD_gen(Q_ctrl_UR_SGD_update_step)(
    S, A_idx, R, a_s, π, σ, ϵ,
    M=1000, epochs=1, τ_k=lambda k: 0.1, Q_net_comp=Q_ctrl_UR_SGD_star
)
Q_ctrl_DS_SGD, e_ctrl_DS_SGD = Q_SGD_gen(Q_ctrl_DS_SGD_update_step)(
    S, A_idx, R, a_s, π, σ, ϵ,
    M=1000, epochs=1, τ_k=lambda k: 0.1, Q_net_comp=Q_ctrl_UR_SGD_star
)
Q_ctrl_BFF_SGD, e_ctrl_BFF_SGD = Q_SGD_gen(Q_ctrl_BFF_SGD_update_step)(
    S, A_idx, R, a_s, π, σ, ϵ,
    M=1000, epochs=1, τ_k=lambda k: 0.1, Q_net_comp=Q_ctrl_UR_SGD_star
)
# %%
Q_s = [Q_ctrl_UR_SGD, Q_ctrl_DS_SGD, Q_ctrl_BFF_SGD]
e_s = [e_ctrl_UR_SGD, e_ctrl_DS_SGD, e_ctrl_BFF_SGD]
lb_s = ["UR", "DS", "BFF"]
plotQ(Q_s, e_s, lb_s, Q_ctrl_UR_SGD_star, "UR *", a_s)

# %%
N = 50  # 30
m = 1000
epochs = 1
δ = 1e-5

def η_k(k): return 0.1
def τ_k(k): return max(0.8*0.9995**k, 0.6)
def β_k(k): return 20
early_stop = 1000

#%%
k_s = np.arange(early_stop)
fig, axs = plt.subplots(figsize=(10,5), ncols=3)
for i,(f,s) in enumerate(zip([η_k, τ_k, β_k], ["η_k","τ_k","β_k"])):
    axs[i].plot(k_s, [f(k) for k in k_s])
    axs[i].set_title(s)
# %%
# Q_ctrl_UR_CBO, e_ctrl_UR_CBO = Q_CBO_gen(Q_ctrl_UR_CBO_L)(
#     S, A_idx, R, a_s, π, σ, ϵ,
#     N=N, m=m, epochs=1, τ_k=τ_k, η_k=η_k, β_k=β_k, δ=1e-5, Q_net_comp=Q_ctrl_UR_SGD_star,early_stop=early_stop
# )
# Q_ctrl_DS_CBO, e_ctrl_DS_CBO = Q_CBO_gen(Q_ctrl_DS_CBO_L)(
#     S, A_idx, R, a_s, π, σ, ϵ,
#     N=N, m=m, epochs=1, τ_k=τ_k, η_k=η_k, β_k=β_k, δ=1e-5, Q_net_comp=Q_ctrl_UR_SGD_star,early_stop=early_stop
# )
Q_ctrl_BFF_CBO, e_ctrl_BFF_CBO = Q_CBO_gen(Q_ctrl_BFF_CBO_L)(
    S, A_idx, R, a_s, π, σ, ϵ,
    N=N, m=m, epochs=1, τ_k=τ_k, η_k=η_k, β_k=β_k, δ=1e-5, Q_net_comp=Q_ctrl_UR_SGD_star,early_stop=early_stop
)
#%%
Q_s = [Q_ctrl_BFF_CBO]
e_s = [e_ctrl_BFF_CBO]
lb_s = ["BFF"]
plotQ(Q_s, e_s, lb_s, Q_ctrl_UR_SGD_star, "UR *", a_s)

# %%
# Q_s = [Q_ctrl_UR_CBO, Q_ctrl_DS_CBO, Q_ctrl_BFF_CBO]
# e_s = [e_ctrl_UR_CBO, e_ctrl_DS_CBO, e_ctrl_BFF_CBO]
# lb_s = ["UR", "DS", "BFF"]
# plotQ(Q_s, e_s, lb_s, Q_ctrl_UR_SGD_star, "UR *")

# %%
# Q_dict = {
#     # "UR": [
#     #     [Q_ctrl_UR_SGD, Q_ctrl_UR_CBO],
#     #     [e_ctrl_UR_SGD, e_ctrl_UR_CBO],
#     #     ["SGD", "CBO"],
#     #     ["C0", "C1"],
#     #     ["solid", "solid"]
#     # ],
#     # "DS": [
#     #     [Q_ctrl_DS_SGD, Q_ctrl_DS_CBO],
#     #     [e_ctrl_DS_SGD, e_ctrl_DS_CBO],
#     #     ["SGD", "CBO"],
#     #     ["C0", "C1"],
#     #     ["solid", "solid"]
#     # ],
#     "BFF": [
#         [Q_ctrl_BFF_SGD, Q_ctrl_BFF_CBO],
#         [e_ctrl_BFF_SGD, e_ctrl_BFF_CBO],
#         ["SGD", "CBO"],
#         ["C0", "C1"],
#         ["solid", "solid"]
#     ],
#     "BFF2": [
#         [Q_ctrl_BFF_SGD, Q_ctrl_BFF_CBO],
#         [e_ctrl_BFF_SGD, e_ctrl_BFF_CBO],
#         ["SGD", "CBO"],
#         ["C0", "C1"],
#         ["solid", "solid"]
#     ]
# }
# plotQ2(Q_dict, Q_ctrl_UR_SGD_star, "UR * SGD", a_s)

# %%
N = 30
m = 1000
epochs = 1
δ = 1e-5
early_stop = 1000

def test():
    Q_ctrl_BFF_CBO, e_ctrl_BFF_CBO = Q_CBO_gen(Q_ctrl_BFF_CBO_L)(
        S, A_idx, R, a_s, π, σ, ϵ,
        N=N, m=m, epochs=1, τ_k=τ_k, η_k=η_k, β_k=β_k, δ=1e-5, Q_net_comp=Q_ctrl_UR_SGD_star,early_stop=early_stop
    )
    Q_s = [Q_ctrl_BFF_CBO]
    e_s = [e_ctrl_BFF_CBO]
    lb_s = ["BFF"]
    plotQ(Q_s, e_s, lb_s, Q_ctrl_UR_SGD_star, "UR *", a_s)
    plt.show()

#%%
for x in [0.999,0.9992,0.9995,0.9998]:
    def η_k(k): return max(0.1*0.9998**k, 0.01)
    def τ_k(k): return max(0.75*x**k, 0.3)
    def β_k(k): return min(8*1.002**k,20)
    test()
    print(f"x={x}")



# %%
