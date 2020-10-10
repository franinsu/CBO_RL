# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange
import matplotlib.colors as c
# %%
n = int(1e4)
x_hat = torch.randn(n)*np.sqrt(0.1)


def L_41(x):
    return torch.exp(torch.sin(2*x**2))+0.1*torch.mean((x-x_hat-np.pi/2)**2)


def L_42(x, B=0, C=0):
    return torch.mean((x-B)**2-10*torch.cos(2*np.pi*(x-B))+10)+C


L = L_42
# %%
x_ls = torch.linspace(-3, 3, 200)
L_ls = torch.tensor([L(x_i) for x_i in x_ls])
# x_star = torch.tensor(np.pi/2)
x_star = x_ls[torch.argmin(L_ls)]
plt.plot(x_ls, L_ls, "k", label=f"$L(x)$")
plt.plot(x_star, L(x_star), ".r", label=f"$x^*$")
plt.legend()
# %%


def SGD(M=10, τ_k=lambda k: 0.1, σ=2):
    x = (torch.rand(1)*2-1)*σ
    x.requires_grad_(True)
    e_ls, L_ls, x_ls = [], [], []
    for i in range(M):
        τ = τ_k(i)
        L_x = L(x)
        L_x.backward()
        with torch.no_grad():
            e_ls.append(torch.norm(x-x_star))
            L_ls.append(L_x)
            x_ls.append(x.item())
            x -= τ*x.grad
    return torch.tensor(e_ls), torch.tensor(L_ls), torch.tensor(x_ls)
# %%


def CBO(M=10, N=30, τ=0.1, σ=2,
        λ=1., τ_k=lambda k: 0.1, η_k=lambda k: 0.5,
        β_k=lambda k: 10, approach="Euler"):
    with torch.no_grad():
        x = (torch.rand(N)*2-1)*σ
        e_ls, L_ls, x_ls, ē_ls, L̄_ls, x̄_ls = [], [], [], [], [], []
        z = torch.zeros(N)
        L_x = torch.tensor([L(x_i) for x_i in x])
        if approach == "Euler":
            def update_x(x, η, τ):
                return x + (-λ*η+τ*np.sqrt(η)*torch.normal(z))*(x-x̄)
        elif approach == "Split":
            def update_x(x, η, τ):
                x = x̄ + (x-x̄)*np.exp(-λ*η)
                return x + τ*np.sqrt(η)*torch.normal(z)*(x-x̄)
        else:
            raise NameError(f"Approach {approach} not existent")
        for i in range(M):
            β = β_k(i)
            τ = τ_k(i)
            η = η_k(i)

            μ = torch.exp(-β * L_x)
            x̄ = torch.sum(x*μ)/torch.sum(μ)
            x = update_x(x, η, τ)
            L_x = torch.tensor([L(x_i) for x_i in x])

            e_ls.append(torch.norm((x-x_star).view(1, -1), dim=0))
            L_ls.append(L_x)
            x_ls.append(x.clone())
            ē_ls.append(torch.norm(x̄- x_star).item())
            L̄_ls.append(L(x̄).item())
            x̄_ls.append(x̄.item())

    return torch.stack(e_ls), torch.stack(L_ls), torch.stack(x_ls), ē_ls, L̄_ls, x̄_ls


# %%
# M=int(1e4)
M = int(1e2)
σ = 3
# %%


# def τ_k(k): return 0.01
def τ_k(k): return 0.002

e_ls_SGD, L_ls_SGD, x_ls_SGD = SGD(M=M, τ_k=τ_k, σ=σ)
# %%
N = 5
λ = 1


def τ_k(k): return 1


def η_k(k): return 0.01


def β_k(k): return 30


e_ls_CBO, L_ls_CBO, x_ls_CBO, ē_ls_CBO, L̄_ls_CBO, x̄_ls_CBO = CBO(
    M=M, N=N, σ=σ, λ=λ, τ_k=τ_k, η_k=η_k, β_k=β_k)
#%%
def pltRun(x_ls, L_ls, e_ls, color, M, label, label_rep):
    rgba = c.to_rgb(color)
    alphas = np.linspace(0.1, 1, M)
    rgba_colors = np.zeros((M, 4))
    rgba_colors[:, :3] = rgba
    rgba_colors[:, 3] = alphas
    axs[0].scatter(x_ls, L_ls, color=rgba_colors, label=f"$x$ SGD", s=20)
    axs[0].plot(x_ls, L_ls, color=rgba, alpha=0.5)
    axs[1].plot(e_ls, color=rgba)
    if label not in labels:
        labels.append(label)
        labels_rep.append(label_rep)
# %%
fig, axs = plt.subplots(ncols=2, figsize=(15, 5))
x_ls_all = torch.linspace(-3, 3, 200)
L_ls_all = torch.tensor([L(x_i) for x_i in x_ls_all])
x_star = x_ls_all[torch.argmin(L_ls_all)]
axs[0].plot(x_ls_all, L_ls_all, "k", label=f"$L(x)$")
axs[0].plot(x_star, L(x_star), ".r", label=f"$x^*$")
axs[0].set_xlabel(r"$x$")
axs[0].set_ylabel(r"$L(x)$")
axs[1].set_xlabel(r"$k$")
axs[1].set_ylabel(r"$e_k$")
labels = [f"$L(x)$", f"$x^*$"]
labels_rep = ["k-", "r."]


pltRun(x_ls_SGD, L_ls_SGD, e_ls_SGD, "C0", M, r"$x_{SGD}$", "C0")
pltRun(x̄_ls_CBO, L̄_ls_CBO, ē_ls_CBO, "C1", M, r"$\bar{x}_{CBO}$", "C1")
for i in range(N):
    pltRun(x_ls_CBO[:, i], L_ls_CBO[:, i],
           e_ls_CBO[:, i], "C2", M, r"$x_{CBO}$", "C2")
fig.legend(labels_rep,
           labels=labels,
           loc="center right",
           borderaxespad=0.1,
           title=""
           )

plt.subplots_adjust(right=0.94)
plt.savefig("figs/Simpler_SGD_vs_CBO.png")
# %%

# %%

# %%
