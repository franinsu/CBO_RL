import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.colors as colors
from torch.distributions.categorical import Categorical
from tqdm.auto import tqdm, trange
plt.style.use('default')
plt.style.use('seaborn-whitegrid')
mpl.rcParams['figure.dpi'] = 120


def sample_trajectory(s_0, α, σ, ϵ, m):
    s_0_size = s_0.size()
    S = torch.zeros(m, *s_0_size)
    S[0] = s_0
    for i in range(m-1):
        S[i+1] = S[i]+α(S[i])*ϵ+σ(S[i])*torch.sqrt(ϵ) * \
            torch.normal(mean=torch.zeros(*s_0_size))
    return S


class V_Net(nn.Module):
    def __init__(self):
        super(V_Net, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)
        self.apply(self.init_params)

    def forward(self, x):
        x = torch.cat((torch.sin(x), torch.cos(x)), 1)
        x = torch.cos(self.fc1(x))
        x = torch.cos(self.fc2(x))
        x = self.fc3(x)
        return x

    def init_params(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight)
            nn.init.normal_(m.bias)


class V_ResNet(V_Net):
    def forward(self, x):
        x = torch.cat((torch.sin(x), torch.cos(x)), 1)
        x_1 = torch.cos(self.fc1(x))
        x = torch.cos(self.fc2(x_1)+x_1)
        x = self.fc3(x)
        return x


def gen_batches(N, M, Rem):
    P = torch.randperm(N)
    if Rem.size()[0] > 0:
        I = torch.cat((Rem, P), 0)
    else:
        I = P
    q = int(np.floor((P.size()[0]+Rem.size()[0])/M))
    B = I[:(q*M)].view(q, M)
    Rem = I[(q*M):]
    return B, Rem


def V_comp(V_net, V_net_comp, x_ls, n):
    diff = V_net(x_ls)-V_net_comp(x_ls)
    return np.sqrt(2*np.pi/n)*torch.norm(diff-torch.mean(diff))


def V_comp_0(V_net, V_net_comp, x_ls, n):
    return np.sqrt(2*np.pi/n)*torch.norm(V_net(x_ls)-V_net_comp(x_ls))


def V_eval_SGD_gen(update_step):
    def algo(S, R, α, σ, ϵ, new_V_net=lambda: V_Net(), M=1000, epochs=100, γ=0.9, τ=0.01, V_net_comp=None, n_comp=1000):
        V_net = new_V_net()
        Rem = torch.tensor([])
        N = S.size()[0]
        i = 0
        if V_net_comp:
            x_ls = torch.linspace(0, 2*np.pi, n_comp+1)[:-1].view(-1, 1)
            e = [V_comp(V_net, V_net_comp, x_ls, n_comp)]
        for k in trange(epochs, leave=False, position=0, desc="Epoch"):
            B, Rem = gen_batches(N-2, M, Rem)
            for B_θ in tqdm(B, leave=False, position=0, desc="Batch"):
                update_step(V_net, γ, τ, S, R, B_θ, α, σ, ϵ)
                with torch.no_grad():
                    for param in V_net.parameters():
                        param -= τ/M*param.grad
                if V_net_comp:
                    e.append(V_comp(V_net, V_net_comp, x_ls, n_comp))
        if V_net_comp:
            return V_net, torch.tensor(e)
        else:
            return V_net
    return algo


def UR_SGD_update_step(V_net, γ, τ, S, R, B_θ, α, σ, ϵ):
    s = S[B_θ]
    s_1 = S[B_θ+1]
    ŝ_1 = s+α(s)*ϵ+σ(s)*torch.sqrt(ϵ)*torch.normal(mean=torch.zeros_like(s))
    f_p = R[B_θ] + γ*V_net(ŝ_1) - V_net(s)
    f = R[B_θ] + γ*V_net(s_1) - V_net(s)
    V_net.zero_grad()
    f_p.backward(f)


def DS_SGD_update_step(V_net, γ, τ, S, R, B_θ, α, σ, ϵ):
    s = S[B_θ]
    s_1 = S[B_θ+1]
    f = R[B_θ] + γ*V_net(s_1) - V_net(s)
    V_net.zero_grad()
    f.backward(f)


def BFF_G_SGD_update_step(V_net, γ, τ, S, R, B_θ, α, σ, ϵ):
    s = S[B_θ]
    s_1 = S[B_θ+1]
    s_2 = S[B_θ+2]
    s̃_1 = s + s_2 - s_1
    f = R[B_θ] + γ*V_net(s_1) - V_net(s)
    f̃ = R[B_θ] + γ*V_net(s̃_1) - V_net(s)
    V_net.zero_grad()
    (f*f̃).backward(torch.ones_like(f)/2.)


def BFF_L_SGD_update_step(V_net, γ, τ, S, R, B_θ, α, σ, ϵ):
    s = S[B_θ]
    s_1 = S[B_θ+1]
    s_2 = S[B_θ+2]
    s̃_1 = s + s_2 - s_1
    f = R[B_θ] + γ*V_net(s_1) - V_net(s)
    f̃ = R[B_θ] + γ*V_net(s̃_1) - V_net(s)
    V_net.zero_grad()
    f̃.backward(f)


def plotV(V_s, e_s, lb_s, V_star, lb_star):
    fig, axs = plt.subplots(figsize=(12, 5), ncols=2)
    x_s = torch.linspace(0, 2*np.pi, 1000)
    y_star = V_star(x_s.view(-1, 1)).view(-1)
    y_s = torch.cat([V(x_s.view(-1, 1)).view(1, -1) for V in V_s])
    y_s -= torch.mean(y_s-y_star.view(1, -1), axis=1).view(-1, 1)
    axs[0].plot(x_s.detach().numpy(), y_star.detach().numpy(),
                label=lb_star, color="black")
    for j in range(0, len(y_s)):
        axs[0].plot(x_s.detach().numpy(),
                    y_s[j].detach().numpy(), label=lb_s[j])
        axs[1].plot(e_s[j]/e_s[j][0], label=lb_s[j])
    axs[0].set_ylabel(r"$V(s)$")
    axs[0].set_xlabel(r"$s$")
    axs[1].set_yscale("log")
    axs[1].set_ylabel(r"$e_k/e_0$ (log scale)")
    axs[1].set_xlabel(r"$k$")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)


def V_eval_CBO_gen(L_f):
    def algo(S, α, σ, ϵ, new_V_net=lambda: V_Net(), N=30, m=1000, epochs=100, γ=0.9, λ=1., δ=1e-3,
             τ_k=lambda k: 0.1, η_k=lambda k: 0.5, β_k=lambda k: 10, V_net_comp=None, n_comp=1000):
        with torch.no_grad():
            V_net = new_V_net()
            V_θ = [new_V_net() for _ in range(N)]
            rem = torch.tensor([])
            L = torch.empty(N)
            n = S.size()[0]
            n_params = sum(param.numel() for param in V_net.parameters())
            i = 0
            if V_net_comp:
                x_ls = torch.linspace(0, 2*np.pi, n+1)[:-1].view(-1, 1)
                e = [V_comp(V_net, V_net_comp, x_ls, n_comp)]
            for k in trange(epochs, leave=False, position=0, desc="Epoch"):
                A, rem = gen_batches(n-2, m, rem)
                for A_θ in tqdm(A, leave=False, position=0, desc="Batch"):
                    β = β_k(i)
                    τ = τ_k(i)
                    η = η_k(i)
                    i += 1
                    L = L_f(V_θ, A_θ, m, γ, S, R, α, σ, ϵ)
                    μ = torch.exp(-β * L)
                    Δx̄2 = 0
                    # Update parameters
                    for x̄X_params in zip(V_net.parameters(), *[V_j.parameters() for V_j in V_θ]):
                        x̄ = x̄X_params[0]
                        X = torch.stack(x̄X_params[1:])
                        x̄_new = torch.einsum("i,i...->...", μ, X)/torch.sum(μ)
                        Δx̄2 += torch.norm(x̄ - x̄_new)**2
                        x̄.copy_(x̄_new)
                        z = torch.zeros(x̄.size())
                        for X_j in x̄X_params[1:]:
                            X_j += (-λ*η+τ*np.sqrt(η)*torch.normal(z))*(X_j-x̄)
                    # Brownian motion if change in V_net params below threshold
                    if (Δx̄2/n_params) < δ:
                        for X_params in zip(*[V_j.parameters() for V_j in V_θ]):
                            z = torch.zeros(X_params[0].size())
                            for X_j in X_params:
                                X_j += τ*np.sqrt(η)*torch.normal(z)
                    if V_net_comp:
                        e.append(V_comp(V_net, V_net_comp, x_ls, n_comp))
            if V_net_comp:
                return V_net, torch.tensor(e)
            else:
                return V_net
    return algo


def UR_CBO_L(V_θ, A_θ, m, γ, S, R, α, σ, ϵ):
    s = S[A_θ]
    s_1 = S[A_θ+1]
    s_2 = S[A_θ+2]
    ŝ_1 = s+α(s)*ϵ+σ(s)*torch.sqrt(ϵ)*torch.normal(mean=torch.zeros_like(s))
    f = torch.cat([R[A_θ] + γ*V_j(s_1)-V_j(s) for V_j in V_θ], 1)
    f̂ = torch.cat([R[A_θ] + γ*V_j(ŝ_1)-V_j(s) for V_j in V_θ], 1)
    return torch.sum(f*f̂, 0)/(2*m)


def DS_CBO_L(V_θ, A_θ, m, γ, S, R, α, σ, ϵ):
    s = S[A_θ]
    s_1 = S[A_θ+1]
    f = torch.cat([R[A_θ] + γ*V_j(s_1)-V_j(s) for V_j in V_θ], 1)
    return torch.sum(f**2, 0)/(2*m)


def BFF_CBO_L(V_θ, A_θ, m, γ, S, R, α, σ, ϵ):
    s = S[A_θ]
    s_1 = S[A_θ+1]
    s_2 = S[A_θ+2]
    s̃_1 = s + s_2 - s_1
    f = torch.cat([R[A_θ] + γ*V_j(s_1)-V_j(s) for V_j in V_θ], 1)
    f̃ = torch.cat([R[A_θ] + γ*V_j(s̃_1)-V_j(s) for V_j in V_θ], 1)
    return torch.sum(f*f̃, 0)/(2*m)


def plotV2(V_dict, V_star, lb_star):
    n_r = len(V_dict)
    fig, axg = plt.subplots(figsize=(12, 8), ncols=2, nrows=n_r)
    x_s = torch.linspace(0, 2*np.pi, 1000)
    y_star = V_star(x_s.view(-1, 1)).view(-1)
    pad = 5
    for i, (plt_name, (V_s, e_s, lb_s, c_s, ls_s)) in enumerate(V_dict.items()):
        y_s = torch.cat([V(x_s.view(-1, 1)).view(1, -1) for V in V_s])
        y_s -= torch.mean(y_s-y_star.view(1, -1), axis=1).view(-1, 1)
        axr = axg[i]
        axr[0].annotate(plt_name, xy=(0, 0.5), xytext=(-axr[0].yaxis.labelpad - pad, 0),
                        xycoords=axr[0].yaxis.label, textcoords='offset points',
                        size=15, ha='right', va='center')
        axr[0].plot(x_s.detach().numpy(), y_star.detach().numpy(),
                    label=lb_star, color="black")
        for j in range(0, len(y_s)):
            axr[0].plot(x_s.detach().numpy(), y_s[j].detach().numpy(),
                        label=lb_s[j], color=c_s[j], ls=ls_s[j])
            axr[1].plot(e_s[j]/e_s[j][0], label=lb_s[j],
                        color=c_s[j], ls=ls_s[j])
        axr[0].set_ylabel(r"$V(s)$")
        axr[0].set_xlabel(r"$s$")
        axr[1].set_yscale("log")
        axr[1].set_ylabel(r"$e_k/e_0$ (log scale)")
        axr[1].set_xlabel(r"$k$")
    handles, labels = axg[-1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.90)


def sample_policy(s_0, π, σ, ϵ, a_s, m, verbose=False):
    s_0 = s_0.view(s_0.size()[0], -1)
    s_0_size = s_0.size()
    S = torch.zeros(m, *s_0_size)
    S[0] = s_0
    A_idx = torch.zeros(m, s_0_size[0])
    if verbose:
        i_range = trange(m-1, leave=False, position=0, desc="Samples")
    else:
        i_range = range(m-1)
    for i in i_range:
        for j, s in enumerate(S[i]):
            A_idx[i, j] = Categorical(probs=π(a_s, s)).sample()
            a = A_idx[i, j]*2. - 1.
            S[i+1, j] = s+a*ϵ+σ * \
                np.sqrt(ϵ)*torch.normal(mean=torch.zeros_like(s))
    return S.view(-1, 1), A_idx.view(-1, 1)


def sample_policy_discrete(s_0, π, σ, ϵ, a_s, m, s_s, n_s, verbose=False):
    # S returned will represent indices
    s_0 = s_0.view(s_0.size()[0], -1)
    s_0_size = s_0.size()
    S = torch.zeros(m, *s_0_size, dtype=int)
    S[0] = s_0
    A_idx = torch.zeros(m, s_0_size[0])
    z = torch.zeros(s_0_size, dtype=float)
    if verbose:
        i_range = trange(m-1, leave=False, position=0, desc="Samples")
    else:
        i_range = range(m-1)
    for i in i_range:
        for j, s in enumerate(S[i]):
            A_idx[i, j] = Categorical(probs=π(a_s, s_s[s])).sample()
            a = a_s[int(A_idx[i, j].item())]
            s̃ = s_s[s]+(2*np.pi/n_s)*a*ϵ+σ*np.sqrt(ϵ)*torch.normal(mean=z)
            S[i+1, j] = torch.argmin(torch.abs(s̃ - s_s))
    return S.squeeze(2), A_idx


class Q_Net(nn.Module):
    def __init__(self):
        super(Q_Net, self).__init__()
        self.fc1 = nn.Linear(2, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 2)
        self.apply(self.init_params)

    def forward(self, x):
        x = torch.cat((torch.sin(x), torch.cos(x)), 1)
        x = torch.cos(self.fc1(x))
        x = torch.cos(self.fc2(x))
        x = self.fc3(x)
        return x

    def init_params(self, m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight)
            nn.init.normal_(m.bias)


class Q_ResNet(Q_Net):
    def forward(self, x):
        x = torch.cat((torch.sin(x), torch.cos(x)), 1)
        x_1 = self.fc1(x)
        x = torch.cos(x_1)
        x = torch.cos(self.fc2(x)+x_1)
        x = self.fc3(x)
        return x


class Q_Tabular(nn.Module):
    def __init__(self, n_a, n_s):
        super(Q_Tabular, self).__init__()
        self.Q_matrix = torch.normal(mean=torch.zeros(n_s, n_a))
        self.Q_matrix.requires_grad = True
        self.n_s = n_s

    def forward(self, s):
        return self.Q_matrix[torch.remainder(torch.tensor(s), self.n_s), :].squeeze()

    def parameters(self):
        return [self.Q_matrix]

    def zero_grad(self):
        if self.Q_matrix.grad != None:
            self.Q_matrix.grad.data.zero_()


def Q_comp(Q_net, Q_net_comp, x_ls, n):
    diff = Q_net(x_ls)-Q_net_comp(x_ls)
    return np.sqrt(2*np.pi/n)*torch.norm(diff-torch.mean(diff, axis=0))


def Q_comp_0(Q_net, Q_net_comp, x_ls, n):
    diff = Q_net(x_ls)-Q_net_comp(x_ls)
    return np.sqrt(2*np.pi/n)*torch.norm(diff)


def Q_SGD_gen(update_step):
    def algo(S, A_idx, R, a_s, π, sample, new_Q_net=lambda: Q_Net(), M=1000, epochs=100, γ=0.9, τ_k=lambda k: 0.1*0.999**k, Q_net_comp=None, x_ls=torch.linspace(0, 2*np.pi, 1000+1)[:-1]):
        Q_net = new_Q_net()
        Rem = torch.tensor([])
        N = S.size()[0]
        i = 0
        if Q_net_comp:
            n = len(x_ls)
            x_ls = x_ls.view(-1, 1)
            e = [Q_comp(Q_net, Q_net_comp, x_ls, n)]
        for k in trange(epochs, leave=False, position=0, desc="Epoch"):
            B, Rem = gen_batches(N-2, M, Rem)
            τ = τ_k(i)
            i += 1
            for B_θ in tqdm(B, leave=False, position=0, desc="Batch"):
                update_step(Q_net, γ, τ, S, A_idx, R, a_s, B_θ, M, π, sample)
                with torch.no_grad():
                    for param in Q_net.parameters():
                        param -= τ/M*param.grad
                if Q_net_comp:
                    e.append(Q_comp(Q_net, Q_net_comp, x_ls, n))
        if Q_net_comp:
            return Q_net, torch.tensor(e)
        else:
            return Q_net
    return algo


def Q_eval_UR_SGD_update_step(Q_net, γ, τ, S, A_idx, R, a_s, B_θ, M, π, sample):
    s = S[B_θ]
    r = R[B_θ].view(-1)
    a_idx = A_idx[B_θ].type(torch.LongTensor).view(-1)
    s_1 = S[B_θ+1]
    s̃_1 = sample(s)
    q = Q_net(s)[np.arange(M), a_idx]
    j = r + γ*torch.sum(Q_net(s_1)*π(a_s, s_1), axis=1) - q
    j̃ = r + γ*torch.sum(Q_net(s̃_1)*π(a_s, s̃_1), axis=1) - q
    Q_net.zero_grad()
    j.backward(j̃)


def Q_eval_DS_SGD_update_step(Q_net, γ, τ, S, A_idx, R, a_s, B_θ, M, π, sample):
    s = S[B_θ]
    r = R[B_θ].view(-1)
    a_idx = A_idx[B_θ].type(torch.LongTensor).view(-1)
    s_1 = S[B_θ+1]
    q = Q_net(s)[np.arange(M), a_idx]
    j = r + γ*torch.sum(Q_net(s_1)*π(a_s, s_1), axis=1) - q
    Q_net.zero_grad()
    j.backward(j)


def Q_eval_BFF_SGD_update_step(Q_net, γ, τ, S, A_idx, R, a_s, B_θ, M, π, sample):
    s = S[B_θ]
    r = R[B_θ].view(-1)
    a_idx = A_idx[B_θ].type(torch.LongTensor).view(-1)
    s_1 = S[B_θ+1]
    s_2 = S[B_θ+2]
    s̃_1 = s+s_2-s_1
    q = Q_net(s)[np.arange(M), a_idx]
    j = r + γ*torch.sum(Q_net(s_1)*π(a_s, s_1), axis=1) - q
    j̃ = r + γ*torch.sum(Q_net(s̃_1)*π(a_s, s̃_1), axis=1) - q
    Q_net.zero_grad()
    j.backward(j̃)


def plotQ(Q_s, e_s, lb_s, Q_star, lb_star, a_s, x_s=torch.linspace(0, 2*np.pi, 1000)):
    a_n = len(a_s)
    n = len(Q_s)
    fig, axs = plt.subplots(figsize=(12, 3), ncols=a_n+1)
    y_star = Q_star(x_s.view(-1, 1))
    y_s = torch.cat([Q(x_s.view(-1, 1)).view(1, -1, a_n) for Q in Q_s])
    y_s -= torch.mean(y_s-y_star.expand(n, -1, a_n), axis=1).view(n, -1, a_n)
    for i in range(a_n):
        axs[i].plot(x_s.detach().numpy(), y_star[:, i].detach(
        ).numpy(), label=lb_star, color="black")
        axs[i].set_ylabel(f"$Q(a_{i},s)$")
        axs[i].set_xlabel(r"$s$")
    for j in range(n):
        for i in range(a_n):
            axs[i].plot(x_s.detach().numpy(),
                        y_s[j, :, i].detach().numpy(), label=lb_s[j])
        axs[a_n].plot(e_s[j]/e_s[j][0], label=lb_s[j])
    axs[a_n].set_yscale("log")
    axs[a_n].set_ylabel(r"$e_k/e_0$ (log scale)")
    axs[a_n].set_xlabel(r"$k$")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.tight_layout()
    plt.subplots_adjust(right=0.87)


def Q_ctrl_UR_SGD_update_step(Q_net, γ, τ, S, A_idx, R, a_s, B_θ, M, π, sample):
    s = S[B_θ]
    r = R[B_θ].view(-1)
    a_idx = A_idx[B_θ].type(torch.LongTensor).view(-1)
    s_1 = S[B_θ+1]
    # ŝ_1 = [sample(s_) for s_ in s]
    ŝ_1 = torch.tensor([sample(s_) for s_ in s]).view(-1,1)
    q = Q_net(s)[np.arange(M), a_idx]
    j = r + γ*torch.max(Q_net(s_1), axis=1).values - q
    ĵ = r + γ*torch.max(Q_net(ŝ_1), axis=1).values - q
    Q_net.zero_grad()
    ĵ.backward(j)


def Q_ctrl_DS_SGD_update_step(Q_net, γ, τ, S, A_idx, R, a_s, B_θ, M, π, sample):
    s = S[B_θ]
    r = R[B_θ].view(-1)
    a_idx = A_idx[B_θ].type(torch.LongTensor).view(-1)
    s_1 = S[B_θ+1]
    q = Q_net(s)[np.arange(M), a_idx]
    j = r + γ*torch.max(Q_net(s_1), axis=1).values - q
    Q_net.zero_grad()
    j.backward(j)


def Q_ctrl_BFF_SGD_update_step(Q_net, γ, τ, S, A_idx, R, a_s, B_θ, M, π, sample):
    s = S[B_θ]
    r = R[B_θ].view(-1)
    a_idx = A_idx[B_θ].type(torch.LongTensor).view(-1)
    s_1 = S[B_θ+1]
    s_2 = S[B_θ+2]
    s̃_1 = s+s_2-s_1
    q = Q_net(s)[np.arange(M), a_idx]
    j = r + γ*torch.max(Q_net(s_1), axis=1).values - q
    j̃ = r + γ*torch.max(Q_net(s̃_1), axis=1).values - q
    Q_net.zero_grad()
    j̃.backward(j)


def Q_CBO_gen(L_f):
    def algo(S, A_idx, R, a_s, π, sample,
             new_Q_net=lambda: Q_Net(), N=30, m=1000, epochs=100, γ=0.9, λ=1., δ=1e-3,
             τ_k=lambda k: 0.1, η_k=lambda k: 0.5, β_k=lambda k: 10, Q_net_comp=None,
             x_ls=torch.linspace(0, 2*np.pi, 1000+1)[:-1], early_stop=None):
        with torch.no_grad():
            Q_net = new_Q_net()
            Q_θ = [new_Q_net() for _ in range(N)]
            rem = torch.tensor([])
            L = torch.empty(N)
            x_ls = x_ls.view(-1, 1)
            n_comp = len(x_ls)
            n = S.size()[0]
            n_params = sum(param.numel() for param in Q_net.parameters())
            i = 0
            if Q_net_comp:
                e = [Q_comp(Q_net, Q_net_comp, x_ls, n_comp)]
            for k in trange(epochs, leave=False, position=0, desc="Epoch"):
                A, rem = gen_batches(n-2, m, rem)
                for A_θ in tqdm(A, leave=False, position=0, desc="Batch"):
                    if early_stop and i > early_stop:
                        continue
                    β = β_k(i)
                    τ = τ_k(i)
                    η = η_k(i)
                    i += 1
                    L = L_f(Q_θ, A_θ, m, γ, S, A_idx,
                            R, a_s, π=π, sample=sample)
                    μ = torch.exp(-β * L)
                    Δx̄2 = 0
                    # Update parameters
                    for x̄X_params in zip(Q_net.parameters(), *[Q_j.parameters() for Q_j in Q_θ]):
                        x̄ = x̄X_params[0]
                        X = torch.stack(x̄X_params[1:])
                        x̄_new = torch.einsum("i,i...->...", μ, X)/torch.sum(μ)
                        Δx̄2 += torch.norm(x̄ - x̄_new)**2
                        x̄.copy_(x̄_new)
                        z = torch.zeros(x̄.size())
                        for X_j in x̄X_params[1:]:
                            X_j += (-λ*η+τ*np.sqrt(η)*torch.normal(z))*(X_j-x̄)
                    # Brownian motion if change in V_net params below threshold
                    if (Δx̄2/n_params) < δ:
                        for X_params in zip(*[Q_j.parameters() for Q_j in Q_θ]):
                            z = torch.zeros(X_params[0].size())
                            for X_j in X_params:
                                X_j += τ*np.sqrt(η)*torch.normal(z)
                    if Q_net_comp:
                        e.append(Q_comp(Q_net, Q_net_comp, x_ls, n_comp))
            if Q_net_comp:
                return Q_net, torch.tensor(e)
            else:
                return Q_net
    return algo


def Q_ctrl_UR_CBO_L(Q_θ, A_θ, m, γ, S, A_idx, R, a_s, π, sample):
    s = S[A_θ]
    r = R[A_θ].view(-1)
    a_idx = A_idx[A_θ].type(torch.LongTensor).view(-1)
    s_1 = S[A_θ+1]
    # ŝ_1 = [sample(s_) for s_ in s]
    ŝ_1 = torch.tensor([sample(s_) for s_ in s]).view(-1,1)
    j = torch.cat([(r + γ*torch.max(Q_j(s_1), axis=1).values -
                    Q_j(s)[np.arange(m), a_idx]).view(-1, 1) for Q_j in Q_θ], 1)
    ĵ = torch.cat([(r + γ*torch.max(Q_j(ŝ_1), axis=1).values -
                     Q_j(s)[np.arange(m), a_idx]).view(-1, 1) for Q_j in Q_θ], 1)
    return torch.sum(j*ĵ, 0)/(2*m)


def Q_ctrl_DS_CBO_L(Q_θ, A_θ, m, γ, S, A_idx, R, a_s, π, sample):
    s = S[A_θ]
    r = R[A_θ].view(-1)
    a_idx = A_idx[A_θ].type(torch.LongTensor).view(-1)
    s_1 = S[A_θ+1]
    j = torch.cat([(r + γ*torch.max(Q_j(s_1), axis=1).values -
                    Q_j(s)[np.arange(m), a_idx]).view(-1, 1) for Q_j in Q_θ], 1)
    return torch.sum(j**2, 0)/(2*m)


def Q_ctrl_BFF_CBO_L(Q_θ, A_θ, m, γ, S, A_idx, R, a_s, π, sample):
    s = S[A_θ]
    r = R[A_θ].view(-1)
    a_idx = A_idx[A_θ].type(torch.LongTensor).view(-1)
    s_1 = S[A_θ+1]
    s_2 = S[A_θ+2]
    s̃_1 = s+s_2-s_1
    j = torch.cat([(r + γ*torch.max(Q_j(s_1), axis=1).values -
                    Q_j(s)[np.arange(m), a_idx]).view(-1, 1) for Q_j in Q_θ], 1)
    j̃ = torch.cat([(r + γ*torch.max(Q_j(s̃_1), axis=1).values -
                     Q_j(s)[np.arange(m), a_idx]).view(-1, 1) for Q_j in Q_θ], 1)
    return torch.sum(j*j̃, 0)/(2*m)


def plotQ2(Q_dict, Q_star, lb_star, a_s, x_s=torch.linspace(0, 2*np.pi, 1000)):
    n_r = len(Q_dict)
    fig, axg = plt.subplots(figsize=(12, 8), ncols=3, nrows=n_r)
    y_star = Q_star(x_s.view(-1, 1))
    a_n = len(a_s)
    pad = 5
    for i, (plt_name, (Q_s, e_s, lb_s, c_s, ls_s)) in enumerate(Q_dict.items()):
        n = len(Q_s)
        y_s = torch.stack([Q(x_s.view(-1, 1)) for Q in Q_s])
        y_s -= torch.unsqueeze(torch.mean(y_s - y_star, axis=1), 1)
        axr = axg[i]
        axr[0].annotate(plt_name, xy=(0, 0.5), xytext=(-axr[0].yaxis.labelpad - pad, 0),
                        xycoords=axr[0].yaxis.label, textcoords='offset points',
                        size=15, ha='right', va='center')
        for k in range(a_n):
            axr[k].plot(x_s.detach().numpy(), y_star[:, k].detach(
            ).numpy(), label=lb_star, color="black")
            axr[k].set_ylabel(f"$Q(a_{k},s)$")
            axr[k].set_xlabel(r"$s$")
        for j in range(n):
            for k in range(a_n):
                axr[k].plot(x_s.detach().numpy(),
                            y_s[j, :, k].detach().numpy(), label=lb_s[j])
            axr[a_n].plot(e_s[j]/e_s[j][0], label=lb_s[j])
        axr[a_n].set_yscale("log")
        axr[a_n].set_ylabel(r"$e_k/e_0$ (log scale)")
        axr[a_n].set_xlabel(r"$k$")
    handles, labels = axg[-1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.90)
