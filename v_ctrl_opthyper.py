# %%
import torch
from helper import *
from IPython import get_ipython
import torch.nn as nn
import pandas as pd
import seaborn as sns
import optuna
import pickle

# %%
def π(a, s):
    return torch.tensor([1 / 2.0, 1 / 2.0]) * torch.ones_like(s)

def r(s):
    return torch.sin(s) + 1

σ = 0.2
ϵ = 2.0 * np.pi / 32.0
s_0 = torch.tensor([0.0])
a_s = torch.tensor([-1.0, 1.0])
# %%
def new_Q_net(): return Q_ResNet()
# %%
S_long = torch.load("cache/S_long_2.pt")
R_long = torch.load("cache/R_long_2.pt")
A_idx_long = torch.load("cache/A_idx_long_2.pt")
S = torch.load("cache/S_2.pt")
R = torch.load("cache/R_2.pt")
A_idx = torch.load("cache/A_idx_2.pt")
Q_ctrl_UR_SGD_star = torch.load("cache/Q_ctrl_UR_SGD_star.pt")
# %%
M = 1000
epochs = 1
# %%
def run_SGD_all(τ_i,τ_f,τ_r):
    def τ_k(k):
        return max( τ_i* τ_r** k, τ_f*τ_i)
    Q_ctrl_UR_SGD, e_ctrl_UR_SGD = Q_SGD_gen(Q_ctrl_UR_SGD_update_step)(
        S, A_idx, R, a_s, π, σ, ϵ,new_Q_net=new_Q_net,
        M=M, epochs=epochs, τ_k=τ_k, Q_net_comp=Q_ctrl_UR_SGD_star
    )
    # Q_ctrl_DS_SGD, e_ctrl_DS_SGD = Q_SGD_gen(Q_ctrl_DS_SGD_update_step)(
    #     S, A_idx, R, a_s, π, σ, ϵ,new_Q_net=new_Q_net,
    #     M=M, epochs=epochs, τ_k=τ_k, Q_net_comp=Q_ctrl_UR_SGD_star
    # )
    # Q_ctrl_BFF_SGD, e_ctrl_BFF_SGD = Q_SGD_gen(Q_ctrl_BFF_SGD_update_step)(
    #     S, A_idx, R, a_s, π, σ, ϵ, new_Q_net=new_Q_net,
    #     M=M, epochs=epochs, τ_k=τ_k, Q_net_comp=Q_ctrl_UR_SGD_star
    # )
    # return sum([np.log(e_s[-1] / e_s[0]) for e_s in [e_ctrl_UR_SGD, e_ctrl_DS_SGD, e_ctrl_BFF_SGD]]) 
    return sum([np.log(e_s[-1] / e_s[0]) for e_s in [e_ctrl_UR_SGD]]) 

# %%
N = 30
m = 1000
epochs = 1
δ = 1e-5
early_stop = 1000
# %%
def run_CBO_all(η_i,η_f,η_r,τ_i,τ_f,τ_r,β_i,β_f,β_r):
    def η_k(k):
        return max( η_i* η_r** k, η_f*η_i)

    def τ_k(k):
        return max( τ_i* τ_r** k, τ_f*τ_i)

    def β_k(k):
        return min( β_i* β_r** k, β_f*β_i)

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
    # Q_ctrl_DS_CBO, e_ctrl_DS_CBO = Q_CBO_gen(Q_ctrl_DS_CBO_L)(
    #     S,
    #     A_idx,
    #     R,
    #     a_s,
    #     π,
    #     σ,
    #     ϵ,
    #     new_Q_net=new_Q_net,
    #     N=N,
    #     m=m,
    #     epochs=epochs,
    #     τ_k=τ_k,
    #     η_k=η_k,
    #     β_k=β_k,
    #     δ=δ,
    #     Q_net_comp=Q_ctrl_UR_SGD_star,
    #     early_stop=early_stop,
    # )
    # Q_ctrl_BFF_CBO, e_ctrl_BFF_CBO = Q_CBO_gen(Q_ctrl_BFF_CBO_L)(
    #     S,
    #     A_idx,
    #     R,
    #     a_s,
    #     π,
    #     σ,
    #     ϵ,
    #     new_Q_net=new_Q_net,
    #     N=N,
    #     m=m,
    #     epochs=epochs,
    #     τ_k=τ_k,
    #     η_k=η_k,
    #     β_k=β_k,
    #     δ=δ,
    #     Q_net_comp=Q_ctrl_UR_SGD_star,
    #     early_stop=early_stop,
    # )
    # return sum([np.log(e_s[-1] / e_s[0]) for e_s in [e_ctrl_UR_CBO, e_ctrl_DS_CBO, e_ctrl_BFF_CBO]])
    return sum([np.log(e_s[-1] / e_s[0]) for e_s in [e_ctrl_UR_CBO]])
# %%
def objective(trial):
    τ_i = trial.suggest_float("τ_i", 0., 1.)
    τ_f = trial.suggest_float("τ_f", 0., 1.)
    τ_r = trial.suggest_float("τ_r", 0.9, 1.)
    return run_SGD_all(τ_i,τ_f,τ_r)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15)
# %%
params = study.best_params
pickle.dump( params, open( "cache/sgd_params.p", "wb" ) )
# %%
optuna.visualization.plot_optimization_history(study)
# %%
optuna.visualization.plot_parallel_coordinate(study)
# %%
def objective(trial):
    η_i = trial.suggest_float("η_i", 0., 1.)
    η_f = trial.suggest_float("η_f", 0., 1.)
    η_r = trial.suggest_float("η_r", 0.9, 1.)
    τ_i = trial.suggest_float("τ_i", 0., 1.)
    τ_f = trial.suggest_float("τ_f", 0., 1.)
    τ_r = trial.suggest_float("τ_r", 0.9, 1.)
    β_i = trial.suggest_float("β_i", 0., 1.)
    β_f = trial.suggest_float("β_f", 0., 1.)
    β_r = trial.suggest_float("β_r", 0.9, 1.)
    return run_CBO_all(η_i,η_f,η_r,τ_i,τ_f,τ_r,β_i,β_f,β_r)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15)
# %%
params = study.best_params
pickle.dump( params, open( "cache/cbo_params.p", "wb" ) )
# %%
optuna.visualization.plot_optimization_history(study)
# %%
optuna.visualization.plot_parallel_coordinate(study)

# %%
