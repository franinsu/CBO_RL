# %%
import torch
from RL_Optimization import *
import optuna
import pickle
# %%
# problem_suffix= "continuous"
# model_suffix = "resnet"
problem_suffix= "discrete"
model_suffix = "tabular"
# %%
if problem_suffix=="continuous":
    from continuous_example import *
    problem_params = get_parameters()
    if model_suffix=="resnet":
        def new_Q_net(): return Q_ResNet()
    else:
        def new_Q_net(): return Q_Net()
else:
    from discrete_example import *
    problem_params = get_parameters()
    n_a = problem_params["n_a"]
    n_s = problem_params["n_s"]
    def new_Q_net(): return Q_Tabular(n_a, n_s)
# %%
# resample_save_policy(problem_suffix)
# %%
S_long = torch.load(f"cache/S_long_{problem_suffix}.pt")
R_long = torch.load(f"cache/R_long_{problem_suffix}.pt")
A_idx_long = torch.load(f"cache/A_idx_long_{problem_suffix}.pt")
S = torch.load(f"cache/S_{problem_suffix}.pt")
R = torch.load(f"cache/R_{problem_suffix}.pt")
A_idx = torch.load(f"cache/A_idx_{problem_suffix}.pt")
# %%
sample = sampler()
a_s = problem_params["a_s"]
x_ls = problem_params["x_ls"]
# %%
# def η_k(k): return max(0.1*0.9992**k, 0.075)
# def τ_k(k): return max(0.8*0.9992**k, 0.3)
# def β_k(k): return min(8*1.002**k,20)
# Q_ctrl_UR_SGD_star = Q_SGD_gen(Q_ctrl_UR_SGD_update_step)(
#     S_long, A_idx_long, R_long, a_s, π, sample, new_Q_net=new_Q_net, M=1000, epochs=1, τ_k=τ_k, x_ls=x_ls
# )
# torch.save(Q_ctrl_UR_SGD_star, f"cache/Q_ctrl_UR_SGD_star_{problem_suffix}_{model_suffix}.pt")
Q_ctrl_UR_SGD_star = torch.load(f"cache/Q_ctrl_UR_SGD_star_{problem_suffix}_{model_suffix}.pt")
# %%
M = 1000
epochs = 1
args_0 = [S, A_idx, R, a_s, π, sample]
common_args = {"new_Q_net": new_Q_net, "Q_net_comp": Q_ctrl_UR_SGD_star,  "epochs": epochs, "x_ls": x_ls}
sgd_u_s = [Q_ctrl_UR_SGD_update_step, Q_ctrl_DS_SGD_update_step, Q_ctrl_BFF_SGD_update_step]
which=[1,0,0]
# %%
def run_SGD_all(τ_i,τ_f,τ_r,):
    def τ_k(k):
        return max( τ_i* τ_r** k, τ_f*τ_i)
    sgd_args = {"τ_k": τ_k,"M": M}
    E = [Q_SGD_gen(u_s)(*args_0, **common_args, **sgd_args)[1] for i,u_s in enumerate(sgd_u_s) if which[i]]
    return sum([np.log(e_s[-1] / e_s[0]) for e_s in E])/len(E)

# %%
N = 30
m = 1000
epochs = 1
δ = 1e-5
early_stop = 1000
cbo_u_s = [Q_ctrl_UR_CBO_L, Q_ctrl_DS_CBO_L,Q_ctrl_BFF_CBO_L]
# %%
def run_CBO_all(η_i,η_f,η_r,τ_i,τ_f,τ_r,β_i,β_f,β_r):
    def η_k(k):
        return max( η_i* η_r** k, η_f*η_i)

    def τ_k(k):
        return max( τ_i* τ_r** k, τ_f*τ_i)

    def β_k(k):
        return min( β_i* β_r** k, β_f*β_i)

    cbo_args = {"N":N,"m":m,"τ_k":τ_k,"η_k":η_k,"β_k":β_k,"δ":δ,"early_stop":early_stop}
    E = [Q_CBO_gen(u_s)(*args_0, **common_args, **cbo_args)[1] for i,u_s in enumerate(cbo_u_s) if which[i]]
    return sum([np.log(e_s[-1] / e_s[0]) for e_s in E])/len(E)
# %%
def objective(trial):
    τ_i = trial.suggest_float("τ_i", 0., 1.)
    τ_f = trial.suggest_float("τ_f", 0., 1.)
    τ_r = trial.suggest_float("τ_r", 0.9, 1.)
    return run_SGD_all(τ_i,τ_f,τ_r)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)
# %%
params = study.best_params
pickle.dump( params, open( f"cache/sgd_params_{problem_suffix}_{model_suffix}.p", "wb" ) )
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
study.optimize(objective, n_trials=5)
# %%
params = study.best_params
pickle.dump( params, open( f"cache/cbo_params_{problem_suffix}_{model_suffix}.p", "wb" ) )
# %%
optuna.visualization.plot_optimization_history(study)
# %%
optuna.visualization.plot_parallel_coordinate(study)

# %%
