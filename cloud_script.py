#!/usr/bin/env python3
# %%
import sys
import pandas as pd
args = sys.argv
# args = "script discrete tabular 0 0 1 0 0 0 0 5 90".split(" ")
_, problem_suffix, model_suffix= args[:3]
resample, reQ, average, landscape,  cuda, n_trials_sgd, n_trials_cbo, n_runs, N = [int(a) for a in args[3:]]
# %%
import torch
from torch.utils.tensorboard import SummaryWriter
from RL_Optimization import *
import optuna
import _pickle as pickle
# %%
if problem_suffix=="continuous":
    from continuous_example import *
    problem_params = get_parameters()
    if model_suffix=="resnet":
        if cuda:
            def new_Q_net(): return Q_ResNet().cuda()
        else:
            def new_Q_net(): return Q_ResNet()
    else:
        if cuda:
            def new_Q_net(): return Q_Net().cuda()
        else:
            def new_Q_net(): return Q_Net()
else:
    from discrete_example import *
    problem_params = get_parameters()
    n_a = problem_params["n_a"]
    n_s = problem_params["n_s"]
    def new_Q_net(): return Q_Tabular(n_a, n_s)
# %%
# import shutil; shutil.rmtree("runs")
writer = SummaryWriter()
if resample:
    print("\n\nRESAMPLING...\n")
    resample_save_policy(problem_suffix)
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
if reQ:
    print("\n\nRECOMPUTING Q_STAR...\n")
    def τ_k(k): return max(0.8*0.9992**k, 0.3)
    Q_ctrl_UR_SGD_star = Q_SGD_gen(Q_ctrl_UR_SGD_update_step)(
        S_long, A_idx_long, R_long, a_s, π, sample, new_Q_net=new_Q_net, M=1000, epochs=1, τ_k=τ_k, x_ls=x_ls, writer=writer, main_tag=f"Q_star recomputation {problem_suffix} {model_suffix}",scalar_tag="Q_star",
    )
    torch.save(Q_ctrl_UR_SGD_star, f"cache/Q_ctrl_UR_SGD_star_{problem_suffix}_{model_suffix}.pt")
Q_ctrl_UR_SGD_star = torch.load(f"cache/Q_ctrl_UR_SGD_star_{problem_suffix}_{model_suffix}.pt")
# %%
M = m = 1000
epochs = 2
δ = 1e-5
# %%
print("\n\nHYPEROPT...\n")
args_0 = [S, A_idx, R, a_s, π, sample]
common_args = {"new_Q_net": new_Q_net, "Q_net_comp": Q_ctrl_UR_SGD_star,  "epochs": epochs, "x_ls": x_ls, "writer":writer}
sgd_u_s = {"UR":Q_ctrl_UR_SGD_update_step, "DS":Q_ctrl_DS_SGD_update_step, "BFF": Q_ctrl_BFF_SGD_update_step}
which=set(["UR"])
# %%
def run_SGD_all(τ_i,τ_f,τ_r,):
    global n_trial
    n_trial += 1
    def τ_k(k):
        return max( τ_i* τ_r** k, τ_f*τ_i)
    sgd_args = {"τ_k": τ_k,"M": M}
    E = [Q_SGD_gen(u_s)(*args_0, **common_args, **sgd_args, main_tag=f"HyperOpt SGD {problem_suffix} {model_suffix}", scalar_tag=f"{s}_{n_trial}")[1] for s,u_s in sgd_u_s.items() if (s in which)]
    return sum([np.log(e_s[-1]) for e_s in E])/len(E)

# %%
cbo_u_s = {"UR": Q_ctrl_UR_CBO_L, "DS": Q_ctrl_DS_CBO_L,"BFF":Q_ctrl_BFF_CBO_L}
def run_CBO_all(η_i,η_f,η_r,τ_i,τ_f,τ_r,β_i,β_f,β_r):
    global n_trial
    n_trial += 1
    def η_k(k):
        return max( η_i* η_r** k, η_f*η_i)

    def τ_k(k):
        return max( τ_i* τ_r** k, τ_f*τ_i)

    def β_k(k):
        return min( β_i* β_r** k, β_f*β_i)

    cbo_args = {"N":N,"m":m,"τ_k":τ_k,"η_k":η_k,"β_k":β_k,"δ":δ}
    E = [Q_CBO_gen(u_s)(*args_0, **common_args, **cbo_args, main_tag=f"HyperOpt CBO {problem_suffix} {model_suffix}", scalar_tag=f"{s}_{n_trial}")[1] for s,u_s in cbo_u_s.items() if (s in which)]
    return sum([np.log(e_s[-1]) for e_s in E])/len(E)
# %%
def objective(trial):
    τ_i = trial.suggest_float("τ_i", 0., 5.)
    τ_f = trial.suggest_float("τ_f", 0., 1.)
    τ_r = trial.suggest_float("τ_r", 0.95, 1.)
    return run_SGD_all(τ_i,τ_f,τ_r)

if n_trials_sgd > 0:
    n_trial = -1
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials_sgd)
    params = study.best_params
    pickle.dump( params, open( f"cache/sgd_params_{problem_suffix}_{model_suffix}.p", "wb" ) )
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(f"figs/Q_ctrl_SGD_hyperopt_history_{problem_suffix}_{model_suffix}.png", engine="kaleido")
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(f"figs/Q_ctrl_SGD_hyperopt_parallel_plot_{problem_suffix}_{model_suffix}.png", engine="kaleido")
# %%
def objective(trial):
    η_i = trial.suggest_float("η_i", 0., 1.)
    η_f = trial.suggest_float("η_f", 0., 1.)
    η_r = trial.suggest_float("η_r", 0.95, 1.)
    τ_i = trial.suggest_float("τ_i", 0., 3.)
    τ_f = trial.suggest_float("τ_f", 0., 1.)
    τ_r = trial.suggest_float("τ_r", 0.95, 1.)
    β_i = trial.suggest_float("β_i", 5., 15.)
    β_f = trial.suggest_float("β_f", 1., 3.)
    β_r = trial.suggest_float("β_r", 1., 1.05)
    return run_CBO_all(η_i,η_f,η_r,τ_i,τ_f,τ_r,β_i,β_f,β_r)

if n_trials_cbo>0:
    n_trial = -1
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials_cbo)
    params = study.best_params
    pickle.dump( params, open( f"cache/cbo_params_{problem_suffix}_{model_suffix}.p", "wb" ) )
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(f"figs/Q_ctrl_CBO_hyperopt_history_{problem_suffix}_{model_suffix}.png", engine="kaleido")
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(f"figs/Q_ctrl_CBO_hyperopt_parallel_plot_{problem_suffix}_{model_suffix}.png", engine="kaleido")

# %%
print("\n\nAVERAGING RESULTS...\n")
if average:
    params = pickle.load(open(f"cache/sgd_params_{problem_suffix}_{model_suffix}.p", "rb"))
    τ_i, τ_f, τ_r = [params[x] for x in ['τ_i', 'τ_f', 'τ_f']]
    def τ_k(k):
        return max(τ_i * τ_r ** k, τ_f*τ_i)
    sgd_args = {"τ_k": τ_k, "M": M}
    def run_SGD_all():
        global n_run
        Qs, es = [], []
        for s,u_s in sgd_u_s.items():
            q, e = Q_SGD_gen(u_s)(
                *args_0, **common_args, **sgd_args,main_tag=f"Averaging SGD {problem_suffix} {model_suffix}", scalar_tag=f"{s}_{n_run}")
            Qs.append(q)
            es.append(e)
        return (Qs, es)
    params = pickle.load(open(f"cache/cbo_params_{problem_suffix}_{model_suffix}.p", "rb"))
    η_i, η_f, η_r, τ_i, τ_f, τ_r, β_i, β_f, β_r = [params[x] for x in [
        'η_i', 'η_f', 'η_r', 'τ_i', 'τ_f', 'τ_r', 'β_i', 'β_f', 'β_r']]

    def η_k(k): return max(η_i * η_r ** k, η_f*η_i)
    def τ_k(k): return max(τ_i * τ_r ** k, τ_f*τ_i)
    def β_k(k): return min(β_i * β_r ** k, β_f*β_i)

    cbo_args = {"N": N, "m": m, "τ_k": τ_k, "η_k": η_k,
                "β_k": β_k, "δ": δ}

    def run_CBO_all():
        global n_run
        Qs, es = [], []
        for s, u_s in cbo_u_s.items():
            q, e = Q_CBO_gen(u_s)(
                *args_0, **common_args, **cbo_args,main_tag=f"Averaging CBO {problem_suffix} {model_suffix}", scalar_tag=f"{s}_{n_run}")
            Qs.append(q)
            es.append(e)
        return (Qs, es)

    all_data = pd.DataFrame(columns=["i", "x", "y", "sampling", "algo", "plot"])
    y_star = Q_ctrl_UR_SGD_star(x_ls.view(-1, 1))
    a_n = len(a_s)
    for j, lb in enumerate(["UR", "DS", "BFF"]):
        for k in range(a_n):
            all_data = all_data.append(
                pd.DataFrame(
                    {
                        "i": 0,
                        "x": x_ls.detach().numpy(),
                        "y": y_star[:, k].detach().numpy(),
                        "sampling": lb,
                        "algo": "UR SGD *",
                        "plot": f"$Q(a_{k},s)$",
                    }
                )
            )
    for n_run in range(n_runs):
        for r, algo in [(run_SGD_all, "SGD"), (run_CBO_all, "CBO")]:
            Q_s, e_s = r()
            y_s = torch.stack([Q(x_ls.view(-1, 1)) for Q in Q_s])
            y_s -= torch.unsqueeze(torch.mean(y_s - y_star, axis=1), 1)
            for j, lb in enumerate(["UR", "DS", "BFF"]):
                for k in range(a_n):
                    all_data = all_data.append(
                        pd.DataFrame(
                            {
                                "i": n_run,
                                "x": x_ls.detach().numpy(),
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
                            "i": n_run,
                            "x": np.arange(len(e_s[j])),
                            "y": e_s[j] / e_s[j][0],
                            "sampling": lb,
                            "algo": algo,
                            "plot": r"$e_k/e_0$",
                        }
                    )
                )

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
    [ax.set_ylim(None, 1) for ax in g.axes[:,-1]]
    [plt.setp(ax.texts, text="") for ax in g.axes.flat]
    g.set_ylabels("")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    g.add_legend(title="")
    plt.savefig(f"figs/Q_ctrl_SGD_vs_CBO_summary_{problem_suffix}_{model_suffix}.png")
# %%
if landscape:
    all_data = pd.DataFrame(columns=["i", "x", "y", "problem", "model"])
    Q_net_comp = common_args["Q_net_comp"]
    n = len(x_ls)
    N = int(1e2)
    δα = 1./N
    α_ls = np.linspace(0, 1, N)
    n_runs = 100
    for i in range(n_runs):
        Q_net, Q_net_0 = new_Q_net(), new_Q_net()
        e = []
        for α in α_ls:
            with torch.no_grad():
                for param, param_i, param_f in zip(Q_net.parameters(), Q_net_0.parameters(), Q_net_comp.parameters()):
                    param += α * param_f + (1-α) * param_i - param
            e.append(Q_comp(Q_net, Q_net_comp, x_ls.view(-1,1), n).detach().numpy().item())
        all_data = all_data.append(
            pd.DataFrame(
                {
                    "i": i,
                    "x": α_ls,
                    "y": e,
                    "problem": problem_suffix,
                    "model": model_suffix
                }
            )
        )
    ax = sns.lineplot(data=all_data, x="x", y="y", units="i",
                alpha=0.7, hue="problem", estimator=None)
    ax.figure.savefig(f"figs/Q_ctrl_landscape_plot_{problem_suffix}_{model_suffix}.png")
# %%
writer.close()
