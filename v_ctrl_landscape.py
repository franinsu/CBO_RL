# %%
import torch
from RL_Optimization import *
import pandas as pd
import seaborn as sns
# %%
all_data = pd.DataFrame(columns=["i", "x", "y", "problem", "model"])
# for problem_suffix, model_suffix in zip(["continuous", "discete"], ["resnet", "tabular"]):
for problem_suffix, model_suffix in zip(["continuous"], ["resnet"]):
    if problem_suffix == "continuous":
        from continuous_example import *
        if model_suffix == "resnet":
            def new_Q_net(): return Q_ResNet()
        else:
            def new_Q_net(): return Q_Net()
    else:
        from discrete_example import *
        def new_Q_net(): return Q_Tabular()
    Q_net_comp = torch.load(
        f"cache/Q_ctrl_UR_SGD_star_{problem_suffix}_{model_suffix}.pt")
    x_ls = get_parameters()["x_ls"].view(-1, 1)
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
            e.append(Q_comp(Q_net, Q_net_comp, x_ls, n).detach().numpy().item())
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

# %%
sns.lineplot(data=all_data, x="x", y="y", units="i",
             alpha=0.7, hue="problem", estimator=None)
# %%
