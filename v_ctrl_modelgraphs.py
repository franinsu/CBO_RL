# %%
from RL_Optimization import *
from torchviz import make_dot
import pickle
# %%
Q = Q_ResNet()
S = torch.load("cache/S_continuous.pt")
dot = make_dot(Q(S[:20]))
dot.format = 'png'
dot.render("figs/Q_ResNet_graph")
# %%
Q = Q_Tabular(2, 32)
S = torch.load("cache/S_discrete.pt")
dot = make_dot(Q(S[:20]))
dot.format = 'png'
dot.render("figs/Q_Tabular_graph")
# %%
for case in ["continuous_resnet", "discrete_tabular"]:
    for opt in ["sgd","cbo"]:
        params = pickle.load(open(f"cache/{opt}_params_{case}.p", "rb"))
        print(opt, case)
        for p,v in params.items():
            print(f"{p}:{v}")
        print(20*"=")
# %%
