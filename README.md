# CBO Applied to RL


```python
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import seaborn as sns
import matplotlib.colors as colors
%matplotlib inline
plt.style.use('default')
# plt.style.use('seaborn-darkgrid')
mpl.rcParams['figure.dpi'] = 120
```


```python
# hacky way of toggling between showing/hiding progress bar

# show
from tqdm.auto import tqdm, trange
# hide
# trange = lambda x, leave=None, desc=None: range(x)
# tqdm = lambda x, leave=None, desc=None: x
```

## Continuous state space (5.1.)

### Outline

- Markov decision process with a continuous state space $\mathbb S = \{s \in (0, 2\pi]\}$.
- Dynamics described by 

$$\begin{aligned}
&s_{m+1} = s_{m} + \alpha(s_m)\epsilon + \sigma(s_m)\sqrt{\epsilon} Z_m\\
&\alpha(s) = 2 \sin(s) \cos(s), \quad \sigma(s) = 1 + \cos(s)^2, \quad \epsilon = 0.1
\end{aligned}$$

- Immediate reward function, $R(s)=(\cos(2s)+1)$.
- Discount factor $\gamma=0.9$.
- 3 layer FCNN, $V(s;\theta)$. Two hidden layers with $\cos$ activation function, and each hidden layer contains 50 neurons.

$$\begin{aligned}
&V(s ; \theta)=V\left(x ;\left\{w_{i}, b_{i}\right\}_{i=1}^{3}\right)=L_{w_{3}, b_{3}} \circ \cos \circ L_{w_{2}, b_{2}} \circ \cos \circ L_{w_{1}, b_{1}}((\cos s, \sin s)) \\
&L_{w_{i}, b_{i}}(x)=w_{i} x+b_{i}, \quad w_{i} \in \mathbb{R}^{n_{i-1} \times n_{i}}, \quad b_{i} \in \mathbb{R}^{n_{i}}, \quad n_{0}=2, n_{1}=n_{2}=50, n_{3}=1
\end{aligned}$$

- $\theta^*$ is computed with Algorithms 1-4 based on trajectory $\{s_m\}_{m=1}^{10^6}$ with

$$
f\left(s_{m}, s_{m+1}, \theta\right)=R\left(s_{m}\right)+\gamma V\left(s_{m+1} ; \theta\right)-V\left(s_{m} ; \theta\right), \quad \tau=0.1, \quad M=1000
$$

- The SGD algorithm runs for a single epoch with the same initialization $\theta_0$. 
- Error at step $k$, $e_k$ is defined as $e_k=\| V(\cdot,\theta_k)-V^*\|_{L^2}$. 
- Reference $V^*(s)$ is computed by running Algorithm 1 for 10 epochs based on longer trajectory $\{s_m\}_{m=1}^{10^7}$, with $\tau=0.01$, $M=1000$.
- Visualize $\log_{10}(e_k/e_0)$.

Made one modification to paper:
- Since $V(s,\theta)\mapsto V(s,\theta)+\delta$ is a symmetry in $f$, then, a better way of measuring error, $e_k$, is 
$$
e_k=\| V(\cdot,\theta_k)-V^* - \mu_k \|_2,\quad \mu_k = \int V(\cdot,\theta_k)-V^*
$$

### Sampling Trajectory


```python
def sample_trajectory(s_0, α, σ, ϵ, m):
    """
    samples trajectory of a particle following dSₜ = α(t) dt + σ(t) Wₜ, approximated by discrete units of time.
    
    parameters:
    s_0  : Starting state of particle
    α    : Drift function
    σ:   Variance function
    ϵ:   Step size
    m:   Number of steps to sample
    
    returns:
    S:   torch.tensor of sampled trajectory, with size (m, *s_0.size())
    """
    s_0_size = s_0.size()
    S = torch.zeros(m, *s_0_size)
    S[0] = s_0
    for i in range(m-1):
        S[i+1] = S[i]+α(S[i])*ϵ+σ(S[i])*torch.sqrt(ϵ)*torch.normal(mean=torch.zeros(*s_0_size))
    return S
```


```python
α = lambda s: 2*torch.sin(s)*torch.cos(s) 
σ = lambda s: 1+torch.cos(s)**2
ϵ = torch.tensor([0.1])
s_0 = torch.tensor([0.])
r = lambda s: torch.cos(2*s)+1
```


```python
# S = sample_trajectory(s_0, α, σ, ϵ, int(1e6))
# R = r(S) 
# S_long = sample_trajectory(s_0, α, σ, ϵ, int(1e7))
# R_long = r(S_long) 
```


```python
# torch.save(S_long, 'S_long.pt')
# torch.save(R_long, 'R_long.pt')
# torch.save(S, 'S.pt')
# torch.save(R, 'R.pt')
```


```python
S_long = torch.load('S_long.pt')
R_long = torch.load('R_long.pt')
S = torch.load('S.pt')
R = torch.load('R.pt')
```


```python
plt.plot(S,".",markersize=0.1);
plt.xlabel(r"$i$");
plt.ylabel(r"$s_i$");
plt.show();
plt.plot(torch.remainder(S,2*np.pi),".",markersize=0.1,alpha=0.1);
plt.xlabel(r"$i$");
plt.ylabel(r"$s_i$");
plt.show();
plt.plot(R,".",markersize=0.1,alpha=0.1);
plt.xlabel(r"$i$");
plt.ylabel(r"$R(s_i)$");
plt.show();
```


![png](CBO_RL_files/CBO_RL_13_0.png)



![png](CBO_RL_files/CBO_RL_13_1.png)



![png](CBO_RL_files/CBO_RL_13_2.png)


### Algos 1-4


```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 50) 
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.cat((torch.sin(x), torch.cos(x)), 1)
        x = torch.cos(self.fc1(x))
        x = torch.cos(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
```


```python
def gen_batches(N, M, Rem):
    """
    generate random batches of indices, keeping track of remainders
    
    parameters:
    N   : number of elements to choose from
    M   : size of batches
    Rem : previous remainder
    
    returns:
    B   : tensor of batches, of size q,M 
    Rem : new remainder
    """
    P = torch.randperm(N)
    if Rem.size()[0] > 0:
        I = torch.cat((Rem, P), 0)
    else:
        I = P
    q = int(np.floor((P.size()[0]+Rem.size()[0])/M))
    B = I[:(q*M)].view(q, M)
    Rem = I[(q*M):]
    return B, Rem
```


```python
def comp(V_net,V_net_comp,x_ls, n):
    diff = V_net(x_ls)-V_net_comp(x_ls)
    return np.sqrt(2*np.pi/n)*torch.norm(diff-torch.mean(diff))
def comp_0(V_net,V_net_comp,x_ls, n):
    return np.sqrt(2*np.pi/n)*torch.norm(V_net(x_ls)-V_net_comp(x_ls))
```


```python
def algorithm_1(V_net, M, epochs,γ,τ, S, R, α,σ,ϵ, V_net_comp=None, n=100):
    Rem = torch.tensor([])
    N = S.size()[0]
    if V_net_comp:
        x_ls = torch.linspace(0,2*np.pi,n+1)[:-1].view(-1,1)
        e = [comp(V_net,V_net_comp,x_ls, n)]
    for k in trange(epochs, leave=False,desc="Epoch"):
        B, Rem = gen_batches(N-1,M, Rem)
        for i,B_θ in enumerate(tqdm(B,leave=False,desc="Batch")): 
            s = S[B_θ]
            s_1 = S[B_θ+1]
            s_p1 = s+α(s)*ϵ+σ(s)*torch.sqrt(ϵ)*torch.normal(mean=torch.zeros_like(s))
            f_p = R[B_θ] + γ*V_net(s_p1) - V_net(s)
            f = R[B_θ] + γ*V_net(s_1) - V_net(s)
            V_net.zero_grad()
            f_p.backward(f)
            with torch.no_grad():
                for param in V_net.parameters():
                    param -= τ/M*param.grad
            if V_net_comp:
                e.append(comp(V_net,V_net_comp,x_ls, n))
    if V_net_comp:
        return V_net, torch.tensor(e)
    else:
        return V_net
```


```python
def algorithm_2(V_net, M, epochs,γ,τ, S, R, V_net_comp=None, n=100):
    Rem = torch.tensor([])
    N = S.size()[0]
    if V_net_comp:
        x_ls = torch.linspace(0,2*np.pi,n+1)[:-1].view(-1,1)
        e = [comp(V_net,V_net_comp,x_ls, n)]
    for k in trange(epochs, leave=False, desc="Epoch"):
        B, Rem = gen_batches(N-1,M, Rem)
        for i, B_θ in enumerate(tqdm(B,leave=False,desc="Batch")):
            s = S[B_θ]
            s_1 = S[B_θ+1]
            f = R[B_θ] + γ*V_net(s_1) - V_net(s)
            V_net.zero_grad()
            f.backward(f)
            with torch.no_grad():
                for param in V_net.parameters():
                    param -= τ/M*param.grad
            if V_net_comp:
                e.append(comp(V_net,V_net_comp,x_ls, n))
    if V_net_comp:
        return V_net, torch.tensor(e)
    else:
        return V_net
```


```python
def algorithm_3(V_net, M, epochs,γ,τ, S, R, V_net_comp=None, n=100):
    Rem = torch.tensor([])
    N = S.size()[0]
    if V_net_comp:
        x_ls = torch.linspace(0,2*np.pi,n+1)[:-1].view(-1,1)
        e = [comp(V_net,V_net_comp,x_ls, n)]
    for k in trange(epochs, leave=False, desc="Epoch"):
        B, Rem = gen_batches(N-2,M, Rem)
        for i, B_θ in enumerate(tqdm(B,leave=False,desc="Batch")):
            s = S[B_θ]
            s_1 = S[B_θ+1]
            s_2 = S[B_θ+2]
            s̃_1 = s + s_2 - s_1 
            f = R[B_θ] + γ*V_net(s_1) - V_net(s)
            f̃ = R[B_θ] + γ*V_net(s̃_1) - V_net(s)
            V_net.zero_grad()
            (f*f̃).backward(torch.ones_like(f)/2.)
            with torch.no_grad():
                for param in V_net.parameters():
                    param -= τ/M*param.grad
            if V_net_comp:
                e.append(comp(V_net,V_net_comp,x_ls, n))
    if V_net_comp:
        return V_net, torch.tensor(e)
    else:
        return V_net
```


```python
def algorithm_4(V_net, M, epochs,γ,τ, S, R, V_net_comp=None, n=100):
    Rem = torch.tensor([])
    N = S.size()[0]
    if V_net_comp:
        x_ls = torch.linspace(0,2*np.pi,n+1)[:-1].view(-1,1)
        e = [comp(V_net,V_net_comp,x_ls, n)]
    for k in trange(epochs, leave=False, desc="Epoch"):
        B, Rem = gen_batches(N-2,M, Rem)
        for i, B_θ in enumerate(tqdm(B,leave=False,desc="Batch")):
            s = S[B_θ]
            s_1 = S[B_θ+1]
            s_2 = S[B_θ+2]
            s̃_1 = s + s_2 - s_1 
            f = R[B_θ] + γ*V_net(s_1) - V_net(s)
            f̃ = R[B_θ] + γ*V_net(s̃_1) - V_net(s)
            V_net.zero_grad()
            f̃.backward(f)
            with torch.no_grad():
                for param in V_net.parameters():
                    param -= τ/M*param.grad
            if V_net_comp:
                e.append(comp(V_net,V_net_comp,x_ls, n))
    if V_net_comp:
        return V_net, torch.tensor(e)
    else:
        return V_net
```


```python
# V_star = Net()
# M = 1000
# epochs = 10
# γ = 0.9
# τ = 0.01
# V_star = algorithm_1(V_star,M,epochs,γ,τ,S_long,R_long,α,σ,ϵ)
```


```python
# torch.save(V_star, "V_star.pt")
```


```python
V_star = torch.load("V_star.pt")
```


```python
M = 1000
epochs = 1
γ = 0.9
τ = 0.1
n = 100
V_1 = Net()
V_1,e_1 = algorithm_1(V_1,M,epochs,γ,τ,S,R,α,σ,ϵ, V_star, n)
V_2 = Net()
V_2,e_2 = algorithm_2(V_2,M,epochs,γ,τ,S,R,V_star,n)
V_3 = Net()
V_3,e_3 = algorithm_3(V_3,M,epochs,γ,τ,S,R,V_star,n)
V_4 = Net()
V_4,e_4 = algorithm_4(V_4,M,epochs,γ,τ,S,R,V_star,n)
```


    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=999.0, style=ProgressStyle(description_width=…


    


    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=999.0, style=ProgressStyle(description_width=…


    


    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=999.0, style=ProgressStyle(description_width=…


    


    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=999.0, style=ProgressStyle(description_width=…


    


```python
fig, axs = plt.subplots(figsize=(12,5),ncols=2)
x_ls = torch.linspace(0,2*np.pi,100)
y_star_ls = V_star(x_ls.view(-1,1)).view(-1)
y_ls = torch.cat([V(x_ls.view(-1,1)).view(1,-1) for V in [V_1,V_2,V_3,V_4]])
y_ls -= torch.mean(y_ls-y_star_ls.view(1,-1),axis=1).view(-1,1)
e_ls = [e_1,e_2,e_3,e_4]
colors = ["tab:blue", "tab:orange","tab:green","tab:red"]
labels = ["Algo 1", "Algo 2", "Algo 3", "Algo 4"]
line_styles = ['solid','solid',(5,(5,5)),(0,(5,5))]
axs[0].plot(x_ls.detach().numpy(),y_star_ls.detach().numpy(),label="Algo 1 *",color="black")
for j in range(0,len(y_ls)):
    axs[0].plot(x_ls.detach().numpy(),y_ls[j].detach().numpy(),label=labels[j],color=colors[j], ls=line_styles[j])
    axs[1].plot(e_ls[j]/e_ls[j][0], label=labels[j],color=colors[j])
axs[0].set_ylabel(r"$V(s)$")
axs[0].set_xlabel(r"$s$")
axs[1].set_yscale("log")
axs[1].set_ylabel(r"$e_k/e_0$ (log scale)")
axs[1].set_xlabel(r"$k$")
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right')
plt.tight_layout()
plt.subplots_adjust(right=0.9)
```


![png](CBO_RL_files/CBO_RL_26_0.png)


### CBO


```python
def algorithm_CBO(V_net, V_s, S, M, m, epochs, γ,λ,σ,η,β, V_net_comp=None, n_comp=100):
    '''
    Implementation of CBO algorithm to minimize J = E[(1/2)E[f(s_{m},s_{m+1};θ)|s_{m}]^2]
    
    Parameters:
        V_net      : Net class 
        V_s        : List of Nets (corresponding to particles in CBO)
        S          : Sample trajectory
        M          : Size of batches in X
        m          : Size of batches in S (set to S.size()[0] for no approximations)
        epochs     : Number of epochs
        γ          : Discount factor
        λ          : Exploitation rate
        σ          : Exploration rate
        η          : Learning rate          
        β          : Characteristic temperatures
        V_net_comp : Net to compare V_net to
        n_comp     : Number of points to use to compare V_net and V_net_comp
        
    Returns:
        V_net      : Net class
        e          : Errors, computed each batch
    '''
    with torch.no_grad():
        Rem = torch.tensor([])
        L = torch.empty(M)
        N = len(V_s)
        n = S.size()[0]
        if V_net_comp:
            x_ls = torch.linspace(0,2*np.pi,n+1)[:-1].view(-1,1)
            e = [comp(V_net,V_net_comp,x_ls, n_comp)]
        for k in trange(epochs, leave=False, desc="Epoch"):
            B, Rem = gen_batches(N,M, Rem)
            for i, B_θ in enumerate(tqdm(B,leave=False,desc="Batch")):
                A_θ = torch.randperm(n-1)[:m]
                s = S[A_θ]
                s_1 = S[A_θ+1]
                V_θ = [V_j for j,V_j in enumerate(V_s) if j in B_θ]
                f = torch.cat([R[A_θ] + γ*V_j(s_1)-V_j(s) for V_j in V_θ],1)
                L = torch.sum(f**2 / 2.,0)/m
                mu = torch.exp(-β*L)
                for xX in zip(V_net.parameters(),*[V_j.parameters() for V_j in V_θ]):
                    xX_tensor = torch.cat([xX_j.view(1,-1) for xX_j in xX])
                    x_new = xX_tensor[0]
                    X_new = xX_tensor[1:]
                    x_new = mu.view(1,-1).mm(X_new)/torch.sum(mu)
                    # Particle-wise noise              
                    #  X_new += -λ*η*(X_new-x_new)+σ*np.sqrt(η)*torch.diag(torch.normal(torch.zeros(M))).mm(X_new-x_new)
                    # Parameter-particle-wise noise
                    X_new += -λ*η*(X_new-x_new)+σ*np.sqrt(η)*torch.normal(torch.zeros(X_new.size()))*(X_new-x_new)
                    xX_new = torch.cat((x_new,X_new)).view(-1, *xX[0].size())
                    for xX_j, xX_new_j in zip(xX,xX_new):
                        xX_j +=  xX_new_j - xX_j
                if V_net_comp:
                    e.append(comp(V_net,V_net_comp,x_ls, n_comp))
        if V_net_comp:
            return V_net, torch.tensor(e)
        else:
            return V_net
```


```python
def algorithm_CBO_BFF(V_net, V_s, S, M, m, epochs, γ,λ,σ,η,β, V_net_comp=None, n_comp=100):
    '''
    Implementation of CBO algorithm to minimize J̃ = E[(1/2)E[f(s_{m},s_{m+1};θ)|s_{m}]E[f(s_{m},s̃_{m+1};θ)|s_{m}]]
    
    Parameters:
        V_net      : Net class 
        V_s        : List of Nets (corresponding to particles in CBO)
        S          : Sample trajectory
        M          : Size of batches in X
        m          : Size of batches in S (set to S.size()[0] for no approximations)
        epochs     : Number of epochs
        γ          : Discount factor
        λ          : Exploitation rate
        σ          : Exploration rate
        η          : Learning rate          
        β          : Characteristic temperatures
        V_net_comp : Net to compare V_net to
        n_comp     : Number of points to use to compare V_net and V_net_comp
        
    Returns:
        V_net      : Net class
        e          : Errors, computed each batch
    '''
    with torch.no_grad():
        Rem = torch.tensor([])
        L = torch.empty(M)
        N = len(V_s)
        n = S.size()[0]
        if V_net_comp:
            x_ls = torch.linspace(0,2*np.pi,n+1)[:-1].view(-1,1)
            e = [comp(V_net,V_net_comp,x_ls, n_comp)]
        for k in trange(epochs, leave=False, desc="Epoch"):
            B, Rem = gen_batches(N,M, Rem)
            for i, B_θ in enumerate(tqdm(B,leave=False,desc="Batch")):
                A_θ = torch.randperm(n-1)[:m]
                s = S[A_θ]
                s_1 = S[A_θ+1]
                s_2 = S[A_θ+2]
                s̃_1 = s + s_2 - s_1
                V_θ = [V_j for j,V_j in enumerate(V_s) if j in B_θ]
                f = torch.cat([R[A_θ] + γ*V_j(s_1)-V_j(s) for V_j in V_θ],1)
                f̃ = torch.cat([R[A_θ] + γ*V_j(s̃_1)-V_j(s) for V_j in V_θ],1)
                L = torch.sum(f*f̃,0)/m
                mu = torch.exp(-β*L)
                for xX in zip(V_net.parameters(),*[V_j.parameters() for V_j in V_θ]):
                    xX_tensor = torch.cat([xX_j.view(1,-1) for xX_j in xX])
                    x_new = xX_tensor[0]
                    X_new = xX_tensor[1:]
                    x_new = mu.view(1,-1).mm(X_new)/torch.sum(mu)
                    # Particle-wise noise              
                    #  X_new += -λ*η*(X_new-x_new)+σ*np.sqrt(η)*torch.diag(torch.normal(torch.zeros(M))).mm(X_new-x_new)
                    # Parameter-particle-wise noise
                    X_new += -λ*η*(X_new-x_new)+σ*np.sqrt(η)*torch.normal(torch.zeros(X_new.size()))*(X_new-x_new)
                    xX_new = torch.cat((x_new,X_new)).view(-1, *xX[0].size())
                    for xX_j, xX_new_j in zip(xX,xX_new):
                        xX_j +=  xX_new_j - xX_j
                if V_net_comp:
                    e.append(comp(V_net,V_net_comp,x_ls, n_comp))
        if V_net_comp:
            return V_net, torch.tensor(e)
        else:
            return V_net
```


```python
N = int(1e4)
M = int(2e3)
m = int(S.size()[0]/M)
epochs = 30

V_CBO = Net()
V_s = [Net() for _ in range(N)]
V_CBO, e_CBO = algorithm_CBO(V_CBO, V_s, S, M, m, epochs, 0.9,5.,3.,0.1, 200., V_net_comp=V_star, n_comp=100)

V_CBO_BFF = Net()
V_s = [Net() for _ in range(N)]
V_CBO_BFF, e_CBO_BFF = algorithm_CBO_BFF(V_CBO_BFF, V_s, S, M, m, epochs, 0.9,5.,3.,0.1, 200., V_net_comp=V_star, n_comp=100)
```


    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=30.0, style=ProgressStyle(description_width='…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=30.0, style=ProgressStyle(description_width='…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=5.0, style=ProgressStyle(description_width='i…



```python
fig, axs = plt.subplots(figsize=(12,5),ncols=2)
x_ls = torch.linspace(0,2*np.pi,300)
y_star_ls = V_star(x_ls.view(-1,1)).view(-1)
y_ls = torch.cat([V(x_ls.view(-1,1)).view(1,-1) for V in [V_CBO, V_CBO_BFF]])
y_ls -= torch.mean(y_ls-y_star_ls.view(1,-1),axis=1).view(-1,1)
e_ls = [e_CBO, e_CBO_BFF]
labels = ["CBO", "CBO BFF"]
line_styles = ['solid','solid']
axs[0].plot(x_ls.detach().numpy(),y_star_ls.detach().numpy(),label="Algo 1 *",color="black")
for j in range(0,len(y_ls)):
    axs[0].plot(x_ls.detach().numpy(),y_ls[j].detach().numpy(),label=labels[j],color=colors[j], ls=line_styles[j])
    axs[1].plot(e_ls[j]/e_ls[j][0], label=labels[j],color=colors[j])
axs[0].set_ylabel(r"$V(s)$")
axs[0].set_xlabel(r"$s$")
axs[1].set_yscale("log")
axs[1].set_ylabel(r"$e_k/e_0$ (log scale)")
axs[1].set_xlabel(r"$k$")
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right')
plt.tight_layout()
plt.subplots_adjust(right=0.9)
```


![png](CBO_RL_files/CBO_RL_31_0.png)


#### Searching Hyperparameter space


```python
V_s = [Net() for _ in range(int(1e2))]
L_m = 1.
n = S.size()[0]
m = int(1e3)
A_θ = torch.randperm(n-1)[:m]
s = S[A_θ]
s_1 = S[A_θ+1]
f = torch.cat([R[A_θ] + γ*V_j(s_1)-V_j(s) for V_j in V_s],1)
L = torch.mean(f**2 / 2.,0)
V_0 = 0
for X in zip(*[V_j.parameters() for V_j in V_s]):
    X_tensor = torch.cat([X_j.view(1,-1) for X_j in X])
    V_0 += ((X_tensor - X_tensor.mean(axis=0).view(1,-1))**2).mean(axis=0).norm()**2.
```


```python
λ_ls = np.linspace(0, 9, 50)
σ_0_ls = np.linspace(0, 4, 40)
def μ_f(β,λ,σ_0):
    M_0 = torch.exp(-β*L).mean()
    return 2*λ - σ_0**2 - 2*σ_0**2*np.exp(-β*L_m)/M_0.detach().numpy()
λ,σ_0 = np.meshgrid(λ_ls, σ_0_ls)
β = 200.
μ = μ_f(β,λ, σ_0)
plt.contourf(λ, σ_0,μ, cmap='RdBu', norm=colors.DivergingNorm(vcenter=0))
plt.title(r"$\mu$, for $\beta$="+f"{β}")
plt.xlabel(r"$λ$")
plt.ylabel(r"$\sigma_0$")
plt.colorbar()
```




    <matplotlib.colorbar.Colorbar at 0x7f43766b8150>




![png](CBO_RL_files/CBO_RL_34_1.png)



```python
M = int(1e4)
m = int(S.size()[0]/M)
epochs = 25
V_ls = []
e_ls = []
labels = []
for λ in tqdm(torch.linspace(5.,6., 1),leave=False, desc="λ"):
    for σ_0 in tqdm(torch.linspace(3.,4., 2),leave=False, desc="σ_0"):
        for β in tqdm(torch.linspace(150.,250., 4), leave=False, desc="β"):
            V_s = [Net() for _ in range(int(1e4))]
            V_CBO, e_CBO = algorithm_CBO(Net(), V_s, S, M, m, epochs, 0.9,λ,σ_0,0.1,β, V_net_comp=V_star, n_comp=100)
            V_ls.append(V_CBO)
            e_ls.append(e_CBO)
            labels.append(f"λ: {λ:0.2f}, σ: {σ_0:0.2f}, β: {β:0.2f},")
```


    HBox(children=(FloatProgress(value=0.0, description='λ', max=1.0, style=ProgressStyle(description_width='initi…



    HBox(children=(FloatProgress(value=0.0, description='σ_0', max=2.0, style=ProgressStyle(description_width='ini…



    HBox(children=(FloatProgress(value=0.0, description='β', max=4.0, style=ProgressStyle(description_width='initi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=25.0, style=ProgressStyle(description_width='…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=25.0, style=ProgressStyle(description_width='…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=25.0, style=ProgressStyle(description_width='…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=25.0, style=ProgressStyle(description_width='…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='β', max=4.0, style=ProgressStyle(description_width='initi…



    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=25.0, style=ProgressStyle(description_width='…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=25.0, style=ProgressStyle(description_width='…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=25.0, style=ProgressStyle(description_width='…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Epoch', max=25.0, style=ProgressStyle(description_width='…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



    HBox(children=(FloatProgress(value=0.0, description='Batch', max=1.0, style=ProgressStyle(description_width='i…



```python
fig, axs = plt.subplots(figsize=(12,5),ncols=2)
x_ls = torch.linspace(0,2*np.pi,100)
y_star_ls = V_star(x_ls.view(-1,1)).view(-1)
y_ls = torch.cat([V(x_ls.view(-1,1)).view(1,-1) for V in V_ls])
y_ls -= torch.mean(y_ls-y_star_ls.view(1,-1),axis=1).view(-1,1)
# colors = ["tab:blue", "tab:orange","tab:green","tab:red","tab:purple"]
# labels = ["Algo 1", "Algo 2", "Algo 3", "Algo 4", "CBO"]
# line_styles = ['solid','solid',(5,(5,5)),(0,(5,5)),'solid']
axs[0].plot(x_ls.detach().numpy(),y_star_ls.detach().numpy(),label="Algo 1 *",color="black")
for j in range(0,len(y_ls)):
    axs[0].plot(x_ls.detach().numpy(),y_ls[j].detach().numpy(),label=labels[j])
    axs[1].plot(e_ls[j]/e_ls[j][0], label=labels[j])
axs[0].set_ylabel(r"$V(s)$")
axs[0].set_xlabel(r"$s$")
axs[1].set_yscale("log")
axs[1].set_ylabel(r"$e_k/e_0$ (log scale)")
axs[1].set_xlabel(r"$k$")
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right')
plt.tight_layout()
plt.subplots_adjust(right=0.9)
```


![png](CBO_RL_files/CBO_RL_36_0.png)


## Discrete state space (5.2.)


```python

```
