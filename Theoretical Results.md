# Theoretical Results

## Relevant previous results

### CBO

**Lemma A.1 Q-evaluation**: If $\sup_{s\in\mathbb S,\theta\in\mathbb R^{d_\theta}}|\partial_s \mathbb E_a[\nabla_\theta Q^\pi(s,a;\theta)|s]|\leq C$ and $\sup_{s\in\mathbb S,a\in \mathbb A} |\mathbb E_a[\mu(s,a)|s]-\mu(s,a)|\leq C$ a.s. then the difference between gradients of US and BFF algorithms for Q-eval is bounded by 
$$
|\mathbb E[\hat F]-\mathbb E[F]|\leq \gamma C^2\mathbb E|\delta(ϵ+o(\epsilon))|
$$
Furthermore, if $|\mathbb E_a[Q^\pi(s,a;\theta)]-Q(s,a;\theta)|$, $|\mathbb E_a[\nabla_\theta Q^\pi(s,a;\theta)]-\nabla_\theta Q(s,a;\theta)|$, $|\mu(s,a)-\mu(s,a')|$, $|r(s,s,a)|\leq C$ for a.s. $\forall s\in\mathbb S, a\in\mathbb A, \theta\in\mathbb R^{d_\theta}$, then the difference in variances can also be bounded by
$$
|\mathbb V[\hat F]- \mathbb V[F]|\leq O(\epsilon).
$$
Definitions:
$$
\begin{aligned}
& F = j(s_m,a_m,s_{m+1};\theta)\nabla_\theta j(s_m, a_m, s_{m+1}';\theta)\\
& \hat F = j(s_m,a_m,s_{m+1};\theta)\nabla_\theta j(s_m, a_m, s_m+\Delta s_{m+1};\theta)\\
& j(s_m,a_m,s_{m+1};\theta_m)= r(s_{m+1},s_m,a_m)+\gamma \mathbb E_{a\sim \pi(\cdot|s_{m+1})}[Q^\pi(s_{m+1,a;\theta})]-Q^\pi(s_m,a_m;\theta)\\
&\delta(s_m, a_m;\theta)=\mathbb E[j(s_m, a_m, s_{m+1};\theta_m)|s_m,a_m]\\
&ds_t=\mu(s_t,a_t)dt+\sigma dW_t
\end{aligned}
$$

Proof:
The expecations of US and BFF gradients are:
$$
\begin{aligned}
\mathbb E [F]=\mathbb E[\mathbb E[j|s_m,a_m]\mathbb[\nabla_\theta j'|s_m,\alpha_m]]=\mathbb E[\delta(s,a;\theta)\nabla_\theta\delta(s,a;\theta)]\\
\mathbb E[\hat F]= \mathbb E[\mathbb E[j\nabla\theta \hat j|s_m,\alpha_m]] = \mathbb E[\delta(s,a;\theta)\mathbb E[\nabla_\theta \hat j|s_m,a_m]]
\end{aligned}
$$
where $j'=j(s_m, a_m,s_{m+1};\theta)$ and $\hat j=j(s_m,a_m,s_m+\Delta s_{m+1};\theta)$. We then see that the difference between the gradients is 
$$
\begin{aligned}
\mathbb E[\hat F]-\mathbb E[F]=\mathbb E[\delta(s_m,a_m)\mathbb E[\nabla_\theta \hat j - \nabla_\theta j'|s_m,a_m]]
\end{aligned}
$$
Henceforth, we drop explicit dependence of $Q^\pi$ on $\theta$, assume all $\nabla=\nabla_\theta$. Using Taylor expansion of $\nabla Q^\pi(s_{m+1},a)\pi(a|s_{m+1})$ around $\nabla Q^\pi (s_m,a)\pi(a,s_m)$ :
$$
\begin{aligned}
&\nabla Q^\pi(s_{m+1},a)\pi(a|s_{m+1}) \\=& \nabla Q^\pi(s_m,a)\pi(a,s_m) + \partial_s(\nabla Q^\pi(s_m,a)\pi(a,s_m))\Delta s_m +\tfrac{1}{2}\partial_2^2(\nabla Q^\pi(s_m,a)\pi(a,s_m) )\Delta s_m^2
\end{aligned}
$$
Substituting $\Delta s_m=\mu(s_m,a_m)\epsilon +\sigma Z_m \sqrt \epsilon$ yields
$$
\begin{aligned}
\nabla_\theta j'
=& \gamma \int da \nabla Q^\pi(s_{m+1},a)\pi(a|s_{m+1})-\nabla Q^\pi(s_m,a_m)\\
=& \underbrace{\gamma \int da \,\nabla Q^\pi(s_{m},a)\pi(a|s_{m})-\nabla Q^\pi(s_m,a_m)}_{f_0}\\
&+\underbrace{\gamma\int da\, \partial_s(\nabla Q^\pi(s_{m},a)\pi(a|s_{m}))\mu(s_m,a_m)}_{f_1}\epsilon\\
&+\underbrace{\gamma\int da\, \partial_s(\nabla Q^\pi(s_{m},a)\pi(a|s_{m}))\sigma }_{f_2}Z_m\sqrt\epsilon\\
&+\underbrace{\gamma\int da\, \partial_s(\nabla Q^\pi(s_{m},a)\pi(a|s_{m}))\tfrac{1}{2}\sigma^2 }_{f_3}Z_m^2\epsilon+o(\epsilon)\\
\end{aligned}
$$
Similarly, we can Taylor expand $\nabla Q^\pi(s_m+\Delta s_{m+1},a)\pi(a|s_m+\Delta s_{m+1})$ around $\nabla Q^\pi(s_m,a)\pi(a,s_m)$, replacing $\Delta s_m$ with $\Delta s_{m+1}$. We also Taylor expand $\mu(s_{m+1},a_{m+1})$ around $\mu(s_m,a_{m+1})$ and use $\Delta s_m=O(\sqrt \epsilon)$ to get $\mu(s_{m+1},a_{m+1})=\mu(s_m,a_{m+1})+o(1)$, thus $\Delta s_{m+1} = \mu(s_m,a_{m+1})\epsilon + \sigma Z_{m+1} \sqrt \epsilon + o(\epsilon)$. Combining the expressions we find
$$
\begin{aligned}
\nabla_\theta \hat j = f_0+\underbrace{\gamma\int da\, \partial_s(\nabla Q^\pi(s_{m},a)\pi(a|s_{m}))\mu(s_m,a_{m+1})}_{\hat f_1}\epsilon+f_2Z_{m+1}\sqrt{\epsilon}+f_3Z_{m+1}^2\epsilon+o(\epsilon)
\end{aligned}
$$
Thus 
$$
\begin{aligned}
\mathbb E[\nabla\hat j - \nabla j'|s_m,a_m] &= \mathbb E[(\hat f_1 - f_1)\epsilon|s_m,a_m]\epsilon+o(\epsilon)\\
&=\gamma\int da\, \partial_s(\nabla Q^\pi(s_{m},a)\pi(a|s_{m}))\mathbb E[\mu(s_m,a_{m+1})-\mu(s_m,a_m)|s_m,a_m]\epsilon+o(\epsilon)
\end{aligned}
$$
Applying the assumptions, we conclude $\mathbb E[\hat F - F]=\gamma C^2\mathbb E [\delta\cdot(\epsilon+o(\epsilon))]$
... variance follows similarly $\blacksquare$
**Lemma A.2 Q-control**: Lef $f(s;\theta)=\max_{a'\in\mathbb A}Q^*(s,a;\theta)$. Suppose $f(s;\theta)$ is continuous in $s\in\mathbb S$ and $\partial_s f(s;\theta),\partial_s^2f(s;\theta)$ exists a.s. Further, assume $\sup_{s\in\mathbb S,\theta\in\mathbb R^{d_\theta}}|\partial_s \nabla_\theta f(s_m;\theta)|s]|$, $\sup_{s\in\mathbb S, a\in \mathbb A}|\mathbb E_a[\mu(s,a)|s]-\mu(s,a)|\leq C$ a.s. Then the difference between gradients of US and BFF algorithms for Q-ctrl is bounded by 
$$
|\mathbb E[\hat F]-\mathbb E[F]|\leq \gamma C^2\mathbb E|\delta(\sqrt ϵ+o(\epsilon))|
$$
Furthermore, if $|\max_a Q(s,a;\theta) - Q(s,a;\theta)|$, $|\nabla_\theta \max_a Q^\pi(s,a;\theta)-\nabla_\theta Q(s,a;\theta)|$, $|\mu(s,a)-\mu(s,a')|$, $|r(s,s,a)|\leq C$ for a.s. $\forall s\in\mathbb S, a\in\mathbb A, \theta\in\mathbb R^{d_\theta}$, then the difference in variances can also be bounded by
$$
|\mathbb V[\hat F]- \mathbb V[F]|\leq O(\sqrt\epsilon).
$$
Definitions:
$$
\begin{aligned}
& j(s_m,a_m,s_{m+1};\theta_m)= r(s_{m+1},s_m,a_m)+\gamma \max_a Q^*(s_{m+1},a;\theta)-Q^\pi(s_m,a_m;\theta)\\
\end{aligned}
$$

**Extensions and proof of Theorem 3.2**

Optimization processes
$$
\begin{aligned}
&\partial_t p = \nabla\left[\mathbb E[F]p +\tfrac{\eta}{2}\nabla(\mathbb V[F] p)\right]\\
&\partial_t \hat p = \nabla\left[\mathbb E[\hat F]\hat p +\tfrac{\eta}{2}\nabla(\mathbb V[\hat F] \hat p)\right]
\end{aligned}
$$
...