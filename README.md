# SplitMeanFlow_MNIST
Test SplitMeanFlow and MeanFlow on MNIST dataset.

Paper: [SplitMeanFlow: Interval Splitting Consistency in Few-Step Generative Modeling](https://arxiv.org/abs/2507.16884)

## RUN
```bash
python train_splitmeanflow.py
python test.py --ckpt splitmeanflow_0070000.pt # generate samples
```

## RESULT
SplitMeanFlow and MeanFlow can generate samples in one step, FlowMatch can not.

![SplitMeanFlow Result](/splitmeanflow_result.png)

## FORMULA
MeanFlow:
```math
\begin{split}
x_1 = \text{noise} \;\;\;\; x_0 = \text{real sample} \\
x_1 \to x_0 \\
x_t = (1-t)x_0 + t x_1 \\
\frac{\mathrm{d}{x_t}}{\mathrm{d}t} = V(x_t,t) \\
\frac{\mathrm{d}{x_t}}{\mathrm{d}t} = x_1 - x_0 \\
\\
U(x_t,r,t) = \frac{1}{t-r}\int_{r}^{t} V(x_{\tau},\tau)d\tau \;\;\;\;\text{define U as average speed between r and t} \\
U(x_t,r,t) = V(x_t,t) - (t-r)\frac{\mathrm{d}{}}{\mathrm{d}t}U
\\
\\
\\u\_tgt = (x_1-x_0) - \frac{\mathrm{d}{}}{\mathrm{d}t}U
\\loss = \Vert U(x_t,r,t) - sg(u\_tgt) \Vert^2
\end{split}
```

SplitMeanFlow:
```math
\begin{split}
\text{Mathematical axiom: the additivity of definite integrals} \\
\because \int_{r}^{t} V(x_{\tau},\tau)d\tau = \int_{r}^{s} V(x_{\tau},\tau)d\tau + \int_{s}^{t} V(x_{\tau},\tau)d\tau \\
\therefore (t-r)U(x_t,r,t) = (s-r)U(x_s,r,s) + (t-s)U(x_t,s,t) \\
\\
\lambda = \frac{t-s}{t-r} \in [0,1] \\
U(x_t,r,t) = (1-\lambda)U(x_s,r,s) + \lambda U(x_t,s,t) \\
\\
\\
x_t = (1-t)x_0 + t x_1 \\
u_2 = U(x_t,s,t) \\
x_s = x_t - (t-s)u_2 \\
u_1 = U(x_s,r,s) \\
u\_tgt = (1-\lambda)u_1 + \lambda u_2 \\
loss = \Vert U(x_t,r,t) - sg(u\_tgt) \Vert^2
\end{split}
```
