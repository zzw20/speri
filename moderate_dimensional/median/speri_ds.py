# ============================================================
# Moderate-dimensional simulation
# Target: Median general response difference
# Estimator: SPERI-DS
# ============================================================

import numpy as np
import pandas as pd
import torch
from scipy.stats import truncnorm, gaussian_kde
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import brentq

# Simulation settings

n_rep = 100

n_p = 5000
n_q = 5000
n = n_p+n_q
d = 4
pi_p = n_p/n

beta_1 = np.full(d, 2.0)
beta_0 = np.full(d, 1.0)

# Truncated normal function

def rtruncnorm(size, a, b, loc=0.0, scale=1.0):
    a_scaled = (a-loc)/scale
    b_scaled = (b-loc)/scale
    return truncnorm.rvs(a_scaled, b_scaled, loc=loc, scale=scale, size=size)

# Neural networks

def build_classifier(d, h=64):
    return nn.Sequential(
        nn.Linear(d, h), nn.ReLU(),
        nn.Linear(h, h), nn.ReLU(),
        nn.Linear(h, 1)
    )

def build_quantile_net(d, h=64):
    return nn.Sequential(
        nn.Linear(d, h), nn.ReLU(),
        nn.Linear(h, h), nn.ReLU(),
        nn.Linear(h, 1)
    )

# Quantile loss

class CheckLoss(nn.Module):
    def __init__(self, tau=0.5):
        super().__init__()
        self.tau = tau

    def forward(self, yhat, y):
        error = y-yhat
        return torch.mean(torch.maximum(self.tau*error, (self.tau-1)*error))

# Neural network training

def fit_net(net, X, y, loss_fn, n_epochs=500, lr=1e-5, batch_size=128,
            val_frac=0.2, patience=20, min_delta=1e-6, min_epochs=50):

    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)

    n_val = int(val_frac*len(X))
    perm = torch.randperm(len(X))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_loader = DataLoader(
        TensorDataset(X[train_idx], y[train_idx]),
        shuffle=True, batch_size=batch_size
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(n_epochs):

        net.train()

        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimizer.step()

        net.eval()

        with torch.no_grad():
            val_loss = loss_fn(net(X[val_idx]), y[val_idx]).item()

        if val_loss+min_delta < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience and epoch+1 >= min_epochs:
                break

    return net

# Median estimating function

def U(y, theta):
    return (y <= theta).astype(float)-0.5

# Storage

results = np.empty((n_rep, 3))

# Monte Carlo simulation

for rep in range(n_rep):

    seed = rep+1

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Generate X

    x_p = rng.uniform(-1, 1, size=(n_p, d))
    x_q = rtruncnorm(n_q*d, -1, 1, 0.5, 1).reshape(n_q, d)

    # Generate outcomes Y

    y1_p = x_p @ beta_1 + rng.normal(0, 1, n_p)
    y1_q = x_q @ beta_1 + rng.normal(0, 1, n_q)
    y0_p = x_p @ beta_0 + rng.normal(0, 1, n_p)
    y0_q = x_q @ beta_0 + rng.normal(0, 1, n_q)

    # Data splitting

    p_A = np.arange(0, n_p//2)
    p_B = np.arange(n_p//2, n_p)

    q_A = np.arange(0, n_q//2)
    q_B = np.arange(n_q//2, n_q)

    # Estimate density ratio

    mse = nn.MSELoss()

    rho1_net = fit_net(
        build_classifier(d),
        np.vstack([x_p[p_B], x_q[q_A]]),
        np.hstack([np.ones(n_p//2), np.zeros(n_q//2)])[:, None],
        mse
    )

    rho2_net = fit_net(
        build_classifier(d),
        np.vstack([x_p[p_A], x_q[q_B]]),
        np.hstack([np.ones(n_p//2), np.zeros(n_q//2)])[:, None],
        mse
    )

    @torch.no_grad()
    def rho1(x):
        p = rho1_net(torch.as_tensor(x, dtype=torch.float32)).numpy().squeeze()
        return p/(1-p+1e-8)

    @torch.no_grad()
    def rho2(x):
        p = rho2_net(torch.as_tensor(x, dtype=torch.float32)).numpy().squeeze()
        return p/(1-p+1e-8)

    # Estimate conditional median

    qloss = CheckLoss(0.5)

    b1_net = fit_net(
        build_quantile_net(d),
        x_q[q_A],
        y0_q[q_A][:, None],
        qloss
    )

    b2_net = fit_net(
        build_quantile_net(d),
        x_q[q_B],
        y0_q[q_B][:, None],
        qloss
    )

    @torch.no_grad()
    def b1(theta, x):
        median = b1_net(torch.as_tensor(x, dtype=torch.float32)).numpy().squeeze()
        return (theta < median).astype(float)-0.5

    @torch.no_grad()
    def b2(theta, x):
        median = b2_net(torch.as_tensor(x, dtype=torch.float32)).numpy().squeeze()
        return (theta < median).astype(float)-0.5

    # SPERI-DS estimator

    theta_p1_hat = np.median(y1_p)

    def eff1(theta):
        term_q = np.mean(
            rho1(x_q[q_B])*
            ((y0_q[q_B] <= theta)-0.5-b1(theta, x_q[q_B]))
        )
        term_p = np.mean(b1(theta, x_p[p_A]))
        return term_q+term_p

    def eff2(theta):
        term_q = np.mean(
            rho2(x_q[q_A])*
            ((y0_q[q_A] <= theta)-0.5-b2(theta, x_q[q_A]))
        )
        term_p = np.mean(b2(theta, x_p[p_B]))
        return term_q+term_p

    theta_p0_hat = 0.5*(brentq(eff1, -5, 5)+brentq(eff2, -5, 5))

    theta_hat = theta_p1_hat-theta_p0_hat

    # Variance estimation

    kde_y1 = gaussian_kde(y1_p, bw_method="scott")
    B1 = 1/kde_y1.evaluate(theta_p1_hat)[0]

    h2 = 0.1
    kernel_y0 = np.exp(-0.5*((y0_q-theta_p0_hat)/h2)**2)/(np.sqrt(2*np.pi)*h2)

    rho_q = (rho1(x_q)+rho2(x_q))/2
    B0 = 1/np.mean(rho_q*kernel_y0)

    r = np.concatenate([np.ones(n_p), np.zeros(n_q)])
    x_all = np.vstack([x_p, x_q])
    y1_all = np.concatenate([y1_p, y1_q])
    y0_all = np.concatenate([y0_p, y0_q])

    rho_hat = (rho1(x_all)+rho2(x_all))/2
    b_hat = b2(theta_p0_hat, x_all)

    IF = (
        r/pi_p*(B0*b_hat-B1*U(y1_all, theta_p1_hat))
        +(1-r)/(1-pi_p)*B0*
        (U(y0_all, theta_p0_hat)-b_hat)*rho_hat
    )

    variance_hat = np.sum(IF**2)/n**2
    se_hat = np.sqrt(variance_hat)

    # Store results

    results[rep, :] = [theta_hat, variance_hat, se_hat]

# Save results

df = pd.DataFrame(results, columns=["estimate", "variance", "se"])
df.to_csv("speri_ds.csv", index=False)