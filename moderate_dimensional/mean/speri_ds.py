# ============================================================
# Moderate-dimensional simulation
# Target: Mean general response difference
# Estimator: SPERI-DS
# ============================================================

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import truncnorm
from scipy.optimize import root_scalar

# Simulation settings

n_rep = 1000

n_p = 5000
n_q = 5000
n = n_p+n_q
d = 4
pi_p = n_p/n

beta_1 = np.full(d, 2.0)
beta_0 = np.full(d, 1.0)

# Truncated normal function

def rtruncnorm(size, a=-1, b=1, loc=0.5, scale=1.0):
    a_scaled = (a-loc)/scale
    b_scaled = (b-loc)/scale
    return truncnorm.rvs(a_scaled, b_scaled, loc=loc, scale=scale, size=size)

# Neural networks

def build_classifier(d, h=256):
    return nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, 1), nn.Sigmoid())

def build_outcome_net(d, h=256):
    return nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, 1))

# Early stopping

class EarlyStopper:
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best = np.inf
        self.count = 0

    def step(self, loss):
        if loss+self.min_delta < self.best:
            self.best = loss
            self.count = 0
        else:
            self.count += 1
        return self.count >= self.patience

# Neural network training

def fit(net, X, y, loss_fn, lr=1e-3, epochs=500, val_frac=0.2, bs=128):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    idx = torch.randperm(len(X))
    n_val = int(val_frac*len(X))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_loader = DataLoader(
        TensorDataset(X[train_idx], y[train_idx]), batch_size=bs, shuffle=True
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    stopper = EarlyStopper()

    for _ in range(epochs):
        net.train()

        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(X[val_idx]), y[val_idx]).item()

        if stopper.step(val_loss):
            break

    return net

# Storage

results = np.empty((n_rep, 3))

# Monte Carlo simulation

for rep in range(n_rep):

    seed = rep+1
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate X

    x_p = np.random.uniform(-1, 1, size=(n_p, d))
    x_q = rtruncnorm(n_q*d).reshape(n_q, d)

    # Generate outcomes Y

    y1_p = x_p @ beta_1 + np.random.normal(0, 1, n_p)
    y0_q = x_q @ beta_0 + np.random.normal(0, 1, n_q)

    # Data splitting

    p_A = np.arange(0, n_p//2)
    p_B = np.arange(n_p//2, n_p)

    q_A = np.arange(0, n_q//2)
    q_B = np.arange(n_q//2, n_q)

    # Estimate density ratio

    bce = nn.BCELoss()

    rho1_net = fit(
        build_classifier(d),
        np.vstack([x_p[p_B], x_q[q_A]]),
        np.vstack([np.ones((n_p//2, 1)), np.zeros((n_q//2, 1))]),
        bce
    )

    rho2_net = fit(
        build_classifier(d),
        np.vstack([x_p[p_A], x_q[q_B]]),
        np.vstack([np.ones((n_p//2, 1)), np.zeros((n_q//2, 1))]),
        bce
    )

    @torch.no_grad()
    def rho1(x):
        p = rho1_net(torch.tensor(x, dtype=torch.float32)).numpy().squeeze()
        return p/(1-p+1e-8)

    @torch.no_grad()
    def rho2(x):
        p = rho2_net(torch.tensor(x, dtype=torch.float32)).numpy().squeeze()
        return p/(1-p+1e-8)

    # Estimate outcome model

    mse = nn.MSELoss()

    b1_net = fit(
        build_outcome_net(d),
        x_q[q_A],
        y0_q[q_A].reshape(-1, 1),
        mse
    )

    b2_net = fit(
        build_outcome_net(d),
        x_q[q_B],
        y0_q[q_B].reshape(-1, 1),
        mse
    )

    @torch.no_grad()
    def mu1(x):
        return b1_net(torch.tensor(x, dtype=torch.float32)).numpy().squeeze()

    @torch.no_grad()
    def mu2(x):
        return b2_net(torch.tensor(x, dtype=torch.float32)).numpy().squeeze()

    # SPERI-DS estimator

    theta_p0_1 = np.mean(rho1(x_q[q_B])*(y0_q[q_B]-mu1(x_q[q_B]))) + \
        np.mean(mu1(x_p[p_A]))

    theta_p0_2 = np.mean(rho2(x_q[q_A])*(y0_q[q_A]-mu2(x_q[q_A]))) + \
        np.mean(mu2(x_p[p_B]))

    theta_p0_hat = (theta_p0_1+theta_p0_2)/2
    theta_p1_hat = np.mean(y1_p)

    theta_hat = theta_p1_hat-theta_p0_hat

    # Variance estimation

    b1_p = mu1(x_p[p_A])-theta_p0_hat
    b2_p = mu2(x_p[p_B])-theta_p0_hat

    IF_p1 = (1/pi_p)*(-b1_p+(y1_p[p_A]-theta_p1_hat))
    IF_p2 = (1/pi_p)*(-b2_p+(y1_p[p_B]-theta_p1_hat))

    b2_q = mu2(x_q[q_A])-theta_p0_hat
    b1_q = mu1(x_q[q_B])-theta_p0_hat

    IF_q1 = -(1/(1-pi_p))*((y0_q[q_A]-theta_p0_hat)-b2_q)*rho2(x_q[q_A])
    IF_q2 = -(1/(1-pi_p))*((y0_q[q_B]-theta_p0_hat)-b1_q)*rho1(x_q[q_B])

    variance_hat = (
        np.sum(IF_p1**2)+np.sum(IF_p2**2)+
        np.sum(IF_q1**2)+np.sum(IF_q2**2)
    )/n**2

    se_hat = np.sqrt(variance_hat)

    # Store results

    results[rep, :] = [theta_hat, variance_hat, se_hat]

# Save results

df = pd.DataFrame(results, columns=["estimate", "variance", "se"])
df.to_csv("speri_ds.csv", index=False)