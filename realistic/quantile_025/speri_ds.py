import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import beta as beta_dist, gaussian_kde
from scipy.optimize import brentq, minimize_scalar


# Simulation settings

B = 1000

# Larger sample sizes are used for a more accurate asymptotic approximation
# for the SPERI-DS estimator.
n1 = 6000
n0 = 9000
d = 4
tau = 0.25

p2_p = 0.31
p2_q = 0.36

params_p = np.array([[1.6995, 0.8077, 4.9653],
                     [1.6976, 1.0739, 7.2052]])

params_q = np.array([[2.2231, 2.5977, 0.9652],
                     [4.1756, 2.5968, 0.6989]])

beta_1 = np.array([63, -2, -26, -5, 18])
beta_0 = np.array([56, -3, -21, -12, 23])

sig1 = 9.32
sig0 = 10.13


# Neural network settings

H_RHO = 8
H_Q = 16

EPOCHS = 600
BATCH_SIZE = 128
LR_RHO = 1e-3
LR_Q = 1e-3
VAL_FRAC = 0.1
PATIENCE = 15
MIN_DELTA = 1e-6
MIN_EPOCHS = 30

LOGIT_CLIP = 30
BANDWIDTH_UNDERSMOOTH = 1
EE_BRACKET_HALF = 0.27
EE_TOL = 1e-12

device = "cpu"


# Data generation

def make_beta_block(n, params, rng):
    a, b = params[0, :], params[1, :]
    cols = [beta_dist.rvs(a[j], b[j], size=n, random_state=rng) for j in range(3)]
    return np.column_stack(cols)


def generate_data(rng):
    sex_q = 1 + rng.binomial(1, p2_q, size=n0)
    sex_p = 1 + rng.binomial(1, p2_p, size=n1)

    cont_q = make_beta_block(n0, params_q, rng)
    cont_p = make_beta_block(n1, params_p, rng)

    x_q = np.column_stack([sex_q, cont_q]).astype(np.float32)
    x_p = np.column_stack([sex_p, cont_p]).astype(np.float32)

    x_q_aug = np.column_stack([np.ones(n0), x_q])
    x_p_aug = np.column_stack([np.ones(n1), x_p])

    y0_q = x_q_aug @ (beta_0/sig0) + rng.normal(0, 1, size=n0)
    y1_p = x_p_aug @ (beta_1/sig1) + rng.normal(0, 1, size=n1)

    return x_q, x_p, y0_q.astype(np.float32), y1_p.astype(np.float32)


# Neural networks

def build_classifier():
    return nn.Sequential(nn.Linear(d, H_RHO), nn.ReLU(),
                         nn.Linear(H_RHO, H_RHO), nn.ReLU(),
                         nn.Linear(H_RHO, H_RHO), nn.ReLU(),
                         nn.Linear(H_RHO, 1))


def build_quantile_net():
    return nn.Sequential(nn.Linear(d, H_Q), nn.ReLU(),
                         nn.Linear(H_Q, H_Q), nn.ReLU(),
                         nn.Linear(H_Q, H_Q), nn.ReLU(),
                         nn.Linear(H_Q, 1))


class CheckLoss(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, yhat, y):
        u = y-yhat
        return torch.mean(torch.maximum(self.tau*u, (self.tau-1)*u))


def fit_net(model, X, y, loss_fn, lr):
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    model = model.to(device)

    n_val = max(1, int(VAL_FRAC*len(X)))
    idx = torch.randperm(len(X))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    loader = DataLoader(TensorDataset(X[train_idx], y[train_idx]),
                        shuffle=True, batch_size=BATCH_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_loss = float("inf")
    bad_epochs = 0

    for epoch in range(EPOCHS):
        model.train()

        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X[val_idx].to(device)),
                               y[val_idx].to(device)).item()

        if val_loss + MIN_DELTA < best_loss:
            best_loss = val_loss
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE and epoch+1 >= MIN_EPOCHS:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model.cpu()


@torch.no_grad()
def predict(model, X):
    return model(torch.as_tensor(X, dtype=torch.float32)).numpy().squeeze()


def odds(model, X):
    logits = np.clip(predict(model, X), -LOGIT_CLIP, LOGIT_CLIP)
    prob = 1/(1+np.exp(-logits))
    return prob/np.clip(1-prob, 1e-12, None)


# Root solver

def find_root(fun, lower, upper):
    try:
        return brentq(fun, lower, upper, xtol=EE_TOL)
    except ValueError:
        pass

    grid = np.linspace(lower, upper, 200)
    values = np.array([fun(x) for x in grid])
    signs = np.sign(values)
    idx = np.where(signs[:-1]*signs[1:] <= 0)[0]

    if idx.size:
        a, b = grid[idx[0]], grid[idx[0]+1]
        try:
            return brentq(fun, a, b, xtol=EE_TOL)
        except ValueError:
            pass

    result = minimize_scalar(lambda x: fun(x)**2, bounds=(lower, upper),
                             method="bounded",
                             options={"xatol": EE_TOL, "maxiter": 500})

    if result.success:
        return float(result.x)

    return float(grid[np.argmin(np.abs(values))])


# Bandwidth for density estimation

def bandwidth(y):
    y = np.asarray(y, dtype=float)
    n = max(len(y), 1)

    if n <= 1:
        return 0.2

    sd = np.std(y, ddof=1)
    iqr = np.subtract(*np.percentile(y, [75, 25]))
    scale = min(sd, iqr/1.34) if np.isfinite(sd) and np.isfinite(iqr) else sd
    h = 0.9*scale*n**(-1/5)

    return max(1e-3, BANDWIDTH_UNDERSMOOTH*h)


def phi(u):
    return np.exp(-0.5*u*u)/np.sqrt(2*np.pi)


# One Monte Carlo replication

def run_sim(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    x_q, x_p, y0_q, y1_p = generate_data(rng)

    p1, p2 = np.arange(0, n1//2), np.arange(n1//2, n1)
    q1, q2 = np.arange(0, n0//2), np.arange(n0//2, n0)

    # Density-ratio models

    bce = nn.BCEWithLogitsLoss()

    X_rho1 = np.vstack([x_p[p2], x_q[q1]])
    y_rho1 = np.hstack([np.ones(len(p2)), np.zeros(len(q1))])[:, None]
    rho1_model = fit_net(build_classifier(), X_rho1, y_rho1, bce, LR_RHO)

    X_rho2 = np.vstack([x_p[p1], x_q[q2]])
    y_rho2 = np.hstack([np.ones(len(p1)), np.zeros(len(q2))])[:, None]
    rho2_model = fit_net(build_classifier(), X_rho2, y_rho2, bce, LR_RHO)

    # Conditional quantile models

    qloss = CheckLoss(tau)

    b1_model = fit_net(build_quantile_net(), x_q[q1],
                       y0_q[q1, None], qloss, LR_Q)
    b2_model = fit_net(build_quantile_net(), x_q[q2],
                       y0_q[q2, None], qloss, LR_Q)

    def b1(theta, X):
        qhat = predict(b1_model, X)
        return (theta < qhat).astype(float)-tau

    def b2(theta, X):
        qhat = predict(b2_model, X)
        return (theta < qhat).astype(float)-tau

    # SPERI-DS estimator

    theta_p1 = float(np.quantile(y1_p, tau))

    def score1(theta):
        return (np.mean(odds(rho1_model, x_q[q2]) *
                        ((y0_q[q2] <= theta).astype(float)-tau-b1(theta, x_q[q2])))
                + np.mean(b1(theta, x_p[p1])))

    def score2(theta):
        return (np.mean(odds(rho2_model, x_q[q1]) *
                        ((y0_q[q1] <= theta).astype(float)-tau-b2(theta, x_q[q1])))
                + np.mean(b2(theta, x_p[p2])))

    lower = theta_p1-1.09-EE_BRACKET_HALF
    upper = theta_p1-1.09+EE_BRACKET_HALF

    theta1 = find_root(score1, lower, upper)
    theta2 = find_root(score2, lower, upper)

    theta_p0 = 0.5*(theta1+theta2)
    theta_hat = theta_p1-theta_p0

    # Density at theta_p1

    h1 = bandwidth(y1_p)
    sd1 = np.std(y1_p, ddof=1)

    kde1 = gaussian_kde(y1_p, bw_method=h1/sd1 if sd1>0 else "scott")
    f1_hat = float(kde1.evaluate(theta_p1)[0])
    b1_full = 1/max(f1_hat, 1e-8)

    # Density at theta_p0 under P

    h0 = bandwidth(y0_q)
    kern_y = phi((y0_q-theta_p0)/h0)/h0

    rho_q = 0.5*(odds(rho1_model, x_q)+odds(rho2_model, x_q))
    f0_hat = np.mean(rho_q*kern_y)
    b0_full = 1/max(f0_hat, 1e-8)

    # Variance estimation

    pi_p = n1/(n1+n0)

    score_p1 = (1/pi_p)*(b0_full*b1(theta_p0, x_p[p1])
                         - b1_full*((y1_p[p1] <= theta_p1).astype(float)-tau))

    score_p2 = (1/pi_p)*(b0_full*b2(theta_p0, x_p[p2])
                         - b1_full*((y1_p[p2] <= theta_p1).astype(float)-tau))

    score_q1 = (1/(1-pi_p))*b0_full*(
        (y0_q[q1] <= theta_p0).astype(float)-tau-b2(theta_p0, x_q[q1])
    )*odds(rho2_model, x_q[q1])

    score_q2 = (1/(1-pi_p))*b0_full*(
        (y0_q[q2] <= theta_p0).astype(float)-tau-b1(theta_p0, x_q[q2])
    )*odds(rho1_model, x_q[q2])

    IF = np.concatenate([score_p1, score_p2, score_q1, score_q2])

    variance_hat = np.var(IF-IF.mean(), ddof=1)/(n1+n0)
    se_hat = np.sqrt(variance_hat)

    return theta_hat, variance_hat, se_hat


# Monte Carlo simulation

results = np.empty((B, 3))

for seed in range(1, B+1):
    estimate, variance, se = run_sim(seed)
    results[seed-1, :] = [estimate, variance, se]
    print(f"Simulation {seed}/{B}")


# Save results

results = pd.DataFrame(results, columns=["estimate","variance","se"])
results.to_csv("speri_ds.csv", index=False)