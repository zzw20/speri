import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import beta as beta_dist


# Simulation settings

n_sim = 1000

# Larger sample sizes are used for a more accurate asymptotic approximation
# for the SPERI-DS estimator.
n1 = 3000
n0 = 4500
d = 4

seed_global = 2025

p2_p = 0.31
p2_q = 0.36

params_p = np.array([
    [1.6995, 0.8077, 4.9653],
    [1.6976, 1.0739, 7.2052]
])

params_q = np.array([
    [2.2231, 2.5977, 0.9652],
    [4.1756, 2.5968, 0.6989]
])

beta_1 = np.array([63, -2, -26, -5, 18])
beta_0 = np.array([56, -3, -21, -12, 23])

sig1 = 9.32
sig0 = 10.13

theta_p0_true = 5.64


# Neural network settings

H = 512
H2 = 512

EPOCHS = 300
BATCH_SIZE = 128
LR = 1e-3
VAL_FRAC = 0.1
PATIENCE = 12
MIN_DELTA = 1e-6
MIN_EPOCHS = 40
WEIGHT_DECAY = 0.0

ODDS_MIN = 1e-6
ODDS_MAX = 1e6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data generation

def make_beta_block(n, params, rng):
    a = params[0, :]
    b = params[1, :]
    cols = [
        beta_dist.rvs(a[j], b[j], size=n, random_state=rng)
        for j in range(3)
    ]
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

    mu0 = x_q_aug @ beta_0/sig0
    mu1 = x_p_aug @ beta_1/sig1

    y0_q = mu0 + rng.normal(0, 1, size=n0)
    y1_p = mu1 + rng.normal(0, 1, size=n1)

    return (
        x_q,
        x_p,
        y0_q.astype(np.float32),
        y1_p.astype(np.float32)
    )


# Neural network

class Net(nn.Module):

    def __init__(self, in_dim=4, h=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h)
        self.fc2 = nn.Linear(h, h)
        self.out = nn.Linear(h, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


def fit_net(model, X, y, is_classifier=False):

    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)

    n = X.shape[0]
    n_val = max(1, int(VAL_FRAC*n))

    idx = torch.randperm(n, device=device)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    if is_classifier:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    model = model.to(device)

    best_loss = float("inf")
    best_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }
    bad_epochs = 0

    for epoch in range(1, EPOCHS+1):

        model.train()

        perm = torch.randperm(X_train.shape[0], device=device)

        for start in range(0, X_train.shape[0], BATCH_SIZE):

            idx_batch = perm[start:start+BATCH_SIZE]

            xb = X_train[idx_batch]
            yb = y_train[idx_batch]

            optimizer.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()

        if val_loss+MIN_DELTA < best_loss:

            best_loss = val_loss
            bad_epochs = 0

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        else:

            bad_epochs += 1

            if bad_epochs >= PATIENCE and epoch >= MIN_EPOCHS:
                break

    model.load_state_dict(best_state)

    return model.to(device).eval()


@torch.no_grad()
def predict(model, X):

    X = torch.as_tensor(
        X,
        dtype=torch.float32,
        device=device
    )

    return model(X).squeeze(1).cpu().numpy()


# One Monte Carlo replication

def run_sim(i):

    rng = np.random.default_rng(seed_global+i)

    torch.manual_seed(seed_global+i)
    np.random.seed(seed_global+i)

    x_q, x_p, y0_q, y1_p = generate_data(rng)

    n1h = n1//2
    n0h = n0//2

    # Density-ratio models

    X_rho1 = np.vstack([
        x_p[n1h:],
        x_q[:n0h]
    ])

    y_rho1 = np.concatenate([
        np.ones(n1-n1h),
        np.zeros(n0h)
    ]).astype(np.float32).reshape(-1, 1)

    rho1_model = fit_net(
        Net(d, H),
        X_rho1,
        y_rho1,
        is_classifier=True
    )

    logits_q2 = predict(rho1_model, x_q[n0h:])
    prob_q2 = 1/(1+np.exp(-np.clip(logits_q2, -30, 30)))

    rho1_q2 = np.clip(
        prob_q2/(1-prob_q2),
        ODDS_MIN,
        ODDS_MAX
    )

    X_rho2 = np.vstack([
        x_p[:n1h],
        x_q[n0h:]
    ])

    y_rho2 = np.concatenate([
        np.ones(n1h),
        np.zeros(n0-n0h)
    ]).astype(np.float32).reshape(-1, 1)

    rho2_model = fit_net(
        Net(d, H),
        X_rho2,
        y_rho2,
        is_classifier=True
    )

    logits_q1 = predict(rho2_model, x_q[:n0h])
    prob_q1 = 1/(1+np.exp(-np.clip(logits_q1, -30, 30)))

    rho2_q1 = np.clip(
        prob_q1/(1-prob_q1),
        ODDS_MIN,
        ODDS_MAX
    )

    # Outcome models

    b1_model = fit_net(
        Net(d, H2),
        x_q[:n0h],
        y0_q[:n0h].reshape(-1, 1)
    )

    b1_p1 = predict(b1_model, x_p[:n1h])
    b1_q2 = predict(b1_model, x_q[n0h:])

    b2_model = fit_net(
        Net(d, H2),
        x_q[n0h:],
        y0_q[n0h:].reshape(-1, 1)
    )

    b2_p2 = predict(b2_model, x_p[n1h:])
    b2_q1 = predict(b2_model, x_q[:n0h])

    # SPERI-DS estimator

    theta_1 = (
        np.mean(y1_p[:n1h]-b1_p1)
        + np.mean((b1_q2-y0_q[n0h:])*rho1_q2)
    )

    theta_2 = (
        np.mean(y1_p[n1h:]-b2_p2)
        + np.mean((b2_q1-y0_q[:n0h])*rho2_q1)
    )

    theta_hat = 0.5*(theta_1+theta_2)

    # Variance estimation

    theta_p1 = np.mean(y1_p)
    theta_p0 = theta_p1-theta_hat
    pi_p = n1/(n1+n0)

    b1_if_p1 = b1_p1-theta_p0_true
    b2_if_p2 = b2_p2-theta_p0_true
    b2_if_q1 = b2_q1-theta_p0_true
    b1_if_q2 = b1_q2-theta_p0_true

    score_p1 = (
        1/pi_p
        * (-b1_if_p1+(y1_p[:n1h]-theta_p1))
    )

    score_p2 = (
        1/pi_p
        * (-b2_if_p2+(y1_p[n1h:]-theta_p1))
    )

    score_q1 = (
        -1/(1-pi_p)
        * ((y0_q[:n0h]-theta_p0)-b2_if_q1)
        * rho2_q1
    )

    score_q2 = (
        -1/(1-pi_p)
        * ((y0_q[n0h:]-theta_p0)-b1_if_q2)
        * rho1_q2
    )

    IF = np.concatenate([
        score_p1,
        score_p2,
        score_q1,
        score_q2
    ])

    variance_hat = np.mean(IF**2)/(n1+n0)
    se_hat = np.sqrt(variance_hat)

    return theta_hat, variance_hat, se_hat


# Monte Carlo simulation

results = np.empty((n_sim, 3))

for i in range(1, n_sim+1):

    estimate, variance, se = run_sim(i)

    results[i-1, :] = [
        estimate,
        variance,
        se
    ]

    print(f"Simulation {i}/{n_sim}")


# Save results

results = pd.DataFrame(
    results,
    columns=["estimate", "variance", "se"]
)

results.to_csv("speri_ds.csv", index=False)