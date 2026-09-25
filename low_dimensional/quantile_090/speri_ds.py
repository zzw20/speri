import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import truncnorm, gaussian_kde
from scipy.optimize import brentq
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count


# -------------------- Settings --------------------

B = 1000
n1 = n0 = 1000
n = n1 + n0
tau = 0.9
TRUE_VALUE = 0.5117

EPOCHS = 600
BATCH_SIZE = 128
LR = 1e-3
VAL_FRAC = 0.2
PATIENCE = 10
MIN_DELTA = 1e-6

RHO_HIDDEN = 32
B_HIDDEN = 32
N_THRESHOLDS = 30
device = "cpu"


# -------------------- Helper functions --------------------

def rtruncnorm(size, mean, sd, lower, upper, rng):
    a, b = (lower - mean) / sd, (upper - mean) / sd
    return truncnorm.rvs(a, b, loc=mean, scale=sd, size=size, random_state=rng)


class RhoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, RHO_HIDDEN), nn.ReLU(),
            nn.Linear(RHO_HIDDEN, RHO_HIDDEN), nn.ReLU(),
            nn.Linear(RHO_HIDDEN, 1)
        )

    def forward(self, x):
        return self.net(x)


class BNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, B_HIDDEN), nn.ReLU(),
            nn.Linear(B_HIDDEN, B_HIDDEN), nn.ReLU(),
            nn.Linear(B_HIDDEN, 1)
        )

    def forward(self, x):
        return self.net(x)


def fit_net(X, y, NetClass, loss_fn):
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    model = NetClass().to(device)

    n_val = max(1, int(VAL_FRAC * len(X)))
    idx = torch.randperm(len(X))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    loader = DataLoader(TensorDataset(X[train_idx], y[train_idx]),
                        batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_loss, best_state, bad_epochs = np.inf, None, 0

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X[val_idx]), y[val_idx]).item()

        if val_loss + MIN_DELTA < best_loss:
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model.cpu()


@torch.no_grad()
def predict(model, X):
    return model(torch.as_tensor(X, dtype=torch.float32)).numpy().squeeze()


def U(y, theta):
    return (y <= theta).astype(float) - tau


def make_cdf_data(X, y):
    lower, upper = np.quantile(y, [0.01, 0.99])
    thresholds = np.linspace(lower, upper, N_THRESHOLDS)

    X_rep = np.repeat(X, N_THRESHOLDS, axis=0)
    theta_rep = np.tile(thresholds, len(y))[:, None]
    y_rep = np.repeat(y, N_THRESHOLDS)

    Z = np.hstack([X_rep, theta_rep])
    target = (y_rep <= theta_rep.squeeze()).astype(np.float32)[:, None]

    return Z, target


# -------------------- One simulation --------------------

def run_sim(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Generate data
    x_p = rng.uniform(-1, 1, n1)
    x_q = rtruncnorm(n0, 0.5, 1, -1, 1, rng)

    y1 = 2 * x_p + rng.normal(0, 1, n1)
    y0 = x_q + rng.normal(0, 1, n0)

    x_p = x_p[:, None]
    x_q = x_q[:, None]

    # Two folds
    p1, p2 = np.arange(n1 // 2), np.arange(n1 // 2, n1)
    q1, q2 = np.arange(n0 // 2), np.arange(n0 // 2, n0)

    # Estimate rho
    X_rho1 = np.vstack([x_p[p1], x_q[q1]])
    X_rho2 = np.vstack([x_p[p2], x_q[q2]])

    R_rho1 = np.vstack([np.ones((len(p1), 1)), np.zeros((len(q1), 1))])
    R_rho2 = np.vstack([np.ones((len(p2), 1)), np.zeros((len(q2), 1))])

    rho_loss = nn.BCEWithLogitsLoss()
    rho1_model = fit_net(X_rho1, R_rho1, RhoNet, rho_loss)
    rho2_model = fit_net(X_rho2, R_rho2, RhoNet, rho_loss)

    def rho(model, X):
        logits = predict(model, X)
        prob = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        return prob / np.clip(1 - prob, 1e-12, None)

    def rho1(X):
        return rho(rho1_model, X)

    def rho2(X):
        return rho(rho2_model, X)

    # Estimate b(theta, x) = F(theta | x) - tau
    Z1, target1 = make_cdf_data(x_q[q1], y0[q1])
    Z2, target2 = make_cdf_data(x_q[q2], y0[q2])

    b_loss = nn.BCEWithLogitsLoss()
    b1_model = fit_net(Z1, target1, BNet, b_loss)
    b2_model = fit_net(Z2, target2, BNet, b_loss)

    def b(model, theta, X):
        Z = np.hstack([X, np.full((len(X), 1), theta)])
        logits = predict(model, Z)
        prob = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        return prob - tau

    def b1(theta, X):
        return b(b1_model, theta, X)

    def b2(theta, X):
        return b(b2_model, theta, X)

    # Estimate theta
    theta_p1 = np.quantile(y1, tau)

    def score1(theta):
        return (np.mean(rho1(x_q[q2]) * (U(y0[q2], theta) - b1(theta, x_q[q2])))
                + np.mean(b1(theta, x_p[p2])))

    def score2(theta):
        return (np.mean(rho2(x_q[q1]) * (U(y0[q1], theta) - b2(theta, x_q[q1])))
                + np.mean(b2(theta, x_p[p1])))

    try:
        theta_p0 = 0.5 * (brentq(score1, -5, 5) + brentq(score2, -5, 5))
    except ValueError:
        return np.nan, np.nan, np.nan

    estimate = theta_p1 - theta_p0

    # Estimate B0 and B1
    B1 = 1 / gaussian_kde(y1, bw_method="scott").evaluate(theta_p1)[0]

    h = 0.2
    kernel_y0 = np.exp(-0.5 * ((y0 - theta_p0) / h) ** 2) / (h * np.sqrt(2 * np.pi))
    rho_q = 0.5 * (rho1(x_q) + rho2(x_q))
    B0 = 1 / np.mean(rho_q * kernel_y0)

    # Influence function
    pi_p = n1 / n

    IF_p1 = (B0 * b1(theta_p0, x_p[p2]) - B1 * U(y1[p2], theta_p1)) / pi_p
    IF_p2 = (B0 * b2(theta_p0, x_p[p1]) - B1 * U(y1[p1], theta_p1)) / pi_p

    IF_q1 = B0 * (U(y0[q1], theta_p0) - b2(theta_p0, x_q[q1])) * rho2(x_q[q1]) / (1 - pi_p)
    IF_q2 = B0 * (U(y0[q2], theta_p0) - b1(theta_p0, x_q[q2])) * rho1(x_q[q2]) / (1 - pi_p)

    IF = np.concatenate([IF_p1, IF_p2, IF_q1, IF_q2])
    variance = np.mean(IF ** 2) / n
    se = np.sqrt(variance)

    return estimate, variance, se


# -------------------- Monte Carlo --------------------

if __name__ == "__main__":

    with ProcessPoolExecutor(max_workers=max(1, cpu_count() - 1)) as pool:
        results = list(pool.map(run_sim, range(1, B + 1)))

    results = pd.DataFrame(results, columns=["estimate", "variance", "se"]).dropna()
    results.to_csv("speri_ds_q90.csv", index=False)

    estimates = results["estimate"]
    se = results["se"]

    coverage = np.mean(
        (estimates - 1.96 * se <= TRUE_VALUE) &
        (TRUE_VALUE <= estimates + 1.96 * se)
    )

    print(f"\nSuccessful reps: {len(results)}/{B}")
    print(f"Mean: {estimates.mean():.4f}")
    print(f"ESE:  {estimates.std(ddof=1):.4f}")
    print(f"ASE:  {se.mean():.4f}")
    print(f"Cov:  {coverage:.4f}")