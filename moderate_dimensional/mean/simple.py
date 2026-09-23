# ============================================================
# Moderate-dimensional simulation
# Target: Mean general response difference
# Estimator: Simple
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

# Simulation settings

n_rep = 1000

n_p = 5000
n_q = 5000
d = 4

beta_1 = np.array([2, 2, 2, 2])
beta_0 = np.array([1, 1, 1, 1])


# Truncated normal function

def rtruncnorm(size, a, b, mean, sd):
    a_scaled = (a - mean) / sd
    b_scaled = (b - mean) / sd
    return truncnorm.rvs(a_scaled, b_scaled, loc=mean, scale=sd, size=size)


# Storage

results = np.empty((n_rep, 3))

# Monte Carlo simulation

for i in range(n_rep):
    np.random.seed(i)

    # Generate X

    x_p = np.random.uniform(-1, 1, size=(n_p, d))
    x_q = rtruncnorm(n_q * d, -1, 1, 0.5, 1).reshape(n_q, d)

    # Generate outcomes Y

    y1_p = x_p @ beta_1 + np.random.normal(0, 1, n_p)
    y0_q = x_q @ beta_0 + np.random.normal(0, 1, n_q)

    # Simple estimator

    theta_p1_hat = np.mean(y1_p)
    theta_p0_hat = np.mean(y0_q)

    theta_hat = theta_p1_hat - theta_p0_hat

    # Variance estimation

    variance_hat = np.var(y1_p, ddof=1) / n_p + np.var(y0_q, ddof=1) / n_q
    se_hat = np.sqrt(variance_hat)

    # Store the results

    results[i, :] = [theta_hat, variance_hat, se_hat]

# Save results

df = pd.DataFrame(results, columns=["estimate", "variance", "se"])
df.to_csv("simple.csv", index=False)