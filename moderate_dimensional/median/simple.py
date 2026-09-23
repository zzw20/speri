# ============================================================
# Moderate-dimensional simulation
# Target: Median general response difference
# Estimator: Simple
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import truncnorm, gaussian_kde

# Simulation settings

n_rep = 1000

n_p = 5000
n_q = 5000
d = 4

beta_1 = np.full(d, 2.0)
beta_0 = np.full(d, 1.0)

# Truncated normal function

def rtruncnorm(size, a, b, loc=0.5, scale=1.0):
    a_scaled = (a-loc)/scale
    b_scaled = (b-loc)/scale
    return truncnorm.rvs(a_scaled, b_scaled, loc=loc, scale=scale, size=size)

# Storage

results = np.empty((n_rep, 3))

# Monte Carlo simulation

for i in range(n_rep):

    rng = np.random.default_rng(i)

    # Generate X

    x_p = rng.uniform(-1, 1, size=(n_p, d))
    x_q = rtruncnorm(n_q*d, -1, 1, loc=0.5, scale=1.0).reshape(n_q, d)

    # Generate outcomes Y

    y1_p = x_p @ beta_1 + rng.normal(0, 1, n_p)
    y0_q = x_q @ beta_0 + rng.normal(0, 1, n_q)

    # Simple estimator

    theta_p1_hat = np.median(y1_p)
    theta_p0_hat = np.median(y0_q)

    theta_hat = theta_p1_hat-theta_p0_hat

    # Variance estimation

    kde_y1 = gaussian_kde(y1_p)
    kde_y0 = gaussian_kde(y0_q)

    f_y1 = kde_y1.evaluate(theta_p1_hat)[0]
    f_y0 = kde_y0.evaluate(theta_p0_hat)[0]

    variance_hat = 1/(4*n_p*f_y1**2) + 1/(4*n_q*f_y0**2)
    se_hat = np.sqrt(variance_hat)

    # Store results

    results[i, :] = [theta_hat, variance_hat, se_hat]

# Save results

df = pd.DataFrame(results, columns=["estimate", "variance", "se"])
df.to_csv("simple.csv", index=False)