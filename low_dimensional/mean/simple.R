# ============================================================
# Low-dimensional simulation
# Target: Mean general response difference
# Estimator: Simple
# ============================================================

library(truncnorm)

# Simulation settings

n_rep <- 1000

n_p <- 1000
n_q <- 1000

beta_1 <- 2
beta_0 <- 1

# Storage

results <- matrix(
  NA_real_, nrow = n_rep, ncol = 3,
  dimnames = list(NULL,c("estimate", "variance", "se"))
)

# Monte Carlo simulation

for (i in 1:n_rep) {
  
  set.seed(i)
  
  # Generate X
  
  x_p <- runif(n_p, min=-1, max=1)
  x_q <- rtruncnorm(n_q,a=-1,b=1,mean=0.5,sd=1)
  
  # Generate outcomes Y
  
  y1_p <- beta_1*x_p + rnorm(n_p,mean=0,sd=1)
  y1_q <- beta_1*x_q + rnorm(n_q,mean=0,sd=1)
  y0_p <- beta_0*x_p + rnorm(n_p,mean=0,sd=1)
  y0_q <- beta_0*x_q + rnorm(n_q,mean=0,sd=1)
  
  # Simple estimator
  
  theta_hat <- mean(y1_p)-mean(y0_q)
  
  # Variance estimation
  
  variance_hat <- var(y1_p)/n_p + var(y0_q)/n_q
  
  se_hat <- sqrt(variance_hat)
  
  # Store the results
  
  results[i, ] <- c(estimate=theta_hat,variance=variance_hat,se=se_hat)
}

# Save results

write.csv(results,file="simple.csv",row.names=FALSE)