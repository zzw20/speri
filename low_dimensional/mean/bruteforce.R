# ============================================================
# Low-dimensional simulation
# Target: Mean general response difference
# Estimator: Bruteforce
# ============================================================

library(truncnorm)

# Simulation settings

n_rep <- 1000

n_p <- 1000
n_q <- 1000
n <- n_p + n_q
pi_p <- n_p / n

beta_1 <- 2
beta_0 <- 1

# Bandwidth for kernel estimation
h <- 0.2

# Gaussian kernel function

gaussian_kernel <- function(x, x_data, h) {
  exp(-0.5 * (x - x_data)^2 / h^2) /
    (sqrt(2 * pi) * h)
}

# Storage

results <- matrix(
  NA_real_, nrow = n_rep, ncol = 3,
  dimnames = list(NULL,c("estimate", "variance", "se"))
)

# Monte Carlo simulation

for (i in 1:n_rep) {
  
  set.seed(i)
  
  # Generate X
  
  x_p <- runif(n_p, min = -1, max = 1)
  x_q <- rtruncnorm(n_q,a=-1,b=1,mean=0.5,sd = 1)
  
  # Generate outcomes Y
  y1_p <- beta_1 * x_p + rnorm(n_p, mean = 0, sd = 1)
  y1_q <- beta_1 * x_q + rnorm(n_q, mean = 0, sd = 1)
  y0_p <- beta_0 * x_p + rnorm(n_p, mean = 0, sd = 1)
  y0_q <- beta_0 * x_q + rnorm(n_q, mean = 0, sd = 1)
  
  # Estimate density ratio rho(x)
  
  rho_hat <- function(x) {
    f_p_hat <- mean(gaussian_kernel(x, x_p, h))
    f_q_hat <- mean(gaussian_kernel(x, x_q, h))
    result <- f_p_hat / f_q_hat
    ifelse(is.nan(result), 0, result)
  }
  
  # Bruteforce estimator
  
  theta_p1_hat <- mean(y1_p)
  rho_q <- vapply(x_q,rho_hat,numeric(1))
  
  theta_p0_hat <- mean(y0_q * rho_q)
  
  theta_hat <- theta_p1_hat - theta_p0_hat
  
  # Variance estimation
  
  b_hat <- function(x) {
    weights <- gaussian_kernel(x, x_q, h)
    result <- mean(weights * (y0_q - theta_p0_hat)) / mean(weights)
    
    ifelse(is.nan(result), 0, result)
  }
  
  # For the mean estimating function U(Y, theta) = Y - theta
  B1 <- -1
  B0 <- -1
  
  influence_function <- function(r, y, x) {
    r/pi_p*(B0*b_hat(x)-B1*(y-theta_p1_hat)) +
      (1-r)/(1-pi_p)*B0*((y-theta_p0_hat)-b_hat(x))*rho_hat(x)
  }
  
  IF_p <- mapply(influence_function,r = rep(1, n_p),y = y1_p,x = x_p)
  
  IF_q <- mapply(influence_function,r = rep(0, n_q),y = y0_q,x = x_q)
  
  variance_hat <- (sum(IF_p^2)+sum(IF_q^2))/n^2
  
  se_hat <- sqrt(variance_hat)
  
  # Store the results
  results[i, ] <- c(estimate = theta_hat,variance = variance_hat,se = se_hat)
}

# Save results

write.csv(results,file="bruteforce.csv",row.names=FALSE)