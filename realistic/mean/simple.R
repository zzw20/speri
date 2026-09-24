# ============================================================
# Realistic simulation
# Target: Mean general response difference
# Estimator: Simple
# ============================================================

# Simulation settings

B <- 1000

n.p <- 1000
n.q <- 1500

p2.p <- 0.31
p2.q <- 0.36

params.p <- matrix(c(1.70, 0.81,
                     4.97, 1.70,
                     1.07, 7.21),
                   byrow = TRUE, nrow = 2)

params.q <- matrix(c(2.22, 2.60,
                     0.97, 4.18,
                     2.60, 0.70),
                   byrow = TRUE, nrow = 2)

beta.1 <- c(-2, -26, -5, 18)
beta.0 <- c(-3, -21, -12, 23)

sig1 <- 9.32
sig0 <- 10.13

# Generate continuous covariates

make.beta.block <- function(n, params) {
  sapply(seq_len(ncol(params)), function(j) {
    rbeta(n, shape1 = params[1, j], shape2 = params[2, j])
  })
}

# Storage

results <- matrix(NA_real_, nrow = B, ncol = 3)

# Monte Carlo simulation

for (b in 1:B) {
  
  set.seed(b)
  
  # Generate X
  
  sex.p <- 1 + rbinom(n.p, 1, p2.p)
  sex.q <- 1 + rbinom(n.q, 1, p2.q)
  
  cont.p <- make.beta.block(n.p, params.p)
  cont.q <- make.beta.block(n.q, params.q)
  
  x.p <- cbind(sex.p, cont.p)
  x.q <- cbind(sex.q, cont.q)
  
  # Generate outcomes Y
  
  y1.p <- as.numeric(
    63 + x.p %*% beta.1 + rnorm(n.p, 0, sig1)
  )
  
  y0.q <- as.numeric(
    56 + x.q %*% beta.0 + rnorm(n.q, 0, sig0)
  )
  
  # Simple estimator
  
  theta.p1.hat <- mean(y1.p)
  theta.p0.hat <- mean(y0.q)
  
  theta.hat <- theta.p1.hat-theta.p0.hat
  
  # Variance estimation
  
  variance.hat <- var(y1.p)/n.p + var(y0.q)/n.q
  se.hat <- sqrt(variance.hat)
  
  # Store results
  
  results[b, ] <- c(theta.hat, variance.hat, se.hat)
}

# Save results

results <- as.data.frame(results)
colnames(results) <- c("estimate", "variance", "se")

write.csv(results, "simple.csv", row.names = FALSE)