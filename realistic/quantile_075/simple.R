# ============================================================
# Realistic simulation
# Target: 0.75-quantile general response difference
# Estimator: Simple
# ============================================================

# Simulation settings

B <- 1000
n.p <- 1000
n.q <- 1500
tau <- 0.75

p2.p <- 0.31
p2.q <- 0.36

params.p <- matrix(c(1.6995, 0.8077,
                     4.9653, 1.6976,
                     1.0739, 7.2052),
                   byrow=TRUE, nrow=2)

params.q <- matrix(c(2.2231, 2.5977,
                     0.9652, 4.1756,
                     2.5968, 0.6989),
                   byrow=TRUE, nrow=2)

beta.1 <- c(-2, -26, -5, 18)
beta.0 <- c(-3, -21, -12, 23)

sig1 <- 9.32
sig0 <- 10.13


# Generate continuous covariates

make.beta.block <- function(n, params) {
  sapply(seq_len(ncol(params)), function(j) {
    rbeta(n, shape1=params[1,j], shape2=params[2,j])
  })
}


# Storage

results <- matrix(NA_real_,nrow=B,ncol=3)


# Monte Carlo simulation

for (b in 1:B) {
  
  set.seed(b)
  
  # Generate X
  
  sex.q <- 1+rbinom(n.q,1,p2.q)
  sex.p <- 1+rbinom(n.p,1,p2.p)
  
  cont.q <- make.beta.block(n.q,params.q)
  cont.p <- make.beta.block(n.p,params.p)
  
  x.q <- cbind(sex.q,cont.q)
  x.p <- cbind(sex.p,cont.p)
  
  # Generate outcomes Y
  
  y0.q <- as.numeric(56/sig0+x.q%*%(beta.0/sig0)+rnorm(n.q,0,1))
  y1.p <- as.numeric(63/sig1+x.p%*%(beta.1/sig1)+rnorm(n.p,0,1))
  
  
  # Simple estimator
  
  theta.p1.hat <- as.numeric(quantile(y1.p,tau))
  theta.q0.hat <- as.numeric(quantile(y0.q,tau))
  
  theta.hat <- theta.p1.hat-theta.q0.hat
  
  
  # Variance estimation
  
  dens.y1 <- density(y1.p)
  f1.hat <- approx(dens.y1$x,dens.y1$y,xout=theta.p1.hat)$y
  
  dens.y0 <- density(y0.q)
  f0.hat <- approx(dens.y0$x,dens.y0$y,xout=theta.q0.hat)$y
  
  variance.hat <- tau*(1-tau)/(n.p*f1.hat^2) +
    tau*(1-tau)/(n.q*f0.hat^2)
  
  se.hat <- sqrt(variance.hat)
  
  
  # Store results
  
  results[b,] <- c(theta.hat,variance.hat,se.hat)
}


# Save results

results <- as.data.frame(results)
colnames(results) <- c("estimate","variance","se")

write.csv(results,"simple.csv",row.names=FALSE)