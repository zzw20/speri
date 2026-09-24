# ============================================================
# Realistic simulation
# Target: Median general response difference
# Estimator: Outcome-model
# ============================================================

# Simulation settings

B <- 1000

# Larger sample sizes are used for a more accurate asymptotic approximation
# for the outcome-model estimator.
n.p <- 3000
n.q <- 4500
n <- n.p+n.q
pi.p <- n.p/n

p2.p <- 0.31
p2.q <- 0.36

params.p <- matrix(c(1.6995, 0.8077,
                     4.9653, 1.6976,
                     1.0739, 7.2052),
                   byrow = TRUE, nrow = 2)

params.q <- matrix(c(2.2231, 2.5977,
                     0.9652, 4.1756,
                     2.5968, 0.6989),
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


# Kernel functions

kern1d <- function(u) {((3-u^2)*dnorm(u))/2}

prod.kernel.weights <- function(x, data, h) {
  if (length(h) == 1L) h <- rep(h, ncol(data))
  u <- sweep(data, 2, x, FUN = "-")
  u <- sweep(u, 2, h, FUN = "/")
  K <- kern1d(u)
  K <- sweep(K, 2, h, FUN = "/")
  apply(K, 1, prod)
}

c0 <- 1.55
d <- 3

bandwidth <- function(x) {
  c0*sqrt(d)*sd(x)*length(x)^(-1/(2*d+1))
}


# Storage

results <- matrix(NA_real_, nrow = B, ncol = 3)


# Monte Carlo simulation

for (b in 1:B) {
  
  set.seed(b)
  
  # Generate X
  
  sex.q <- 1+rbinom(n.q, 1, p2.q)
  sex.p <- 1+rbinom(n.p, 1, p2.p)
  
  cont.q <- make.beta.block(n.q, params.q)
  cont.p <- make.beta.block(n.p, params.p)
  
  x.q <- cbind(sex.q, cont.q)
  x.p <- cbind(sex.p, cont.p)
  
  # Generate outcomes Y
  
  y0.q <- as.numeric(
    56/sig0 + x.q %*% (beta.0/sig0) + rnorm(n.q, 0, 1)
  )
  
  y1.p <- as.numeric(63/sig1 + x.p %*% (beta.1/sig1) + rnorm(n.p, 0, 1))
  
  # Build sex-specific data
  
  sex.values <- sort(unique(x.q[, 1]))
  
  Xq.list <- setNames(vector("list", length(sex.values)), sex.values)
  Xp.list <- setNames(vector("list", length(sex.values)), sex.values)
  y0.list <- setNames(vector("list", length(sex.values)), sex.values)
  h.list <- setNames(vector("list", length(sex.values)), sex.values)
  
  for (g in sex.values) {
    
    idx.q <- which(x.q[, 1] == g)
    idx.p <- which(x.p[, 1] == g)
    
    if (length(idx.q) == 0L || length(idx.p) == 0L) next
    
    Xq.list[[as.character(g)]] <- x.q[idx.q, , drop = FALSE]
    Xp.list[[as.character(g)]] <- x.p[idx.p, , drop = FALSE]
    y0.list[[as.character(g)]] <- y0.q[idx.q]
    
    h.list[[as.character(g)]] <- apply(
      x.q[idx.q, 2:4, drop = FALSE], 2, bandwidth
    )
  }
  
  
  # Density ratio
  
  rho <- function(x) {
    
    key <- as.character(x[1])
    
    Xq.g <- Xq.list[[key]]
    Xp.g <- Xp.list[[key]]
    h <- h.list[[key]]
    
    if (is.null(Xq.g) || is.null(Xp.g) || is.null(h)) return(0)
    
    wp <- prod.kernel.weights(x[2:4], Xp.g[, 2:4, drop = FALSE], h)
    
    wq <- prod.kernel.weights(x[2:4], Xq.g[, 2:4, drop = FALSE], h)
    
    out <- mean(wp)/mean(wq)
    
    ifelse(is.finite(out), out, 0)
  }
  
  
  # Conditional sign-score model
  
  b.point <- function(theta, x) {
    
    key <- as.character(x[1])
    
    Xq.g <- Xq.list[[key]]
    y0.g <- y0.list[[key]]
    h <- h.list[[key]]
    
    if (is.null(Xq.g) || is.null(y0.g) || is.null(h)) return(0)
    
    wq <- prod.kernel.weights(x[2:4], Xq.g[, 2:4, drop = FALSE], h)
    
    z <- as.numeric((y0.g <= theta)-0.5)
    
    num <- mean(wq*z)
    num <- num+sign(num)*1e-4
    
    den <- mean(wq)
    den <- den+sign(den)*1e-4
    
    num/den
  }
  
  b.sum <- function(theta) {
    mean(apply(x.p, 1,function(x) b.point(theta, x)))
  }
  
  
  # Outcome-model estimator
  
  theta.p1.hat <- median(y1.p)
  
  bracket <- quantile(y0.q,c(0.01, 0.99),names = FALSE)
  
  score.left <- b.sum(bracket[1])
  score.right <- b.sum(bracket[2])
  
  if (!is.finite(score.left) ||
      !is.finite(score.right) ||
      score.left*score.right > 0) {
    
    rho.q <- apply(x.q, 1, rho)
    rho.q <- pmin(pmax(rho.q, 1e-3), 1e3)
    
    theta.p0.hat <- matrixStats::weightedMedian(y0.q,w = rho.q)
    
  } else {
    
    theta.p0.hat <- uniroot(function(theta) b.sum(theta),interval = bracket)$root
  }
  
  theta.hat <- theta.p1.hat-theta.p0.hat
  
  
  # Conditional sign score for variance estimation
  
  b.sign <- function(x) {
    b.point(theta.p0.hat, x)
  }
  
  
  # Density at theta_p1
  
  dens1 <- density(y1.p,kernel = "gaussian",bw = "nrd0")
  
  f1.hat <- as.numeric(approx(dens1$x,dens1$y,xout = theta.p1.hat,rule = 2)$y)
  
  f1.hat <- min(max(f1.hat, 1e-4), 10)
  b1 <- 1/f1.hat
  
  
  # Density at theta_p0 under P
  
  rho.q <- apply(x.q, 1, rho)
  rho.q <- pmin(pmax(rho.q, 1e-3), 1e3)
  
  h2 <- 0.2
  
  kern.y <- dnorm((y0.q-theta.p0.hat)/h2)/h2
  
  f0.hat <- mean(rho.q*kern.y)
  f0.hat <- min(max(f0.hat, 1e-4), 10)
  
  b0 <- 1/f0.hat
  
  
  # Variance estimation
  
  u1 <- as.numeric((y1.p <= theta.p1.hat)-0.5)
  u0 <- as.numeric((y0.q <= theta.p0.hat)-0.5)
  
  b.p <- apply(x.p, 1, b.sign)
  b.q <- apply(x.q, 1, b.sign)
  
  eff.p <- 1/pi.p*(b0*b.p-b1*u1)
  
  eff.q <- 1/(1-pi.p)*b0*(u0-b.q)*rho.q
  
  IF <- c(eff.p, eff.q)
  
  variance.hat <- mean(IF^2)/n
  se.hat <- sqrt(variance.hat)
  
  
  # Store results
  
  results[b, ] <- c(theta.hat,variance.hat,se.hat)
}


# Save results

results <- as.data.frame(results)
colnames(results) <- c("estimate", "variance", "se")

write.csv(results, "outcome_model.csv", row.names = FALSE)