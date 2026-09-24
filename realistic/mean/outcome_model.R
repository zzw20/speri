# ============================================================
# Realistic simulation
# Target: Mean general response difference
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

kern1d <- function(u) {
  ((3-u^2)*dnorm(u))/2
}

prod.kernel.weights <- function(x, data, h) {
  if (length(h) == 1L) h <- rep(h, ncol(data))
  u <- sweep(data, 2, x, FUN = "-")
  u <- sweep(u, 2, h, FUN = "/")
  K <- kern1d(u)
  K <- sweep(K, 2, h, FUN = "/")
  apply(K, 1, prod)
}

c0 <- 1.3
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
  
  y1.p <- as.numeric(
    63/sig1 + x.p %*% (beta.1/sig1) + rnorm(n.p, 0, 1)
  )
  
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
  
  # theta_p1
  
  theta.p1.hat <- mean(y1.p)
  
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
  
  # Outcome model for point estimation
  
  b.est <- function(x) {
    
    key <- as.character(x[1])
    
    Xq.g <- Xq.list[[key]]
    y0.g <- y0.list[[key]]
    h <- h.list[[key]]
    
    if (is.null(Xq.g) || is.null(y0.g) || is.null(h)) return(0)
    
    wq <- prod.kernel.weights(x[2:4], Xq.g[, 2:4, drop = FALSE], h)
    
    num <- mean(wq*(theta.p1.hat-y0.g))
    num <- num+sign(num)*1e-3
    
    den <- mean(wq)
    den <- den+sign(den)*1e-3
    
    num/den
  }
  
  # Outcome-model estimator
  
  theta.hat <- mean(apply(x.p, 1, b.est))
  theta.p0.hat <- theta.p1.hat-theta.hat
  
  # Outcome model for variance estimation
  
  b.IF <- function(x) {
    
    key <- as.character(x[1])
    
    Xq.g <- Xq.list[[key]]
    y0.g <- y0.list[[key]]
    h <- h.list[[key]]
    
    if (is.null(Xq.g) || is.null(y0.g) || is.null(h)) return(0)
    
    wq <- prod.kernel.weights(x[2:4], Xq.g[, 2:4, drop = FALSE], h)
    
    num <- mean(wq*(y0.g-theta.p0.hat))
    num <- num+sign(num)*1e-4
    
    den <- mean(wq)
    den <- den+sign(den)*1e-4
    
    num/den
  }
  
  # Variance estimation
  
  eff.func <- function(r, y, x) {
    r/pi.p*(-b.IF(x)+(y-theta.p1.hat)) +
      (1-r)/(1-pi.p)*(-(y-theta.p0.hat-b.IF(x))*rho(x))
  }
  
  eff.p <- sapply(1:n.p, function(i) {
    eff.func(1, y1.p[i], x.p[i, ])
  })
  
  eff.q <- sapply(1:n.q, function(i) {
    eff.func(0, y0.q[i], x.q[i, ])
  })
  
  IF <- c(eff.p, eff.q)
  
  variance.hat <- mean(IF^2)/n
  se.hat <- sqrt(variance.hat)
  
  # Store results
  
  results[b, ] <- c(theta.hat, variance.hat, se.hat)
}

# Save results

results <- as.data.frame(results)
colnames(results) <- c("estimate", "variance", "se")

write.csv(results, "outcome_model.csv", row.names = FALSE)