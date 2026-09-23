# ============================================================
# Low-dimensional simulation
# Target: Median general response difference
# Estimator: Outcome model
# ============================================================

library(truncnorm)

# Simulation settings

n_rep <- 1000

n_p <- 1000
n_q <- 1000
n <- n_p+n_q
pi_p <- n_p/n

beta_1 <- 2
beta_0 <- 1

# Bandwidth for kernel estimation
h <- 0.2

# Gaussian kernel function

kern <- function(cond,cond_data,h) {
  exp(-0.5*apply(as.matrix(cond_data),1,function(x) sum((cond-x)^2))/h^2)/
    (sqrt(2*pi)*h)
}

# Storage

results <- matrix(NA_real_,nrow=n_rep,ncol=3,
                  dimnames=list(NULL,c("estimate","variance","se")))

# Monte Carlo simulation

for (i in 1:n_rep) {
  
  set.seed(i)
  
  # Generate X
  
  x_p <- runif(n_p,-1,1)
  x_q <- rtruncnorm(n_q,-1,1,0.5,1)
  
  # Generate outcomes Y
  
  y1_p <- beta_1*x_p+rnorm(n_p,0,1)
  y1_q <- beta_1*x_q+rnorm(n_q,0,1)
  y0_p <- beta_0*x_p+rnorm(n_p,0,1)
  y0_q <- beta_0*x_q+rnorm(n_q,0,1)
  
  # Outcome model estimator
  
  theta_p1_hat <- median(y1_p)
  
  b_temp <- function(theta_p0,x) {
    result <- mean(kern(x,x_q,h)*((y0_q <= theta_p0)-0.5))/
      mean(kern(x,x_q,h))
    ifelse(is.nan(result),0,result)
  }
  
  b_sum <- function(theta_p0,x) {
    result <- mapply(b_temp,x=x,MoreArgs=list(theta_p0=theta_p0))
    sum(result)
  }
  
  theta_p0_hat <- uniroot(b_sum,c(-1,1),x=x_p)$root
  
  theta_hat <- theta_p1_hat-theta_p0_hat
  
  # Variance estimation
  
  rho_hat <- function(x) {
    result <- mean(kern(x,x_p,h))/mean(kern(x,x_q,h))
    ifelse(is.nan(result),0,result)
  }
  
  b_hat <- function(x) {
    result <- mean(kern(x,x_q,h)*((y0_q <= theta_p0_hat)-0.5))/
      mean(kern(x,x_q,h))
    ifelse(is.nan(result),0,result)
  }
  
  density_est <- density(y1_p,kernel="gaussian",bw="nrd0")
  B1 <- 1/approx(density_est$x,density_est$y,xout=theta_p1_hat)$y
  
  h2 <- 0.2
  kernel_y0 <- dnorm((y0_q-theta_p0_hat)/h2)/h2
  
  rho_q <- sapply(x_q,rho_hat)
  B0 <- 1/(sum(rho_q*kernel_y0)/n_q)
  
  influence_function <- function(r,y,x) {
    r/pi_p*(B0*b_hat(x)-B1*((y <= theta_p1_hat)-0.5))+
      (1-r)/(1-pi_p)*B0*(((y <= theta_p0_hat)-0.5)-b_hat(x))*rho_hat(x)
  }
  
  IF_p <- mapply(influence_function,rep(1,n_p),y1_p,x_p)
  IF_q <- mapply(influence_function,rep(0,n_q),y0_q,x_q)
  
  variance_hat <- (sum(IF_p^2)+sum(IF_q^2))/n^2
  se_hat <- sqrt(variance_hat)
  
  # Store the results
  
  results[i, ] <- c(estimate=theta_hat,variance=variance_hat,se=se_hat)
}

# Save results

write.csv(results,file="outcome_model.csv",row.names=FALSE)