# test data.
rm(list=ls())

# create test data.
m <- 50
x1 <- runif(m, min=1, max=10)
x2 <- rnorm(m, mean=0, sd=1)
x3 <- rnorm(m, mean=0, sd=1)

# feature scaling.
x1_s <- (x1-mean(x1))/(max(x1)-min(x1))
x2_s <- (x2-mean(x2))/(max(x2)-min(x2))
x3_s <- (x3-mean(x3))/(max(x3)-min(x3))

# binomial logistic regression.


# create artificial data.
beta1 <- c(2,0.1,3,-3)
p <- 1/(1 + exp(-(beta1[1] + beta1[2]*x1 + beta1[3]*x2 + beta1[4]*x3)))

dd <- data.frame(x1=x1, x2=x2, x3=x3, x1_s=x1_s, x2_s=x2_s, x3_s=x3_s)
U <- runif(nrow(dd))
dd$y <- as.numeric(U < p)

computeCost <- function(w, dd)  {
  p <- 1/(1+exp(-(w[1] + w[2]*dd$x1_s + w[3]*dd$x2_s + w[4]*dd$x3_s)))
    
  y0 <- dd$y == 0
  y1 <- !y0
  
  cost <- -sum(log(1-p[y0])) - sum(log(p[y1]))
  cost
}

computePartialD <- function(w,dd,index)  {
  epsilon <- 0.0001
  w1 <- w
  w2 <- w
  w1[index] <- w1[index] - epsilon
  w2[index] <- w2[index] + epsilon

  cost1 <- computeCost(w1, dd)
  cost2 <- computeCost(w2, dd)
  
  (cost2-cost1)/(2*epsilon)
}


# find the weights with GLM function.
g <- glm(y ~ x1_s + x2_s + x3_s, data=dd, family=binomial)
summary(g)

# batch gradient descent.
w <- c(1,1,1,1)
dcostdw <- numeric(4)
l2 <- 1
alpha <- 0.1
iter <- 1
ep <- 0.005
while (iter < 3000)  {
  
  p <- with(dd, 1/(1+exp(-(w[1] + w[2]*x1_s + w[3]*x2_s + w[4]*x3_s))))
  
  dcostdw[1] <- with(dd, -sum((y-p)))
  dcostdw[2] <- with(dd, -sum(x1_s*(y-p)))
  dcostdw[3] <- with(dd, -sum(x2_s*(y-p)))
  dcostdw[4] <- with(dd, -sum(x3_s*(y-p)))
  
  # numerical gradient checking.
#    print(paste(dcostdw[1], dcostdw[2], dcostdw[3], dcostdw[4],":", computePartialD(w, dd,1), computePartialD(w,dd,2), computePartialD(w,dd,3), computePartialD(w,dd,4)))

  
  w[1] <- w[1] - alpha*dcostdw[1]
  w[2] <- w[2] - alpha*dcostdw[2]
  w[3] <- w[3] - alpha*dcostdw[3]
  w[4] <- w[4] - alpha*dcostdw[4]
  
  l2 <- sqrt(dcostdw[1]^2 + dcostdw[2]^2 + dcostdw[3]^2 + dcostdw[4]^2)
  print(paste("iter", iter, " - ",computeCost(w,dd)))
  
  iter <- iter + 1
}


# standard errors.
p <- with(dd, 1/(1+exp(-(w[1] + w[2]*x1_s + w[3]*x2_s + w[4]*x3_s))))

# compute the observed information matrix for the sample (estimating nI(w))
I <- matrix(0, nrow=4, ncol=4)
I[1,1] <- with(dd, sum(p*(1-p)))
I[1,2] <- with(dd, sum(x1_s*p*(1-p)))
I[1,3] <- with(dd, sum(x2_s*p*(1-p)))
I[1,4] <- with(dd, sum(x3_s*p*(1-p)))

I[2,1] <- I[1,2]
I[2,2] <- with(dd, sum(x1_s^2*p*(1-p)))
I[2,3] <- with(dd, sum(x1_s*x2_s*p*(1-p)))
I[2,4] <- with(dd, sum(x1_s*x3_s*p*(1-p)))

I[3,1] <- I[1,3]
I[3,2] <- I[2,3]
I[3,3] <- with(dd, sum(x2_s^2*p*(1-p)))
I[3,4] <- with(dd, sum(x2_s*x3_s*p*(1-p)))

I[4,1] <- I[1,4]
I[4,2] <- I[2,4]
I[4,3] <- I[3,4]
I[4,4] <- with(dd, sum(x3_s^2*p*(1-p)))

Var <- solve(I)
sqrt(Var[1,1])  # standard error estimate for w0
sqrt(Var[2,2])  # standard error estimate for w1
sqrt(Var[3,3])  # standard error estimate for w2
sqrt(Var[4,4])  # standard error estimate for w3
