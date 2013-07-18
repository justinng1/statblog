###################################
# Author:  Justin Ng
# Email:   justinng1@gmail.com
# Website: http://justinng1.wordpress.com
#
# Post:    http://justinng1.wordpress.com/2013/07/16/multinomial-logistic-regression/
###################################

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

# multinomial logistic regression.


# create artificial data.
beta1 <- c(-1,0.1,0.5,-3)
beta2 <- c(-0.7,0.1,-2.5,3)
c1 <- exp(beta1[1] + beta1[2]*x1 + beta1[3]*x2 + beta1[4]*x3)
c2 <- exp(beta2[1] + beta2[2]*x1 + beta2[3]*x2 + beta2[4]*x3)
den <- 1 + c1 + c2
p1 <- c1/den
p2 <- c2/den
p3 <- 1-p1-p2

dd <- data.frame(x1=x1, x2=x2, x3=x3, x1_s=x1_s, x2_s=x2_s, x3_s=x3_s)
U <- runif(nrow(dd))
dd$y1 <- ifelse(U < p1, 1, 0)
dd$y2 <- ifelse(U > (1-p2), 1, 0)
dd$y3 <- as.numeric(!(dd$y1 | dd$y2))
dd$y <- ifelse(dd$y1 == 1, 1, ifelse(dd$y2 == 1, 2, 3))

# w[num, node]
computeCost <- function(w, dd)  {
  c1 <- with(dd, exp(w[1,1] + w[2,1]*x1_s + w[3,1]*x2_s + w[4,1]*x3_s))
  c2 <- with(dd, exp(w[1,2] + w[2,2]*x1_s + w[3,2]*x2_s + w[4,2]*x3_s))
  den <- 1 + c1 + c2
  p1 <- c1/den
  p2 <- c2/den
  p3 <- 1-p1-p2
  
  y1 <- dd$y1 == 1
  y2 <- dd$y2 == 1
  y3 <- !(dd$y1 | dd$y2)
  
  cost <- -sum(log(p1[y1])) - sum(log(p2[y2])) - sum(log(p3[y3]))
  cost
}

computePartialD <- function(w,dd,num,node)  {
  epsilon <- 0.0001
  w1 <- w
  w2 <- w
  w1[num, node] <- w1[num, node] - epsilon
  w2[num, node] <- w2[num, node] + epsilon
  
  cost1 <- computeCost(w1, dd)
  cost2 <- computeCost(w2, dd)
  
  (cost2-cost1)/(2*epsilon)
}


# batch gradient descent.
w <- matrix(0, nrow=4, ncol=3)
dcostdw <- matrix(0, nrow=nrow(w), ncol=ncol(w))
l2 <- 1
alpha <- 0.5
iter <- 1
ep <- 0.005

y_1 <- dd$y1 == 1
y_2 <- dd$y2 == 1
y_3 <- dd$y3 == 1

dd1 <- dd[y_1, ]
dd2 <- dd[y_2, ]
dd3 <- dd[y_3, ]
while (iter < 10000)  {
  
  c1 <- with(dd, exp(w[1,1] + w[2,1]*x1_s + w[3,1]*x2_s + w[4,1]*x3_s))
  c2 <- with(dd, exp(w[1,2] + w[2,2]*x1_s + w[3,2]*x2_s + w[4,2]*x3_s))
  den <- 1 + c1 + c2
  p1 <- c1/den
  p2 <- c2/den
  p3 <- 1-p1-p2
  
  # num, node
  p1_1 <- p1[y_1]
  p1_2 <- p1[y_2]
  p1_3 <- p1[y_3]
  
  p2_1 <- p2[y_1]
  p2_2 <- p2[y_2]
  p2_3 <- p2[y_3]
  
  p3_1 <- p3[y_1]
  p3_2 <- p3[y_2]
  p3_3 <- p3[y_3]
  
  
  dcostdw[1,1] <- -sum(1-p1_1) + sum(p1_2) + sum(p1_3)
  dcostdw[2,1] <- -sum(dd1$x1_s*(1-p1_1)) + sum(dd2$x1_s*p1_2) + sum(dd3$x1_s*p1_3)
  dcostdw[3,1] <- -sum(dd1$x2_s*(1-p1_1)) + sum(dd2$x2_s*p1_2) + sum(dd3$x2_s*p1_3)
  dcostdw[4,1] <- -sum(dd1$x3_s*(1-p1_1)) + sum(dd2$x3_s*p1_2) + sum(dd3$x3_s*p1_3)
  
  dcostdw[1,2] <- sum(p2_1) - sum(1-p2_2) + sum(p2_3)
  dcostdw[2,2] <- sum(dd1$x1_s*p2_1) - sum(dd2$x1_s*(1-p2_2)) + sum(dd3$x1_s*p2_3)
  dcostdw[3,2] <- sum(dd1$x2_s*p2_1) - sum(dd2$x2_s*(1-p2_2)) + sum(dd3$x2_s*p2_3)
  dcostdw[4,2] <- sum(dd1$x3_s*p2_1) - sum(dd2$x3_s*(1-p2_2)) + sum(dd3$x3_s*p2_3)
  
  
  # numerical gradient checking.
#      print(paste(dcostdw[1,1], dcostdw[2,1], dcostdw[3,1], dcostdw[4,1],":", computePartialD(w, dd,1,1), computePartialD(w,dd,2,1), computePartialD(w,dd,3,1), computePartialD(w,dd,4,1)))
  w <- w - alpha*dcostdw
  
#   l2 <- sqrt(dcostdw[1]^2 + dcostdw[2]^2 + dcostdw[3]^2 + dcostdw[4]^2)
  print(paste("iter", iter, " - ",computeCost(w,dd)))
  
  iter <- iter + 1
}

# standard errors.
c1 <- with(dd, exp(w[1,1] + w[2,1]*x1_s + w[3,1]*x2_s + w[4,1]*x3_s))
c2 <- with(dd, exp(w[1,2] + w[2,2]*x1_s + w[3,2]*x2_s + w[4,2]*x3_s))
den <- 1 + c1 + c2
p1 <- c1/den
p2 <- c2/den
p3 <- 1-p1-p2

# compute the observed information matrix for the sample (estimating nI(w))
I <- matrix(0, nrow=8, ncol=8)
I[1,1] <- with(dd, -sum(p1*(1-p1)))  #w01^2
I[1,2] <- with(dd, -sum(x1_s*p1*(1-p1)))  #w01w11
I[1,3] <- with(dd, -sum(x2_s*p1*(1-p1)))  #w01w21
I[1,4] <- with(dd, -sum(x3_s*p1*(1-p1)))  #w01w31
I[1,5] <- with(dd, sum(p1*p2))  #w01w02
I[1,6] <- with(dd, sum(x1_s*p1*p2))  #w01w12
I[1,7] <- with(dd, sum(x2_s*p1*p2))  #w01w22
I[1,8] <- with(dd, sum(x3_s*p1*p2))  #w01w32

I[2,1] <- I[1,2]
I[2,2] <- with(dd, -sum(x1_s^2*p1*(1-p1)))  #w11^2
I[2,3] <- with(dd, -sum(x1_s*x2_s*p1*(1-p1)))  #w11w21
I[2,4] <- with(dd, -sum(x1_s*x3_s*p1*(1-p1)))  #w11w31
I[2,5] <- with(dd, sum(x1_s*p1*p2))  #w11w02
I[2,6] <- with(dd, sum(x1_s^2*p1*p2))  #w11w12
I[2,7] <- with(dd, sum(x1_s*x2_s*p1*p2))  #w11w22
I[2,8] <- with(dd, sum(x1_s*x3_s*p1*p2))  #w11w32

I[3,1] <- I[1,3]
I[3,2] <- I[2,3]
I[3,3] <- with(dd, -sum(x2_s^2*p1*(1-p1)))  #w21^2
I[3,4] <- with(dd, -sum(x2_s*x3_s*p1*(1-p1)))  #w21w31
I[3,5] <- with(dd, sum(x2_s*p1*p2))  #w21w02
I[3,6] <- with(dd, sum(x2_s*x1_s*p1*p2))  #w21w12
I[3,7] <- with(dd, sum(x2_s^2*p1*p2))  #w21w22
I[3,8] <- with(dd, sum(x2_s*x3_s*p1*p2))  #w21w32

I[4,1] <- I[1,4]  #w31w01
I[4,2] <- I[2,4]  #w31w11
I[4,3] <- I[3,4]  #w31w21
I[4,4] <- with(dd, -sum(x3_s^2*p1*(1-p1)))  #w31^2
I[4,5] <- with(dd, sum(x3_s*p1*p2))  #w31w02
I[4,6] <- with(dd, sum(x3_s*x1_s*p1*p2))  #w31w12
I[4,7] <- with(dd, sum(x3_s*x2_s*p1*p2))  #w31w22
I[4,8] <- with(dd, sum(x3_s^2*p1*p2))  #w31w32

I[5,1] <- I[1,5]  #w02w01
I[5,2] <- I[2,5]  #w02w11
I[5,3] <- I[3,5]  #w02w21
I[5,4] <- I[4,5]  #w02w31
I[5,5] <- with(dd, -sum(p2*(1-p2)))  #w02^2
I[5,6] <- with(dd, -sum(x1_s*p2*(1-p2)))  #w02w12
I[5,7] <- with(dd, -sum(x2_s*p2*(1-p2)))  #w02w22
I[5,8] <- with(dd, -sum(x3_s*p2*(1-p2)))  #w02w32

I[6,1] <- I[1,6]  #w12w01
I[6,2] <- I[2,6]  #w12w11
I[6,3] <- I[3,6]  #w12w21
I[6,4] <- I[4,6]  #w12w31
I[6,5] <- I[5,6]  #w12w02
I[6,6] <- with(dd, -sum(x1_s^2*p2*(1-p2)))  #w12^2
I[6,7] <- with(dd, -sum(x1_s*x2_s*p2*(1-p2)))  #w12w22
I[6,8] <- with(dd, -sum(x1_s*x3_s*p2*(1-p2)))  #w12w32

I[7,1] <- I[1,7]  #w22w01
I[7,2] <- I[2,7]  #w22w11
I[7,3] <- I[3,7]  #w22w21
I[7,4] <- I[4,7]  #w22w31
I[7,5] <- I[5,7]  #w22w02
I[7,6] <- I[6,7]  #w22w12
I[7,7] <- with(dd, -sum(x2_s^2*p2*(1-p2)))  #w22^2
I[7,8] <- with(dd, -sum(x2_s*x3_s*p2*(1-p2)))  #w22w32

I[8,1] <- I[1,8]  #w32w01
I[8,2] <- I[2,8]  #w32w11
I[8,3] <- I[3,8]  #w32w21
I[8,4] <- I[4,8]  #w32w31
I[8,5] <- I[5,8]  #w32w02
I[8,6] <- I[6,8]  #w32w12
I[8,7] <- I[7,8]  #w32w22
I[8,8] <- with(dd, -sum(x3_s^2*p2*(1-p2)))  #w32^2

I <- -I

Var <- solve(I)
sqrt(Var[1,1])  # standard error estimate for w01
sqrt(Var[2,2])  # standard error estimate for w11
sqrt(Var[3,3])  # standard error estimate for w21
sqrt(Var[4,4])  # standard error estimate for w31
sqrt(Var[5,5])  # standard error estimate for w02
sqrt(Var[6,6])  # standard error estimate for w12
sqrt(Var[7,7])  # standard error estimate for w22
sqrt(Var[8,8])  # standard error estimate for w32

