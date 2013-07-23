###################################
# Author:  Justin Ng
# Email:   justinng1@gmail.com
# Website: http://justinng1.wordpress.com
#
# Post:    http://justinng1.wordpress.com/2013/07/22/regularized-cost-functions/
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
c1 <- exp(beta1[1] + beta1[2]*x1)
c2 <- exp(beta2[1] + beta2[2]*x1)
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
computeCost <- function(w, dd, alpha)  {
  c1 <- with(dd, exp(w[1,1] + w[2,1]*x1_s + w[3,1]*x2_s + w[4,1]*x3_s))
  c2 <- with(dd, exp(w[1,2] + w[2,2]*x1_s + w[3,2]*x2_s + w[4,2]*x3_s))
  den <- 1 + c1 + c2
  p1 <- c1/den
  p2 <- c2/den
  p3 <- 1-p1-p2
  
  y1 <- dd$y1 == 1
  y2 <- dd$y2 == 1
  y3 <- !(dd$y1 | dd$y2)
  L2 <- sum(w[2:nrow(w),]^2)
  
  cost <- -sum(log(p1[y1])) - sum(log(p2[y2])) - sum(log(p3[y3])) + alpha/2*L2
  cost
}

computePartialD <- function(w,dd,alpha,num,node)  {
  epsilon <- 0.0001
  w1 <- w
  w2 <- w
  w1[num, node] <- w1[num, node] - epsilon
  w2[num, node] <- w2[num, node] + epsilon
  
  cost1 <- computeCost(w1, dd, alpha)
  cost2 <- computeCost(w2, dd, alpha)
  
  (cost2-cost1)/(2*epsilon)
}


# batch gradient descent.
w <- matrix(0, nrow=4, ncol=3)
dcostdw <- matrix(0, nrow=nrow(w), ncol=ncol(w))
l2 <- 1
pconst <- 0.0000001
alpha <- 0.1
iter <- 1
ep <- 0.005

y_1 <- dd$y1 == 1
y_2 <- dd$y2 == 1
y_3 <- dd$y3 == 1

dd1 <- dd[y_1, ]
dd2 <- dd[y_2, ]
dd3 <- dd[y_3, ]
while (iter < 10)  {
  
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
  dcostdw[2,1] <- -sum(dd1$x1_s*(1-p1_1)) + sum(dd2$x1_s*p1_2) + sum(dd3$x1_s*p1_3) + pconst*w[2,1]
  dcostdw[3,1] <- -sum(dd1$x2_s*(1-p1_1)) + sum(dd2$x2_s*p1_2) + sum(dd3$x2_s*p1_3) + pconst*w[3,1]
  dcostdw[4,1] <- -sum(dd1$x3_s*(1-p1_1)) + sum(dd2$x3_s*p1_2) + sum(dd3$x3_s*p1_3) + pconst*w[4,1]
  
  dcostdw[1,2] <- sum(p2_1) - sum(1-p2_2) + sum(p2_3)
  dcostdw[2,2] <- sum(dd1$x1_s*p2_1) - sum(dd2$x1_s*(1-p2_2)) + sum(dd3$x1_s*p2_3) + pconst*w[2,2]
  dcostdw[3,2] <- sum(dd1$x2_s*p2_1) - sum(dd2$x2_s*(1-p2_2)) + sum(dd3$x2_s*p2_3) + pconst*w[3,2]
  dcostdw[4,2] <- sum(dd1$x3_s*p2_1) - sum(dd2$x3_s*(1-p2_2)) + sum(dd3$x3_s*p2_3) + pconst*w[4,2]
  
  
  # numerical gradient checking.
     print(paste(dcostdw[1,1], dcostdw[2,1], dcostdw[3,1], dcostdw[4,1],":", computePartialD(w, dd,pconst,1,1), computePartialD(w,dd,pconst,2,1), computePartialD(w,dd,pconst,3,1), computePartialD(w,dd,pconst,4,1)))
  w <- w - alpha*dcostdw
  
#   l2 <- sqrt(dcostdw[1]^2 + dcostdw[2]^2 + dcostdw[3]^2 + dcostdw[4]^2)
  print(paste("iter", iter, " - ",computeCost(w,dd, pconst)))
  
  iter <- iter + 1
}
