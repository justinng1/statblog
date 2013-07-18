###################################
# Author:  Justin Ng
# Email:   justinng1@gmail.com
# Website: justinng1.wordpress.com
###################################

# test data.
rm(list=ls())

# create test data.
m <- 50
x1 <- runif(m, min=1, max=10)
x2 <- rnorm(m, mean=-5, sd=10)
x3 <- rnorm(m, mean=1000, sd=50)
y <- 6 + 2*x1 + 3*x2 + 4*x3 + rnorm(m, mean=0, sd=10)


# feature scaling.  won't converge without this.
x1_s <- (x1-mean(x1))/(max(x1)-min(x1))
x2_s <- (x2-mean(x2))/(max(x2)-min(x2))
x3_s <- (x3-mean(x3))/(max(x3)-min(x3))

# find the weights using the GLM function.
dd <- data.frame(x1=x1, x2=x2, x3=x3, y=y, x1_s=x1_s, x2_s=x2_s, x3_s=x3_s)
g <- glm(y ~ x1_s + x2_s + x3_s, data=dd)
summary(g)


# batch gradient descent.
w <- c(1,1,1,1)
l2 <- 1
alpha <- 1
iter <- 1
ep <- 0.0001
while (l2 > 0.01)  {
  
  dcostdw0 <- with(dd,1/m*sum((w[1] + w[2]*x1_s + w[3]*x2_s + w[4]*x3_s - y)))
  dcostdw1 <- with(dd,1/m*sum((w[1] + w[2]*x1_s + w[3]*x2_s + w[4]*x3_s - y)*x1_s))
  dcostdw2 <- with(dd,1/m*sum((w[1] + w[2]*x1_s + w[3]*x2_s + w[4]*x3_s - y)*x2_s))
  dcostdw3 <- with(dd,1/m*sum((w[1] + w[2]*x1_s + w[3]*x2_s + w[4]*x3_s - y)*x3_s))
  
  # numerical gradient checking.  To make sure that it works.
#     dcostdw0_a <- with(dd,1/(2*m)/(2*ep)*(sum((w[1]+ep + w[2]*x1_s + w[3]*x2_s + w[4]*x3_s - y)^2) - sum((w[1]-ep + w[2]*x1_s + w[3]*x2_s + w[4]*x3_s - y)^2)))
#     dcostdw1_a <- with(dd,1/(2*m)/(2*ep)*(sum((w[1] + (w[2]+ep)*x1_s + w[3]*x2_s + w[4]*x3_s - y)^2) - sum((w[1] + (w[2]-ep)*x1_s + w[3]*x2_s + w[4]*x3_s - y)^2)))
#     dcostdw2_a <- with(dd,1/(2*m)/(2*ep)*(sum((w[1] + w[2]*x1_s + (w[3]+ep)*x2_s + w[4]*x3_s - y)^2) - sum((w[1] + w[2]*x1_s + (w[3]-ep)*x2_s + w[4]*x3_s - y)^2)))
#     dcostdw3_a <- with(dd,1/(2*m)/(2*ep)*(sum((w[1] + w[2]*x1_s + w[3]*x2_s + (w[4]+ep)*x3_s - y)^2) - sum((w[1] + w[2]*x1_s + w[3]*x2_s + (w[4]-ep)*x3_s - y)^2)))
#     
  #   print(paste(dcostdw0,dcostdw1,dcostdw2,dcostdw3,"::",dcostdw0_a,dcostdw1_a,dcostdw2_a,dcostdw3_a))
  
  w[1] <- w[1] - alpha*dcostdw0
  w[2] <- w[2] - alpha*dcostdw1
  w[3] <- w[3] - alpha*dcostdw2
  w[4] <- w[4] - alpha*dcostdw3
  
  cost <- with(dd, 1/(2*m)*sum((w[1] + w[2]*x1_s + w[3]*x2_s + w[4]*x3_s - y)^2))
  l2 <- sqrt(dcostdw0^2 + dcostdw1^2 + dcostdw2^2 + dcostdw3^2)
  print(paste("iter", iter, " - ",cost))
  
  iter <- iter + 1
}

w  # pretty close to GLM.