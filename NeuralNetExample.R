
#Picked up from :  http://jee3.web.rice.edu/neuralnets-1.r

rm(list=ls())
set.seed(1231239)

# create an unusual regression problem

# note that these variables will already have the proper scaling
x<-runif(1000)
z<-runif(1000)

# the relationship between y and the independent variables
# is complex and non-linear
y<-sin(3*pi*x)+cos(3*pi*z)+rnorm(1000, mean=0, sd=0.25)
plot(y~x)
plot(y~z)
dat<-data.frame(x,z,y)

# estimate a neural network with one hidden layer of 4 nodes
library(neuralnet)
nn<-neuralnet(y~x+z, data=dat, hidden=4, stepmax=2e05, threshold=0.02, lifesign="full")
plot(nn)

# do the predictions of the neural network look good?
x.test<-seq(from=0, to=1, by=0.01)
y.fit<-compute(nn, covariate=matrix(c(x.test, rep(0.5, length(x.test))), nrow=length(x.test), ncol=2))$net.result
plot(y~x, data=dat)
lines(y.fit~x.test, type="l", col="red", lwd=2)

z.test<-seq(from=0, to=1, by=0.01)
y.fit<-compute(nn, covariate=matrix(c(rep(1, length(z.test)), z.test), nrow=length(z.test), ncol=2))$net.result
plot(y~z, data=dat)
lines(y.fit~z.test, type="l", col="red", lwd=2)
#...no. We need to allow a more complex fitting surface.

# Try eight nodes in the hidden layer.
nn<-neuralnet(y~x+z, data=dat, hidden=8, stepmax=2e05, threshold=0.02, lifesign="full")
plot(nn)

x.test<-seq(from=0, to=1, by=0.01)
y.fit<-compute(nn, covariate=matrix(c(x.test, rep(0.5, length(x.test))), nrow=length(x.test), ncol=2))$net.result
plot(y~x, data=dat)
lines(y.fit~x.test, type="l", col="red", lwd=2)

z.test<-seq(from=0, to=1, by=0.01)
y.fit<-compute(nn, covariate=matrix(c(rep(1, length(z.test)), z.test), nrow=length(z.test), ncol=2))$net.result
plot(y~z, data=dat)
lines(y.fit~z.test, type="l", col="red", lwd=2)
# much better!








rm(list=ls())
set.seed(29124)

# create a complex classification problem

x<-runif(1000)
z<-runif(1000)
o<-order(z)

# the points will lie (noisily) in one class if inside
# a circle centered at zero with radius (0.5)^(1/2), and 
# in another class if outside this circle
y<-ifelse(x^2+z^2+runif(1000, min=-0.15, max=0.15)>0.5, 1, 0)
plot(x~z, pch=y)
dat<-data.frame(x,z,y)

# can a glm (probit) model correctly classify these observations?
glm.mod<-glm(y~x+z, family=binomial)
glm.class<-ifelse(predict(glm.mod, type="response")>0.5, "red", "blue")
plot(x~z, pch=y, col=glm.class)
z.plot<-seq(from=0, to=sqrt(0.5), by=0.0001)
lines(sqrt(0.5-z.plot^2)~z.plot)
#...no, or at least not without some substantial modification
# (like polynomial terms). It draws a line through the circular boundary space.

# a neural network with a single layer of four hidden nodes
# performs this easily.
library(neuralnet)
nn<-neuralnet(y~x+z, data=dat, linear.output=FALSE, hidden=4, lifesign="full")
plot(nn)

# inside the sample data set, the neural network model classifies
# quite well.
nn.dat<-prediction(nn)$data
nn.class<-ifelse(nn.dat[,"y"]>0.5, "red", "blue")
plot(x[o]~z[o], pch=y[o], col=nn.class)
z.plot<-seq(from=0, to=sqrt(0.5), by=0.0001)
lines(sqrt(0.5-z.plot^2)~z.plot)
# ...maybe a little too well. Might be some evidence of overfitting.
# How does it do out of sample?

xx<-runif(1000)
zz<-runif(1000)
yy<-ifelse(xx^2+zz^2+runif(1000, min=-0.15, max=0.15)>0.5, 1, 0)

nn.class.p<-ifelse(compute(nn, covariate=data.frame(x=xx, z=zz))$net.result>0.5, "red", "blue")
plot(xx~zz, pch=yy, col=nn.class.p)
z.plot<-seq(from=0, to=sqrt(0.5), by=0.0001)
lines(sqrt(0.5-z.plot^2)~z.plot)
# ah, much better. We have almost perfect separation of the classes at the limit
# of what is possible given noise levels.



# We can compare the out-of-sample fit characteristics of the glm
# model and the neural network model using the ROC curve.
glm.class.y<-predict(glm.mod, type="response", newdata=data.frame(x=xx, z=zz))
library(ROCR)
pred <- prediction(predictions=glm.class.y, labels=yy)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
plot(perf, col=rainbow(10))

nn.class.py<-compute(nn, covariate=data.frame(x=xx, z=zz))$net.result
pred <- prediction(predictions=nn.class.py, labels=yy)
perf <- performance(pred, measure = "tpr", x.measure = "fpr") 
plot(perf, col="blue", add=T)
#...not surprisingly, the neural network model does better.






rm(list=ls())
set.seed(02134)

# create a non-linear feature extraction problem
x<-runif(1000)
z<-sin(4*x)+runif(1000, min=-0.25, max=0.25)
plot(x~z)

# x and z clearly have a rather complex common dimensionality.
# We could map the values of x and z that we observe to the 
# value of a continuous variable. And what would that continuous
# variable be?

library(pcaMethods)
pca.nn<-nlpca(cbind(scale(x),scale(z)), nPcs=1, maxSteps=1000)

# What is the hidden dimension?
plot(x~pca.nn@scores)
plot(z~pca.nn@scores)

# so the hidden dimension is associated with small values of z for 
# large and small vales, and large values of z for middling values.

