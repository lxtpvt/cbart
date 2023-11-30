library(BART)
library(cbart)
library(matlib)
#=================================================================
# simulated data
#=================================================================
sigma = .1
f = function(x) {x^3}
set.seed(17)
ntr = 200
xtr = sort(2*runif(ntr)-1)
# i.i.d. errors
ytr_iid = f(xtr) + sigma*rnorm(ntr)
# create auto-correlated data
e = rep(0,ntr)
e[1] = sigma*rnorm(1)
rou = 0.8
for (i in 2:200) {
  e[i] = rou*e[i-1] + sigma*rnorm(1)
}
# auto-correlated errors
ytr_arc = f(xtr) + e
# test data
xte = seq(-1,.99,by=.01)
nte = length(xte)

#=================================================================
# helper function
#=================================================================
createSigamInv<-function(n,rou,sigma){
  A = diag(n)
  for (i in 1:(n-1)) {
    A[i+1,i] = rou
  }
  InvA = inv(A)
  SigmaInv = sigma^(-2)*t(InvA)%*%InvA
  return(SigmaInv)
}

#=======================
# i.i.d. errors
#=======================

# BART
set.seed(14) #it is MCMC, set the seed!!
rb_iid = wbart(xtr,ytr_iid,xte)

# CBART
sigma = 0.1
SigmaInv = sigma^(-2)*diag(ntr)
set.seed(14) #it is MCMC, set the seed!!
cbart_iid = cbart(x.train = xtr, y.train = ytr_iid,SigmaInv=SigmaInv, x.test=xte)

# plot 
plot(xtr,ytr_iid,cex=.5,main = "Figure 2: (a)")
lines(xte,f(xte),col="black",lwd=1.5,lty=1)
lines(xte,rb_iid$yhat.test.mean,col="blue",lwd=1.5,lty=1)
lines(xte,cbart_iid$yhat.test.mean,col="red",lwd=1.5,lty=2)
legend("topleft", legend=c("BART", "CBART"), col=c("blue", "red"), lty=c(1,2), cex=0.8)

#=======================
# auto-correlated errors
#=======================
# BART
set.seed(14) #it is MCMC, set the seed!!
rb_arc = wbart(xtr,ytr_arc,xte)

# GPCBART
rou = 0.8
sigma = 0.1
createSigamInv(ntr,rou,sigma)->SigmaInv
set.seed(14) #it is MCMC, set the seed!!
cbart_arc = cbart(x.train = xtr, y.train = ytr_arc,SigmaInv=SigmaInv, x.test=xte)

# plot 
plot(xtr,ytr_arc,cex=.5,main = "Figure 2: (b)")
lines(xte,f(xte),col="black",lwd=1.5,lty=1)
lines(xte,rb_arc$yhat.test.mean,col="blue",lwd=1.5,lty=1)
lines(xte,cbart_arc$yhat.test.mean,col="red",lwd=1.5,lty=1)
legend("topleft", legend=c("BART", "CBART"), col=c("blue", "red"), lty=c(1,1), cex=0.8)

#========================
# Results analysis
#========================
# fitting-to-data
ytr_arc_mean = mean(ytr_arc)
e = ytr_arc - ytr_arc_mean
sst = t(e)%*%e
e_train_bart = rb_arc$yhat.train.mean - ytr_arc
e_train_cbart = cbart_arc$yhat.train.mean - ytr_arc
# MSE
t(e_train_bart)%*%e_train_bart/ntr -> mse_train_bart
t(e_train_cbart)%*%e_train_cbart/ntr -> mse_train_cbart

mse_train_bart
mse_train_cbart
(mse_train_cbart-mse_train_bart)/mse_train_bart
# R-square
sse_bart = mse_train_bart*ntr
sse_cbart = mse_train_cbart*ntr
R2_bart = 1-sse_bart/sst
R2_cbart = 1-sse_cbart/sst

R2_bart
R2_cbart
(R2_cbart-R2_bart)/R2_bart
# fitting-to-f
f_real = f(xte)
f_bart = apply(rb_arc$yhat.test,2,mean)
f_cbart = cbart_arc$yhat.test.mean

f_bart-f_real -> e_bart
f_cbart-f_real -> e_cbart

t(e_bart)%*%e_bart/ntr -> mse_f_bart
t(e_cbart)%*%e_cbart/ntr -> mse_f_cbart

mse_f_bart
mse_f_cbart

(mse_f_cbart-mse_f_bart)/mse_f_bart

