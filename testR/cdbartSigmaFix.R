#=================================================================
# simulated data
#=================================================================
sigma = .1
f = function(x) {x^2}
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
# BART vs GPCBART
#=================================================================
library(BART)
library(GPCBART)
# Initialization
nd=1000
burn=500
nkeeptrain=nd
nkeeptest=nd
nkeeptestmean=nd
nkeeptreedraws=nd
printevery=100

m = 200 #number of trees
nc={100} #number of cut points

k=2
power=2
mybeta=power
base=0.95
alpha=base

#================================================================
# i.i.d. errors
#=======================
# plot 
plot(xtr,ytr_iid,cex=.5,main = "Noises are i.i.d.")
lines(xte,f(xte),col="black",lwd=1.5,lty=1)

# BART 
rb = wbart(xtr,ytr_iid,xte,ntree=50,nskip=burn,ndpost=nd)
lines(xte,apply(rb$yhat.test,2,mean),col="blue",lwd=1.5,lty=1)

# GPCBART
tau_iid = (max(ytr_iid) - min(ytr_iid))/(2 * k * sqrt(m))
SigmaInv = sigma^(-2)*diag(ntr)
res_iid = cdbartSigmaFix_inf(ntr,1,nte,xtr,ytr_iid,xte,m,nc,nd,burn,mybeta,alpha,tau_iid,0, 0, 0, 0, 0, 0, 0,
                         nkeeptrain, nkeeptest, nkeeptestmean, nkeeptreedraws, printevery, SigmaInv)

lines(xte,res_iid$temean,col="red",lwd=1.5,lty=2)
legend("bottomright", legend=c("BART", "GPCBART"), col=c("blue", "red"), lty=c(1,2), cex=0.8)


#================================================================
# auto-correlated errors
#=======================
# plot 
plot(xtr,ytr_arc,cex=.5,main = "Noises are auto-correlated")
lines(xte,f(xte),col="black",lwd=1.5,lty=1)

# BART 
rb = wbart(xtr,ytr_arc,xte,ntree=50,nskip=burn,ndpost=nd)
lines(xte,apply(rb$yhat.test,2,mean),col="blue",lwd=1.5,lty=1)

# GPCBART
tau_arc = (max(ytr_arc) - min(ytr_arc))/(2 * k * sqrt(m));
A = diag(ntr)
for (i in 1:(ntr-1)) {
  A[i+1,i] = rou
}
library(matlib)
InvA = inv(A)
SigmaInv = sigma^(-2)*t(InvA)%*%InvA
res_arc = cdbartSigmaFix_inf(ntr,1,nte,xtr,ytr_arc,xte,m,nc,nd,burn,mybeta,alpha,tau_arc,0, 0, 0, 0, 0, 0, 0,
                         nkeeptrain, nkeeptest, nkeeptestmean, nkeeptreedraws, printevery, SigmaInv)

lines(xte,res_arc$temean,col="red",lwd=1.5,lty=1)
legend("bottomright", legend=c("BART", "GPCBART"), col=c("blue", "red"), lty=c(1,1), cex=0.8)
