
#'
#' @export
AR1Cov<-function(n,rou,sigma){
  Sigma = diag(n)*sigma^2
  for (i in 1:(n-1)) {
    Sigma[i+1,i] = rou
    Sigma[i,i+1] = rou
  }
  SigmaInv = chol2inv(Sigma)
  return(list(cov=Sigma,covInv=SigmaInv))
}
#'
#' @export
rmvn <- function(n, mu=0, V = matrix(1)){
  p <- length(mu)
  if(any(is.na(match(dim(V),p))))
    stop("Dimension problem!")
  D <- chol(V)
  t(matrix(rnorm(n*p), ncol=p)%*%D + rep(mu,rep(n,p)))
}
#'
#' @export
maternCov<-function(v,phi,sigma,coords){
  if(!(v %in% c(1/2,3/2,5/2))){
    return("v= 1/2, 3/2, or 5/2 !")
  }
  D <- as.matrix(dist(coords))
  if(v==1/2){
    R <- exp(-D/phi)
  }else if(v==3/2){
    R <- (1+sqrt(3)*D/phi)*exp(-sqrt(3)*D/phi)
  }else{
    R <- (1+sqrt(5)*D/phi+5*D^2/(3*phi^2))*exp(-sqrt(5)*D/phi)
  }
  Sigma = (sigma^2)*R
  SigmaInv = chol2inv(Sigma)
  return(list(cov=Sigma,covInv=SigmaInv))
}
#'
#' @export
proposedMSE<-function(n,Sigma,tau){
  w <- rmvn(1, rep(0,n), Sigma)
  e <- rnorm(n, 0, tau)
  a = w+e
  return(sum(a^2)/n)
}
#'
#' @export
weightedSearchData<-function(data, w){
  coords<-data$coords
  X<-data$X
  y<-data$y
  
  # linear nodel
  lmd = lm(y~X)
  # BART
  bartmd = wbart(X,y)
  # Remove the mean trend
  zeromeany=NULL
  for (wi in w) {
    temp = (1-wi)*lmd$residuals + wi*(y - bartmd$yhat.train.mean)
    zeromeany = cbind(zeromeany,temp)
  }
  return(zeromeany)
}



