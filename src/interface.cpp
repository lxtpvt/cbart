
#include "cdbartSigmaFix.h"
using namespace Rcpp;

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp 
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//


// [[Rcpp::export]]
List cdbartSigmaFix_inf(
    size_t n,            //number of observations in training data
    size_t p,		//dimension of x
    size_t np,		//number of observations in test data
    NumericVector ixv,		//x, train,  pxn (transposed so rows are contiguous in memory)
    NumericVector iyv,		//y, train,  nx1
    NumericVector ixpv,		//x, test, pxnp (transposed so rows are contiguous in memory)
    Eigen::MatrixXd& SigmaInv,
    size_t m,		//number of trees
    IntegerVector numcutv,		//number of cut points
    size_t nd,		//number of kept draws (except for thinnning ..)
    size_t burn,		//number of burn-in draws skipped
    double mybeta,
    double alpha,
    double tau,
    bool dart,
    double theta,
    double omega,
    double a,
    double b,
    double rho,
    bool aug,
    size_t nkeeptrain,
    size_t nkeeptest,
    size_t nkeeptestme,
    size_t nkeeptreedraws,
    size_t printevery
)
{
  
  double* ix = &ixv[0];
  double* iy = &iyv[0];
  double* ixp = &ixpv[0];
  int* numcut = &numcutv[0];
  double* trmean=new double[n];
  double* temean=new double[np];
  
  // double* _trdraw=new double[nkeeptrain*n];
  // double* _tedraw=new double[nkeeptest*np];
  
  cdbartSigmaFix(n, p, np, ix, iy, ixp, m, numcut, nd, burn, mybeta, alpha, tau, dart, theta, omega, a, b, 
    rho, aug, nkeeptrain, nkeeptest, nkeeptestme, nkeeptreedraws, printevery, trmean, temean, SigmaInv);
  
  NumericVector trmeanv = NumericVector(trmean,trmean+n);
  NumericVector temeanv = NumericVector(temean,temean+np);
  // NumericVector trdrawv = NumericVector(_trdraw,_trdraw+nkeeptrain*n);
  // NumericVector tedrawv = NumericVector(_tedraw,temean+nkeeptest*np);
  //--------------------------------------------------
  //use wrap to return computed totdim to R as part of a list
  List ret; //list to return
  ret["yhat.train.mean"] = trmeanv;
  ret["yhat.test.mean"] = temeanv;
  // ret["yhat.train"]=trdrawv;
  // ret["yhat.test"]=tedrawv;
  return ret;
  
}