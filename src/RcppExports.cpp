// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// cdbartSigmaFix_inf
List cdbartSigmaFix_inf(size_t n, size_t p, size_t np, NumericVector ixv, NumericVector iyv, NumericVector ixpv, Eigen::MatrixXd& SigmaInv, size_t m, IntegerVector numcutv, size_t nd, size_t burn, double mybeta, double alpha, double tau, bool dart, double theta, double omega, double a, double b, double rho, bool aug, size_t nkeeptrain, size_t nkeeptest, size_t nkeeptestme, size_t nkeeptreedraws, size_t printevery);
RcppExport SEXP _cbart_cdbartSigmaFix_inf(SEXP nSEXP, SEXP pSEXP, SEXP npSEXP, SEXP ixvSEXP, SEXP iyvSEXP, SEXP ixpvSEXP, SEXP SigmaInvSEXP, SEXP mSEXP, SEXP numcutvSEXP, SEXP ndSEXP, SEXP burnSEXP, SEXP mybetaSEXP, SEXP alphaSEXP, SEXP tauSEXP, SEXP dartSEXP, SEXP thetaSEXP, SEXP omegaSEXP, SEXP aSEXP, SEXP bSEXP, SEXP rhoSEXP, SEXP augSEXP, SEXP nkeeptrainSEXP, SEXP nkeeptestSEXP, SEXP nkeeptestmeSEXP, SEXP nkeeptreedrawsSEXP, SEXP printeverySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< size_t >::type n(nSEXP);
    Rcpp::traits::input_parameter< size_t >::type p(pSEXP);
    Rcpp::traits::input_parameter< size_t >::type np(npSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type ixv(ixvSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type iyv(iyvSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type ixpv(ixpvSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd& >::type SigmaInv(SigmaInvSEXP);
    Rcpp::traits::input_parameter< size_t >::type m(mSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type numcutv(numcutvSEXP);
    Rcpp::traits::input_parameter< size_t >::type nd(ndSEXP);
    Rcpp::traits::input_parameter< size_t >::type burn(burnSEXP);
    Rcpp::traits::input_parameter< double >::type mybeta(mybetaSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< bool >::type dart(dartSEXP);
    Rcpp::traits::input_parameter< double >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type omega(omegaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< bool >::type aug(augSEXP);
    Rcpp::traits::input_parameter< size_t >::type nkeeptrain(nkeeptrainSEXP);
    Rcpp::traits::input_parameter< size_t >::type nkeeptest(nkeeptestSEXP);
    Rcpp::traits::input_parameter< size_t >::type nkeeptestme(nkeeptestmeSEXP);
    Rcpp::traits::input_parameter< size_t >::type nkeeptreedraws(nkeeptreedrawsSEXP);
    Rcpp::traits::input_parameter< size_t >::type printevery(printeverySEXP);
    rcpp_result_gen = Rcpp::wrap(cdbartSigmaFix_inf(n, p, np, ixv, iyv, ixpv, SigmaInv, m, numcutv, nd, burn, mybeta, alpha, tau, dart, theta, omega, a, b, rho, aug, nkeeptrain, nkeeptest, nkeeptestme, nkeeptreedraws, printevery));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_cbart_cdbartSigmaFix_inf", (DL_FUNC) &_cbart_cdbartSigmaFix_inf, 26},
    {NULL, NULL, 0}
};

RcppExport void R_init_cbart(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
