#include <Rcpp.h>
#include "screening.h"

//' Do predictions for ScreeningModel1
//' @param t double vector of times to evaluate
//' @param tj double vector of screening times
//' @param scale1 Weibull scale for onset
//' @param shape Weibull shape for onset
//' @param shape2 Weibull shape for clinical diagnosis
//' @param scale2 Weibull scale for clinical diagnosis
//' @param beta false negative fraction for screening
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @param simple whether to calculate Z using a simple method (complement) or integration (defaults to true)
//' @return data-frame with elements t, X, Y, Z (for state probabilities) and I (for incidence)
//' @importFrom Rcpp sourceCpp
//' @examples
//' par(mfrow=1:2)
//' x=seq(0,30,length=301)
//' screening_model_1_predictions(t=x, tj=vector("double"), shape1=1.5, shape2=1.5, scale1=20, scale2=10, beta=0.4) |>
//'     with(matplot(time,cbind(X,Y,Z),type="l", lty=1, main="No screening"))
//' screening_model_1_predictions(t=x, tj=c(10,15, 20, 25), shape1=1.5, shape2=1.5, scale1=20, scale2=10, beta=0.4) |>
//'     with(matplot(time,cbind(X,Y,Z),type="l", lty=1, main="Screening"))
//' @export
// [[Rcpp::export]]
Rcpp::DataFrame
screening_model_1_predictions(std::vector<double> t, std::vector<double> tj,
			      double shape1=1, double scale1=1,
			      double shape2=1, double scale2=1,
			      double beta=0.05,
			      double tol = 1e-6) {
  // Initialize the ScreeningModel with appropriate parameters
  screening::ScreeningModel1 m(tj, beta,
			       [&](double u) { return R::dweibull(u,shape1,scale1,0); },
			       [&](double u) { return R::pweibull(u,shape1,scale1,0,0); },
			       [&](double u) { return R::dweibull(u,shape2,scale2,0); },
			       [&](double u) { return R::pweibull(u,shape2,scale2,0,0); },
			       tol);
  return m.predictions(t,tj);
}

//' Do likelihood calculations for ScreeningModel1
//' @param inputs list of list with elements of t for the evaluation time, tj for the screening times and type for the type of likelihood (1=No cancer detected, 2=Screen-detected cancer, 3=Interval cancer)
//' @param tj double vector of screening times
//' @param scale1 Weibull scale for onset
//' @param shape Weibull shape for onset
//' @param shape2 Weibull shape for clinical diagnosis
//' @param scale2 Weibull scale for clinical diagnosis
//' @param beta false negative fraction for screening
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @param simple true whether to use the simple calculation for Z
//' @return list with element t for the times t and element p for a numeric matrix of the probabilities, including column names X, Y and Z.
//' @export
// [[Rcpp::export]]
std::vector<double>
screening_model_1_likes(Rcpp::List inputs, double shape1 = 1.0, double scale1 = 1.0, 
			double shape2 = 1.0, double scale2 = 1.0, double beta = 0.05,
			double tol=1e-6) {
  screening::ScreeningModel1 m({}, beta,
			       [&](double u) { return R::dweibull(u,shape1,scale1,0); },
			       [&](double u) { return R::pweibull(u,shape1,scale1,0,0); },
			       [&](double u) { return R::dweibull(u,shape2,scale2,0); },
			       [&](double u) { return R::pweibull(u,shape2,scale2,0,0); },
			       tol);
  return m.likes(inputs);
}
