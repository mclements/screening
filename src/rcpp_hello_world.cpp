#include <Rcpp.h>
#include "screening.h"

//' Do predictions for ScreeningModel1
//' @name ScreeningModel1
//' @param t double vector of times to evaluate
//' @param ti double vector of screening times
//' @param scale1 Weibull scale for onset
//' @param shape Weibull shape for onset
//' @param shape2 Weibull shape for clinical diagnosis
//' @param scale2 Weibull scale for clinical diagnosis
//' @param beta false negative fraction for screening
//' @param simple whether to calculate Z using a simple method (complement) or integration (defaults to true)
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
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
screening_model_1_predictions(std::vector<double> t,
			      std::vector<double> ti,
			      double shape1=1,
			      double scale1=1,
			      double shape2=1,
			      double scale2=1,
			      double beta=0.05,
			      bool simple = true,
			      double tol = 1e-6) {
  // Initialize the ScreeningModel with appropriate parameters
  screening::ScreeningModel1 m([&](double u) { return R::dweibull(u,shape1,scale1,0); },
			       [&](double u) { return R::pweibull(u,shape1,scale1,0,0); },
			       [&](double u) { return R::dweibull(u,shape2,scale2,0); },
			       [&](double u) { return R::pweibull(u,shape2,scale2,0,0); },
			       beta,
			       tol);
  m.update(ti);
  return m.predictions(t,simple);
}

//' Do likelihood calculations for ScreeningModel1
//' @name ScreeningModel1
//' @param inputs list of list with elements of t for the evaluation time, tj for the screening times and type for the type of likelihood (1=No cancer detected, 2=Screen-detected cancer, 3=Interval cancer)
//' @param scale1 Weibull scale for onset
//' @param shape Weibull shape for onset
//' @param shape2 Weibull shape for clinical diagnosis
//' @param scale2 Weibull scale for clinical diagnosis
//' @param beta false negative fraction for screening
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @return vector of likelihoods
//' @export
// [[Rcpp::export]]
std::vector<double>
screening_model_1_likes(Rcpp::List inputs, double shape1 = 1.0, double scale1 = 1.0, 
			double shape2 = 1.0, double scale2 = 1.0, double beta = 0.05,
			double tol=1e-6) {
  screening::ScreeningModel1 m([&](double u) { return R::dweibull(u,shape1,scale1,0); },
			       [&](double u) { return R::pweibull(u,shape1,scale1,0,0); },
			       [&](double u) { return R::dweibull(u,shape2,scale2,0); },
			       [&](double u) { return R::pweibull(u,shape2,scale2,0,0); },
			       beta,
			       tol);
  return m.likes(inputs);
}

//' Do predictions for ScreeningModel2
//' @name ScreeningModel2
//' @param t double vector of times to evaluate
//' @param ti double vector of screening times
//' @param yi double vector of screening times
//' @param scale1 Weibull scale for onset
//' @param shape Weibull shape for onset
//' @param shape2 Weibull shape for clinical diagnosis
//' @param scale2 Weibull scale for clinical diagnosis
//' @param beta0 intercept for logistic model for false negative fraction
//' @param beta1 slope of log(yi) for logistic model for false negative fraction
//' @param simple whether to calculate Z using a simple method (complement) or integration (defaults to true)
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @return data-frame with elements t, X, Y, Z (for state probabilities) and I (for incidence)
//' @importFrom Rcpp sourceCpp
//' @examples
//' par(mfrow=1:2)
//' x=seq(0,30,length=301)
//' screening_model_2_predictions(t=x, ti=vector("double"), yi=vector("double"), shape1=1.5, shape2=1.5, scale1=20, scale2=10) |>
//'     with(matplot(time,cbind(X,Y,Z),type="l", lty=1, main="No screening"))
//' screening_model_2_predictions(t=x, ti=c(10,15, 20, 25), yi=c(3,3,3), shape1=1.5, shape2=1.5, scale1=20, scale2=10) |>
//'     with(matplot(time,cbind(X,Y,Z),type="l", lty=1, main="Screening"))
//' @export
// [[Rcpp::export]]
Rcpp::DataFrame
screening_model_2_predictions(std::vector<double> t,
			      std::vector<double> ti,
			      std::vector<double> yi,
			      double shape1=1, double scale1=1,
			      double shape2=1, double scale2=1,
			      double beta0=-3.0,
			      double beta1=1.0,
			      bool simple = true,
			      double tol = 1e-6) {
  // Initialize the model with appropriate parameters
  screening::ScreeningModel2 m([&](double u) { return R::dweibull(u,shape1,scale1,0); },
			       [&](double u) { return R::pweibull(u,shape1,scale1,0,0); },
			       [&](double u) { return R::dweibull(u,shape2,scale2,0); },
			       [&](double u) { return R::pweibull(u,shape2,scale2,0,0); },
			       [&](double y) { return 1.0/(1.0+std::exp(-(beta0+beta1*std::log(y)))); },
			       tol);
  m.update(ti,yi);
  return m.predictions(t,simple);
}

//' Do likelihood calculations for ScreeningModel2
//' @name ScreeningModel2
//' @param inputs list of list with elements of t for the evaluation time, tj for the screening times and type for the type of likelihood (1=No cancer detected, 2=Screen-detected cancer, 3=Interval cancer)
//' @param scale1 Weibull scale for onset
//' @param shape Weibull shape for onset
//' @param shape2 Weibull shape for clinical diagnosis
//' @param scale2 Weibull scale for clinical diagnosis
//' @param beta0 intercept for logistic model for false negative fraction
//' @param beta1 slope of log(yi) for logistic model for false negative fraction
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @return vector of likelihoods
//' @export
// [[Rcpp::export]]
std::vector<double>
screening_model_2_likes(Rcpp::List inputs,
			double shape1 = 1.0,
			double scale1 = 1.0, 
			double shape2 = 1.0,
			double scale2 = 1.0, 
			double beta0=-3.0,
			double beta1=1.0,
			double tol=1e-6) {
  screening::ScreeningModel2 m([&](double u) { return R::dweibull(u,shape1,scale1,0); },
			       [&](double u) { return R::pweibull(u,shape1,scale1,0,0); },
			       [&](double u) { return R::dweibull(u,shape2,scale2,0); },
			       [&](double u) { return R::pweibull(u,shape2,scale2,0,0); },
			       [&](double y) { return 1.0/(1.0+std::exp(-(beta0+beta1*std::log(y)))); },
			       tol);
  return m.likes(inputs);
}




//' Do predictions for ScreeningModel3
//' @name ScreeningModel3
//' @param t double vector of times to evaluate
//' @param ti double vector of screening times
//' @param yi double vector of screening times
//' @param scale1 Weibull scale for onset
//' @param shape Weibull shape for onset
//' @param shape2 Weibull shape for clinical diagnosis
//' @param scale2 Weibull scale for clinical diagnosis
//' @param beta0 intercept for logistic model for false negative fraction
//' @param beta1 slope of log(yi) for logistic model for false negative fraction
//' @param PrFalseNegBx probability of a false negative biopsy | cancer, biopsy undertaken
//' @param simple whether to calculate Z using a simple method (complement) or integration (defaults to true)
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @return data-frame with elements t, X, Y, Z (for state probabilities) and I (for incidence)
//' @importFrom Rcpp sourceCpp
//' @examples
//' par(mfrow=1:2)
//' x=seq(0,30,length=301)
//' screening_model_3_predictions(t=x, ti=vector("double"), yi=vector("double"), bxi=vector("integer"), shape1=1.5, shape2=1.5, scale1=20, scale2=10) |>
//'     with(matplot(time,cbind(X,Y,Z),type="l", lty=1, main="No screening"))
//' screening_model_3_predictions(t=x, ti=c(10,15, 20, 25), yi=c(3,3,3), bxi=c(FALSE,FALSE,TRUE), shape1=1.5, shape2=1.5, scale1=20, scale2=10) |>
//'     with(matplot(time,cbind(X,Y,Z),type="l", lty=1, main="Screening"))
//' @export
// [[Rcpp::export]]
Rcpp::DataFrame
screening_model_3_predictions(std::vector<double> t,
			      std::vector<double> ti,
			      std::vector<double> yi,
			      std::vector<int> bxi,
			      double shape1=1, double scale1=1,
			      double shape2=1, double scale2=1,
			      double beta0=-3.0,
			      double beta1=1.0,
			      double PrFalseNegBx=0.05,
			      bool simple = true,
			      double tol = 1e-6) {
  // Initialize the model with appropriate parameters
  screening::ScreeningModel3 m([&](double u) { return R::dweibull(u,shape1,scale1,0); },
			       [&](double u) { return R::pweibull(u,shape1,scale1,0,0); },
			       [&](double u) { return R::dweibull(u,shape2,scale2,0); },
			       [&](double u) { return R::pweibull(u,shape2,scale2,0,0); },
			       [&](double y) { return 1.0/(1.0+std::exp(-(beta0+beta1*std::log(y)))); },
			       PrFalseNegBx,
			       tol);
  m.update(ti,yi,bxi);
  return m.predictions(t,simple);
}

//' Do likelihood calculations for ScreeningModel3
//' @name ScreeningModel3
//' @param inputs list of list with elements of t for the evaluation time, tj for the screening times and type for the type of likelihood (1=No cancer detected, 2=Screen-detected cancer, 3=Interval cancer)
//' @param scale1 Weibull scale for onset
//' @param shape Weibull shape for onset
//' @param shape2 Weibull shape for clinical diagnosis
//' @param scale2 Weibull scale for clinical diagnosis
//' @param beta0 intercept for logistic model for false negative fraction
//' @param beta1 slope of log(yi) for logistic model for false negative fraction
//' @param PrFalseNegBx probability of a false negative biopsy | cancer, biopsy undertaken
//' @param tol double for the numeric tolerance of the integration (default=1e-6)
//' @return vector of likelihoods
//' @export
// [[Rcpp::export]]
std::vector<double>
screening_model_3_likes(Rcpp::List inputs,
			double shape1 = 1.0,
			double scale1 = 1.0, 
			double shape2 = 1.0,
			double scale2 = 1.0, 
			double beta0=-3.0,
			double beta1=1.0,
			double PrFalseNegBx=0.05,
			double tol=1e-6) {
  screening::ScreeningModel3 m([&](double u) { return R::dweibull(u,shape1,scale1,0); },
			       [&](double u) { return R::pweibull(u,shape1,scale1,0,0); },
			       [&](double u) { return R::dweibull(u,shape2,scale2,0); },
			       [&](double u) { return R::pweibull(u,shape2,scale2,0,0); },
			       [&](double y) { return 1.0/(1.0+std::exp(-(beta0+beta1*std::log(y)))); },
			       PrFalseNegBx,
			       tol);
  return m.likes(inputs);
}
