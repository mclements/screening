// [[Rcpp::depends(BH)]]
#include <Rcpp.h>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <vector>
#include <cmath>
#include <functional>

// // Templated class for a simple screening model with onset and clinical diagnosis
// // T1 is the type for f1
// // T2 is the type for S1
// // T3 is the type for f2
// // T4 is the type for S2
// template<class T1, class T2, class T3, class T4>
// class ScreeningModel1 {
// public:
//   size_t fulln, n;
//   std::vector<double> ti, tj;
//   double beta, error;
//   T1 f1;
//   T2 S1;
//   T3 f2;
//   T4 S2;
//   ScreeningModel1(std::vector<double> ti, double beta, T1 f1, T2 S1, T3 f2, T4 S2) :
//     fulln(ti.size()), ti(ti), beta(beta), f1(f1), S1(S1), f2(f2), S2(S2)  { }
//   // *NB*: use tj and n in the calculations!
//   void setup(double t) {
//     tj.resize(0);
//     tj.push_back(0.0);
//     for (size_t i=0; i<fulln && ti[i]<t; i++) {
//       tj.push_back(ti[i]);
//     }
//     tj.push_back(t);
//     n = tj.size()-2;
//   }
//   void update(std::vector<double> ti) {
//     this->ti=ti;
//     fulln=ti.size();
//   }
//   double X(double s, bool reset = true) {
//     if (reset) setup(s);
//     return S1(s);
//   }
//   double Y(double t, bool reset = true) {
//     using namespace boost::math::quadrature;
//     if (reset) setup(t);
//     double value = 0.0;
//     for (size_t i=0; i<=n; i++) {
//       auto fn = [&](double x) { return f1(x)*S2(t-x) * pow(beta,n-i); };
//       value += gauss_kronrod<double, 15>::integrate(fn, tj[i], tj[i+1], 5, 1e-6, &error);
//     }
//     return value;
//   }
//   double I(double t, bool reset=true) {
//     using namespace boost::math::quadrature;
//     if (reset) setup(t);
//     double value = 0.0;
//     for (size_t i=0; i<=n; i++) {
//       auto fn = [&](double x) { return f1(x)*f2(t-x)*pow(beta,n-i); };
//       value += gauss_kronrod<double, 15>::integrate(fn, tj[i], tj[i+1], 5, 1e-6, &error);
//     }
//     return value;
//   }
//   double Z(double t, bool reset=true, bool simple=true) {
//     using namespace boost::math::quadrature;
//     if (simple) return (1.0 - (X(t)+Y(t)));
//     if (reset) setup(t);
//     double value = 0.0, error2;
//     // cumulative incidence
//     for (size_t i=0; i<=n; i++) {
//       for (size_t j=i; j<=n; j++) {
// 	auto fn = [&](double s) { 
// 	  auto inner = [&](double u) { return f1(s)*pow(beta,j-i)*f2(u-s); };
// 	  return gauss_kronrod<double, 15>::integrate(inner, std::max(s,tj[j]),
// 						      tj[j+1], 5, 1e-6, &error2);
// 	};
// 	value += gauss_kronrod<double, 15>::integrate(fn, tj[i], tj[i+1], 5, 1e-6, &error);
//       }
//     }
//     // screen-detected
//     for (size_t i=0; i<n; i++) {
//       for (size_t j=i; j<n; j++) {
// 	auto fn = [&](double u) { return f1(u)*S2(tj[j+1]-u)*pow(beta,j-i)*(1-beta); };
// 	value += gauss_kronrod<double, 15>::integrate(fn, tj[i], tj[i+1], 5, 1e-6, &error);
//       }
//     }
//     return value;
//   }
// };

// class for a simple screening model with onset and clinical diagnosis
class ScreeningModel1 {
public:
  size_t fulln, n;
  std::vector<double> ti, tj;
  double beta, error;
  using fun_type = std::function<double(double)>;
  fun_type f1, S1, f2, S2;
  ScreeningModel1(std::vector<double> ti, double beta, fun_type f1, fun_type S1,
		  fun_type f2, fun_type S2) :
    fulln(ti.size()), ti(ti), beta(beta), f1(f1), S1(S1), f2(f2), S2(S2)  { }
  // *NB*: use tj and n in the calculations!
  void setup(double t) {
    tj.resize(0);
    tj.push_back(0.0);
    for (size_t i=0; i<fulln && ti[i]<t; i++) {
      tj.push_back(ti[i]);
    }
    tj.push_back(t);
    n = tj.size()-2;
  }
  void update(std::vector<double> ti) {
    this->ti=ti;
    fulln=ti.size();
  }
  double X(double s, bool reset = true) {
    if (reset) setup(s);
    return S1(s);
  }
  double Y(double t, bool reset = true) {
    using namespace boost::math::quadrature;
    if (reset) setup(t);
    double value = 0.0;
    for (size_t i=0; i<=n; i++) {
      auto fn = [&](double x) { return f1(x)*S2(t-x) * pow(beta,n-i); };
      value += gauss_kronrod<double, 15>::integrate(fn, tj[i], tj[i+1], 5, 1e-6, &error);
    }
    return value;
  }
  double I(double t, bool reset=true) {
    using namespace boost::math::quadrature;
    if (reset) setup(t);
    double value = 0.0;
    for (size_t i=0; i<=n; i++) {
      auto fn = [&](double x) { return f1(x)*f2(t-x)*pow(beta,n-i); };
      value += gauss_kronrod<double, 15>::integrate(fn, tj[i], tj[i+1], 5, 1e-6, &error);
    }
    return value;
  }
  double Z(double t, bool reset=true, bool simple=true) {
    using namespace boost::math::quadrature;
    if (simple) return (1.0 - (X(t)+Y(t)));
    if (reset) setup(t);
    double value = 0.0, error2;
    // cumulative incidence
    for (size_t i=0; i<=n; i++) {
      for (size_t j=i; j<=n; j++) {
	auto fn = [&](double s) { 
	  auto inner = [&](double u) { return f1(s)*pow(beta,j-i)*f2(u-s); };
	  return gauss_kronrod<double, 15>::integrate(inner, std::max(s,tj[j]),
						      tj[j+1], 5, 1e-6, &error2);
	};
	value += gauss_kronrod<double, 15>::integrate(fn, tj[i], tj[i+1], 5, 1e-6, &error);
      }
    }
    // screen-detected
    for (size_t i=0; i<n; i++) {
      for (size_t j=i; j<n; j++) {
	auto fn = [&](double u) { return f1(u)*S2(tj[j+1]-u)*pow(beta,j-i)*(1-beta); };
	value += gauss_kronrod<double, 15>::integrate(fn, tj[i], tj[i+1], 5, 1e-6, &error);
      }
    }
    return value;
  }
};


//' Do predictions for ScreeningModel1
//' @param t double vector of times to evaluate
//' @param tj double vector of screening times
//' @param scale1 Weibull scale for onset
//' @param shape Weibull shape for onset
//' @param shape2 Weibull shape for clinical diagnosis
//' @param scale2 Weibull scale for clinical diagnosis
//' @param beta false negative fraction for screening
//' @return data-frame with elements time, X, Y, Z (for state probabilities) and I (for incidence)
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
screening_model_1_predictions(std::vector<double> time, std::vector<double> tj,
			      double shape1=1, double scale1=1,
			      double shape2=1, double scale2=1,
			      double beta=0.05) {
  // Initialize the ScreeningModel with appropriate parameters
  ScreeningModel1 m(tj, beta,
		    [&](double u) { return R::dweibull(u,shape1,scale1,0); },
		    [&](double u) { return R::pweibull(u,shape1,scale1,0,0); },
		    [&](double u) { return R::dweibull(u,shape2,scale2,0); },
		    [&](double u) { return R::pweibull(u,shape2,scale2,0,0); });
  Rcpp::NumericMatrix out(time.size(), 4);
  for (size_t i=0; i<time.size(); i++) {
    // Calculate the probabilities for each state
    out(i,0) = m.X(time[i]);  // Probability of being in state X (healthy)
    out(i,1) = m.Y(time[i]);  // Probability of being in state Y (preclinical)
    out(i,2) = m.Z(time[i]);  // Probability of being in state Z (diagnosed)
    out(i,3) = m.I(time[i]);  // Probability of being in state Z (diagnosed)
  }
  using namespace Rcpp;
  // Return relevant quantities in a vector
  return DataFrame::create(_("time")=time,
			   _("X")=out.column(0),
			   _("Y")=out.column(1),
			   _("Z")=out.column(2),
			   _("I")=out.column(3));
}

//' Do likelihood calculations for ScreeningModel1
//' @param inputs list of list with elements of t for the evaluation time, tj for the screening times and type for the type of likelihood (1=No cancer detected, 2=Screen-detected cancer, 3=Interval cancer)
//' @param tj double vector of screening times
//' @param scale1 Weibull scale for onset
//' @param shape Weibull shape for onset
//' @param shape2 Weibull shape for clinical diagnosis
//' @param scale2 Weibull scale for clinical diagnosis
//' @param beta false negative fraction for screening
//' @return list with element t for the times t and element p for a numeric matrix of the probabilities, including column names X, Y and Z.
//' @export
// [[Rcpp::export]]
std::vector<double>
screening_model_1_likes(Rcpp::List inputs, double shape1 = 1.0, double scale1 = 1.0, 
			double shape2 = 1.0, double scale2 = 1.0, double beta = 0.05) {
  using Rcpp::as;
  std::vector<double> out(inputs.size());
  ScreeningModel1 m({}, beta,
		    [&](double u) { return R::dweibull(u,shape1,scale1,0); },
		    [&](double u) { return R::pweibull(u,shape1,scale1,0,0); },
		    [&](double u) { return R::dweibull(u,shape2,scale2,0); },
		    [&](double u) { return R::pweibull(u,shape2,scale2,0,0); });
  for (int i = 0; i < inputs.size(); i++) {
    Rcpp::List input = inputs(i);
    double t = as<double>(input("t"));
    int type = as<int>(input("type"));
    m.update(as<std::vector<double>>(input("ti")));
    // Calculate the likelihood contribution for this observation
    if (type == 1) {  // No cancer detected: P_X(t) + P_Y(t)
      out[i] = m.X(t) + m.Y(t);
    } 
    else if (type == 2) {  // Screen-detected cancer: P_Y(t-) * (1 - beta)
      out[i] = m.Y(t) * (1 - beta);
    } 
    else if (type == 3) {  // Interval cancer: I(t)
      out[i] = m.I(t);
    } 
    else {
      out[i] = -1.0;  // Invalid type
    }
    if (i % 10 == 0) {
      R_CheckUserInterrupt();  // Check if user hit Ctrl-C to stop execution
    }
  }
  return out;
}
