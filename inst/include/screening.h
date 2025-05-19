#ifndef SCREENING_SCREENING_H
#define SCREENING_SCREENING_H = 1

// [[Rcpp::depends(BH)]]
#include <Rcpp.h>
#include <boost/math/quadrature/gauss_kronrod.hpp>

namespace screening {

  // Abstract templated class for a base screening model
  // T1 is the type for f1
  // T2 is the type for S1
  // T3 is the type for f2
  // T4 is the type for S2
  template<class T1, class T2, class T3, class T4>
  class AbstractScreeningModel {
  public:
    T1 f1;
    T2 S1;
    T3 f2;
    T4 S2;
    std::vector<double> tj;
    size_t n;
    double error;
    double tol;
    AbstractScreeningModel(T1 f1,
			   T2 S1,
			   T3 f2,
			   T4 S2,
			   double tol = 1.0e-6) : f1(f1), S1(S1), f2(f2), S2(S2), tol(tol) { }
    virtual ~AbstractScreeningModel() { }
    // *NB*: use tj and n in the calculations!
    virtual void setup(double t) = 0;
    virtual double prod_beta(size_t i, size_t n, bool detected = false) = 0;
    double X(double s, bool reset = true) {
      if (reset) setup(s);
      return S1(s);
    }
    double Y(double t, bool reset = true) {
      using namespace boost::math::quadrature;
      if (reset) setup(t);
      double value = 0.0;
      for (size_t i=0; i<=n; i++) {
	auto fn = [&](double x) { return f1(x)*S2(t-x) * prod_beta(i,n); };
	value += gauss_kronrod<double, 15>::integrate(fn, tj[i], tj[i+1], 5, tol, &error);
      }
      return value;
    }
    double I(double t, bool reset=true) {
      using namespace boost::math::quadrature;
      if (reset) setup(t);
      double value = 0.0;
      for (size_t i=0; i<=n; i++) {
	auto fn = [&](double x) { return f1(x)*f2(t-x)*prod_beta(i,n); };
	value += gauss_kronrod<double, 15>::integrate(fn, tj[i], tj[i+1], 5, tol, &error);
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
	    auto inner = [&](double u) { return f1(s)*prod_beta(i,j)*f2(u-s); };
	    return gauss_kronrod<double, 15>::integrate(inner, std::max(s,tj[j]),
							tj[j+1], 5, tol, &error2);
	  };
	  value += gauss_kronrod<double, 15>::integrate(fn, tj[i], tj[i+1], 5, tol, &error);
	}
      }
      // screen-detected
      for (size_t i=0; i<n; i++) {
	for (size_t j=i; j<n; j++) {
	  auto fn = [&](double u) { return f1(u)*S2(tj[j+1]-u)*prod_beta(i,j,true); };
	  value += gauss_kronrod<double, 15>::integrate(fn, tj[i], tj[i+1], 5, tol, &error);
	}
      }
      return value;
    }
    virtual std::vector<double> likes(Rcpp::List inputs) = 0;
    Rcpp::DataFrame predictions(std::vector<double> t, std::vector<double> tj, bool simple = true) {
      using namespace Rcpp;
      NumericMatrix out(t.size(), 4);
      for (size_t i=0; i<t.size(); i++) {
	// Calculate the probabilities for each state
	out(i,0) = X(t[i]);  // Probability of being in state X (healthy)
	out(i,1) = Y(t[i]);  // Probability of being in state Y (preclinical)
	out(i,2) = Z(t[i], true, simple);  // Probability of being in state Z (diagnosed)
	out(i,3) = I(t[i]);  // Probability of being in state Z (diagnosed)
      }
      // Return relevant quantities in a vector
      return DataFrame::create(_("t")=t,
			       _("X")=out.column(0),
			       _("Y")=out.column(1),
			       _("Z")=out.column(2),
			       _("I")=out.column(3));
    }
  };

  // Templated class for a simple screening model with onset and clinical diagnosis
  template<class T1, class T2, class T3, class T4>
  class ScreeningModel1 : public AbstractScreeningModel<T1,T2,T3,T4> {
  public:
    size_t fulln;
    std::vector<double> ti;
    double beta;
    ScreeningModel1(std::vector<double> ti,
		    double beta,
		    T1 f1,
		    T2 S1,
		    T3 f2,
		    T4 S2,
		    double tol = 1e-6) :
      AbstractScreeningModel<T1,T2,T3,T4>(f1,S1,f2,S2, tol),
      fulln(ti.size()), ti(ti), beta(beta) { }
    // *NB*: use tj and n in the calculations!
    void setup(double t) {
      this->tj.resize(0);
      this->tj.push_back(0.0);
      for (size_t i=0; i<fulln && ti[i]<t; i++) {
	this->tj.push_back(ti[i]);
      }
      this->tj.push_back(t);
      this->n = this->tj.size()-2;
    }
    void update(std::vector<double> ti) {
      this->ti=ti;
      fulln=ti.size();
    }
    // can we generalise this function?
    double prod_beta(size_t i, size_t n, bool detected = false) {
      double value = 1.0;
      for (size_t j=i; j<n; j++) value *= beta;
      if (detected) value *= 1.0-beta; 
      return value;
    }
    std::vector<double> likes(Rcpp::List inputs) {
      using Rcpp::as;
      std::vector<double> out(inputs.size());
      for (int i = 0; i < inputs.size(); i++) {
	Rcpp::List input = inputs(i);
	double t = as<double>(input("t"));
	int type = as<int>(input("type"));
	update(as<std::vector<double>>(input("ti")));
	// Calculate the likelihood contribution for this observation
	if (type == 1) {  // No cancer detected: P_X(t) + P_Y(t)
	  out[i] = this->X(t) + this->Y(t);
	} 
	else if (type == 2) {  // Screen-detected cancer: P_Y(t-) * (1 - beta)
	  out[i] = this->Y(t) * (1 - beta);
	} 
	else if (type == 3) {  // Interval cancer: I(t)
	  out[i] = this->I(t);
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
  };
    
  // Second screening model with onset, clinical diagnosis and test sensitivity
  template<class T1, class T2, class T3, class T4, class T5>
  class ScreeningModel2 : public AbstractScreeningModel<T1,T2,T3,T4> {
  public:
    size_t fulln;
    std::vector<double> ti, yi, pi, pj;
    T5 PrFalseNeg;
    ScreeningModel2(std::vector<double> ti,
		    std::vector<double> yi,
		    std::vector<double> pi,
		    T1 f1, T2 S1, T3 f2, T4 S2, T5 PrFalseNeg,
		    double tol = 1e-6) :
      AbstractScreeningModel<T1,T2,T3,T4>(f1,S1,f2,S2, tol),     
      fulln(ti.size()), ti(ti), yi(yi), pi(pi), PrFalseNeg(PrFalseNeg) {}
    // *NB*: use tj and n in the calculations!
    void setup(double t) {
      this->tj.resize(0);
      pj.resize(0);
      this->tj.push_back(0.0);
      pj.push_back(1.0);
      for (size_t i=0; i<fulln && ti[i]<t; i++) {
	this->tj.push_back(ti[i]);
	pj.push_back(pi[i]);
      }
      this->tj.push_back(t);
      pj.push_back(pi[this->n]);
      this->n = this->tj.size()-2;
    }
    void update(std::vector<double> ti,
		std::vector<double> yi,
		std::vector<double> pi) {
      this->ti=ti;
      this->yi=yi;
      this->pi=pi;
      fulln=this->ti.size();
    }
    double prod_beta(size_t i, size_t n, bool detected = false) {
      double beta = 1.0;
      for (size_t j=i; j<n; j++) beta *= PrFalseNeg(yi[j]);
      if (detected) beta *= 1.0 - PrFalseNeg(yi[this->n]); 
      return beta;
    }
    double like_negative_screening(double s, bool reset = true) {
      using namespace boost::math::quadrature;
      if (reset) setup(s);
      double value = this->S1(s)*prod_beta(1,this->n);
      for (size_t i=0; i<=this->n; i++) {
	auto fn = [&](double x) {
	  return this->f1(x)*this->S2(s-x) * prod_beta(1,this->n); // pow(beta,n-i)
	};
	value += gauss_kronrod<double, 15>::integrate(fn, this->tj[i], this->tj[i+1], 5, this->tol, &this->error);
      }
      return value;
    }
    double like_interval_cancer(double t, bool reset=true) {
      using namespace boost::math::quadrature;
      if (reset) setup(t);
      double value = 0.0;
      for (size_t i=0; i<=this->n; i++) {
	auto fn = [&](double x) {
	  return this->f1(x)*this->f2(t-x)*prod_beta(1,this->n);
	};
	value += gauss_kronrod<double, 15>::integrate(fn, this->tj[i], this->tj[i+1], 5, this->tol, &this->error);
      }
      return value;
    }
    double like_screen_detected_cancer(double t, bool reset = true) {
      using namespace boost::math::quadrature;
      if (reset) setup(t);
      double value = 0.0;
      for (size_t i=0; i<=this->n; i++) {
	auto fn = [&](double x) {
	  return this->f1(x)*this->S2(t-x) * prod_beta(1,this->n,true);
	};
	value += gauss_kronrod<double, 15>::integrate(fn, this->tj[i], this->tj[i+1], 5, this->tol, &this->error);
      }
      return value;
    }
    // Which test characteristic should be used if t==t_j?
    // Should this be a different input?
    std::vector<double> likes(Rcpp::List inputs) {
      using Rcpp::as;
      std::vector<double> out(inputs.size());
      for (int i = 0; i < inputs.size(); i++) {
	Rcpp::List input = inputs(i);
	double t = as<double>(input("t"));
	int type = as<int>(input("type"));
	update(as<std::vector<double>>(input("ti")),
	       as<std::vector<double>>(input("yi")));
	// Calculate the likelihood contribution for this observation
	if (type == 1) {  // No cancer detected: P_X(t) + P_Y(t)
	  out[i] = this->X(t) + this->Y(t);
	} 
	else if (type == 2) {  // Screen-detected cancer: P_Y(t-) * (1 - beta)
	  out[i] = this->Y(t) * (1 - PrFalseNeg(yi[this->n-1]));
	} 
	else if (type == 3) {  // Interval cancer: I(t)
	  out[i] = this->I(t);
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
  };

  // Third screening model with onset, clinical diagnosis and screening histories
  // and test sensitivity
  template<class T1, class T2, class T3, class T4>
  class ScreeningModel3 : public AbstractScreeningModel<T1,T2,T3,T4> {
  public:
    size_t fulln;
    std::vector<double> ti, yi, pi, pj;
    double PrFalseNegBx;
    ScreeningModel3(std::vector<double> ti, // length n
		    std::vector<double> yi, // length n
		    std::vector<double> pi, // length n+1 (!!) 
		    T1 f1, T2 S1, T3 f2, T4 S2,
		    double PrFalseNegBx,
		    double tol = 1e-6, bool simple = true) :
      AbstractScreeningModel<T1,T2,T3,T4>(f1,S1,f2,S2,tol),
      fulln(ti.size()), ti(ti), yi(yi), pi(pi),
      PrFalseNegBx(PrFalseNegBx) { }
    // *NB*: use tj and n in the calculations!
    void setup(double t) {
      this->tj.resize(0);
      pj.resize(0);
      this->tj.push_back(0.0);
      pj.push_back(1.0);
      for (size_t i=0; i<fulln && ti[i]<t; i++) {
	this->tj.push_back(ti[i]);
	pj.push_back(pi[i]);
      }
      this->tj.push_back(t);
      pj.push_back(pi[this->n+1]);
      this->n = this->tj.size()-2;
    }
    void update(std::vector<double> ti, std::vector<double> yi, std::vector<double> pi) {
      this->ti=ti;
      this->yi=yi;
      this->pi=pi;
      fulln=ti.size();
    }
    double prod_beta(size_t i, size_t n, bool detected = false) {
      double beta = 1.0;
      for (size_t j=i; j<n; j++) beta *= pj[j];
      if (detected) beta *= 1.0 - PrFalseNegBx; 
      return beta;
    }
    double like_neg_screening(double s, bool reset = true) {
      using namespace boost::math::quadrature;
      if (reset) setup(s);
      double value = this->S1(s)*prod_beta(1,this->n);
      for (size_t i=0; i<=this->n; i++) {
	auto fn = [&](double x) {
	  return this->f1(x)*this->S2(s-x) * prod_beta(1,this->n);
	};
	value += gauss_kronrod<double, 15>::integrate(fn, this->tj[i], this->tj[i+1], 5, this->tol, &this->error);
      }
      return value;
    }
    double like_interval_cancer(double t, bool reset=true) {
      using namespace boost::math::quadrature;
      if (reset) setup(t);
      double value = 0.0;
      for (size_t i=0; i<=this->n; i++) {
	auto fn = [&](double x) {
	  return this->f1(x)*this->f2(t-x)*prod_beta(1,this->n);
	};
	value += gauss_kronrod<double, 15>::integrate(fn, this->tj[i], this->tj[i+1], 5, this->tol, &this->error);
      }
      return value;
    }
    double like_screen_detected_cancer(double t, bool reset=true) {
      using namespace boost::math::quadrature;
      if (reset) setup(t);
      double value = 0.0;
      for (size_t i=0; i<=this->n; i++) {
	auto fn = [&](double x) {
	  return this->f1(x)*this->f2(t-x)*prod_beta(1,this->n+1); // requires pj[n+1] = P(Biopsy|PSA_{n+1})P(True pos biopsy|Cancer)
	};
	value += gauss_kronrod<double, 15>::integrate(fn, this->tj[i], this->tj[i+1], 5, this->tol, &this->error);
      }
      return value;
    }
    // Which test characteristic should be used if t==t_j?
    // Should this be a different input?
    std::vector<double> likes(Rcpp::List inputs) {
      using Rcpp::as;
      std::vector<double> out(inputs.size());
      for (int i = 0; i < inputs.size(); i++) {
	Rcpp::List input = inputs(i);
	double t = as<double>(input("t"));
	int type = as<int>(input("type"));
	update(as<std::vector<double>>(input("ti")),
	       as<std::vector<double>>(input("yi")));
	// Calculate the likelihood contribution for this observation
	if (type == 1) {  // No cancer detected: P_X(t) + P_Y(t)
	  out[i] = this->X(t) + this->Y(t);
	} 
	else if (type == 2) {  // Screen-detected cancer: P_Y(t-) * (1 - beta)
	  out[i] = this->Y(t) * prod_beta(1,this->n) * (1 - PrFalseNegBx);
	} 
	else if (type == 3) {  // Interval cancer: I(t)
	  out[i] = this->I(t);
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
  };

  
} // end of namespace screening

#endif /* end of include guard: SCREENING_SCREENING_H */
