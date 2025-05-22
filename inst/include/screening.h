#ifndef SCREENING_SCREENING_H
#define SCREENING_SCREENING_H = 1

// [[Rcpp::depends(BH)]]
#include <Rcpp.h>
#include <boost/math/quadrature/gauss_kronrod.hpp>

namespace screening {

  /**
     Currently, we assume three types of screening models:
     1. Screening episodes with end times of the episodes and whether a cancer was detected
     2. Screening episodes with end times of the episodes, the/a biomarker value and
        whether a cancer was detected
     3. Screening episodes with end times of the episodes, the/a biomarker value, whether a
        biopsy was undertaken and whether a cancer was detected

     We have factored some functionality into `AbstractScreeningModel`.

     To be resolved:
     - What if t is the same as an episode time? This is typically true for observed data. Assume cadlag?
     - Get the math right!

     Example 1:
     - Single screen at time t_1 with *no* follow-up after that time (that is, t_1=t)
     - Let t_0=0
     - Approach: n=1 and t=t_1+\epsilon for machine \epsilon (may lead to numerical issues with finding a solution)
     - Approach: when t=t_i then n=i-1 and include a flag and include tests through to t_i
     
     Example 2: OK
     - Single screen at time t_1 with follow-up to time t>t_1
     - Let t_0=0
     - Let n=1
     - Let t_2=t
     
     Example 3: OK
     - No screening episodes to time t
     - Let t_0=0
     - Let n=0
     - Let t_1=t
     
   */
  
  // Abstract templated class for a base screening model
  // T1 is the type for f1
  // T2 is the type for S1
  // T3 is the type for f2
  // T4 is the type for S2
  template<class T1, class T2, class T3, class T4>
  class AbstractScreeningModel {
  public:
    T1 f1; // density for onset
    T2 S1; // survival for onset
    T3 f2; // density for clinical symptoms
    T4 S2; // survival for clinical symptoms
    std::vector<double> ti, tj; // episodes times -- original and working set (prepended with 0 and appended with observed time)
    double tol; // integration tolerance (conservative!)
    size_t fulln, n; // number of episodes (original and working set)
    double error; // used in the integrations
    int offset; // 1 if the follow-up ends at a screen, otherwise 0
    AbstractScreeningModel(T1 f1,
			   T2 S1,
			   T3 f2,
			   T4 S2,
			   double tol = 1.0e-6) : f1(f1), S1(S1), f2(f2), S2(S2),
						  tol(tol), fulln(0) { }
    //' default destructor
    virtual ~AbstractScreeningModel() { }
    //' Product of false negative test outcomes possibly followed by a true positive test at the end.
    //' If i=n=0 then trivially no tests. This implies that i=n will also lead to no tests.
    virtual double prod_beta(size_t i, size_t n, bool detected = false) = 0;
    //' Calculate likelihoods for different inputs
    virtual std::vector<double> likes(Rcpp::List inputs) = 0;
    //' Set up tj and n for observation time t 
    void setup(double t) {
      size_t i;
      tj.resize(0);
      tj.push_back(0.0);
      for (i=0; i<fulln && ti[i]<t; i++) {
	tj.push_back(ti[i]);
      }
      offset = (i+1<fulln && ti[i]==t) ? 1 : 0;
      tj.push_back(t);
      n = tj.size()-2; // number of episodes (excluding the last episode if time equals t)
    }
    void update(std::vector<double> ti) {
      this->ti=ti;
      fulln=ti.size();
    }
    double X(double t, bool reset = true) {
      if (reset) setup(t);
      return S1(t);
    }
    double Y(double t, bool reset = true) {
      using namespace boost::math::quadrature;
      if (reset) setup(t);
      double value = 0.0;
      for (size_t i=0; i<=n; i++) {
	// For onset between tj[i] to tj[i+1] with false negative tests through to the end of follow-up
	auto fn = [&](double x) { return f1(x)*S2(t-x) * prod_beta(i, n+offset); };
	value += gauss_kronrod<double, 15>::integrate(fn, tj[i], tj[i+1], 5, tol, &error);
      }
      return value;
    }
    double I(double t, bool reset=true) {
      using namespace boost::math::quadrature;
      if (reset) setup(t);
      double value = 0.0;
      for (size_t i=0; i<=n; i++) {
	auto fn = [&](double x) { return f1(x)*f2(t-x)*prod_beta(i, n); };
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
      for (size_t i=0; i<n+offset; i++) {
	for (size_t j=i; j<n+offset; j++) {
	  auto fn = [&](double u) { return f1(u)*S2(tj[j+1]-u)*prod_beta(i,j,true); };
	  value += gauss_kronrod<double, 15>::integrate(fn, tj[i], tj[i+1], 5, tol, &error);
	}
      }
      return value;
    }
    Rcpp::DataFrame predictions(std::vector<double> t, bool simple = true) {
      using namespace Rcpp;
      // this->update(ti);
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

  // A simple screening model with onset and clinical diagnosis
  template<class T1, class T2, class T3, class T4>
  class ScreeningModel1 : public AbstractScreeningModel<T1,T2,T3,T4> {
  public:
    double beta;
    ScreeningModel1(T1 f1,
		    T2 S1,
		    T3 f2,
		    T4 S2,
		    double beta,
		    double tol = 1e-6) :
      AbstractScreeningModel<T1,T2,T3,T4>(f1,S1,f2,S2, tol),
      beta(beta) { }
    double prod_beta(size_t i, size_t n, bool detected = false) {
      double value = 1.0;
      for (size_t j=i; j<n; j++) value *= (detected && j+1==n ? 1.0-beta : beta);
      return value;
    }
    std::vector<double> likes(Rcpp::List inputs) {
      using Rcpp::as;
      std::vector<double> out(inputs.size());
      for (int i = 0; i < inputs.size(); i++) {
	Rcpp::List input = inputs(i);
	double t = as<double>(input("t"));
	int type = as<int>(input("type"));
	this->update(as<std::vector<double>>(input("ti")));
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
    std::vector<double> yi;
    T5 PrFalseNeg;
    ScreeningModel2(T1 f1, T2 S1, T3 f2, T4 S2, T5 PrFalseNeg,
		    double tol = 1e-6) :
      AbstractScreeningModel<T1,T2,T3,T4>(f1,S1,f2,S2, tol),     
      PrFalseNeg(PrFalseNeg) {}
    void update(std::vector<double> ti,
		std::vector<double> yi) {
      AbstractScreeningModel<T1,T2,T3,T4>::update(ti);
      this->yi=yi;
    }
    double prod_beta(size_t i, size_t n, bool detected = false) {
      double value = 1.0;
      for (size_t j=i; j<this->n; j++)
	value *= (detected && j+1==this->n ? 1.0-PrFalseNeg(yi[j]) : PrFalseNeg(yi[j]));
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
	  out[i] = this->Y(t) * (1 - PrFalseNeg(yi[this->n-1])); // is this correct??
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
  template<class T1, class T2, class T3, class T4, class T5>
  class ScreeningModel3 : public AbstractScreeningModel<T1,T2,T3,T4> {
  public:
    std::vector<double> yi; // biomarker - should this be std::vector<T>?
    std::vector<int> bxi; // biopsy indicator
    T5 PrNoBx;
    double PrFalseNegBx;
    ScreeningModel3(T1 f1, T2 S1, T3 f2, T4 S2,
		    T5 PrNoBx,
		    double PrFalseNegBx,
		    double tol = 1e-6) :
      AbstractScreeningModel<T1,T2,T3,T4>(f1,S1,f2,S2,tol),
      PrNoBx(PrNoBx), PrFalseNegBx(PrFalseNegBx) { }
    void update(std::vector<double> ti, std::vector<double> yi, std::vector<int> bxi) {
      AbstractScreeningModel<T1,T2,T3,T4>::update(ti);
      this->yi=yi;
      this->bxi=bxi;
    }
    double prod_bx(size_t i, size_t n, bool detected = false) {
      double value = 1.0;
      for (size_t j=i; j<n; j++)
	value *= (bxi[j]==0 ? PrNoBx(yi[j]) : (1.0-PrNoBx(yi[j])));
      return value;
    }
    double prod_beta(size_t i, size_t n, bool detected = false) {
      double value = 1.0;
      for (size_t j=i; j<n; j++)
	value *= (bxi[j]==0 ? PrNoBx(yi[j]) : (1.0-PrNoBx(yi[j]))*(detected && j+1==n ? (1.0 - PrFalseNegBx) : PrFalseNegBx));
      return value;
    }
    double like_neg_screening(double s, bool reset = true) {
      using namespace boost::math::quadrature;
      if (reset) this->setup(s);
      double value = this->S1(s)*prod_bx(0,this->n);
      for (size_t i=0; i<=this->n; i++) {
	auto fn = [&](double x) {
	  return this->f1(x)*this->S2(s-x) * prod_bx(0,i) * prod_beta(i,this->n);
	};
	value += gauss_kronrod<double, 15>::integrate(fn, this->tj[i], this->tj[i+1],
						      5, this->tol, &this->error);
      }
      return value;
    }
    double like_interval_cancer(double t, bool reset=true) {
      using namespace boost::math::quadrature;
      if (reset) this->setup(t);
      double value = 0.0;
      for (size_t i=0; i<=this->n; i++) {
	auto fn = [&](double x) {
	  return this->f1(x)*this->f2(t-x)*prod_bx(0,i)*prod_beta(i,this->n);
	};
	value += gauss_kronrod<double, 15>::integrate(fn, this->tj[i], this->tj[i+1], 5, this->tol, &this->error);
      }
      return value;
    }
    double like_screen_detected_cancer(double t, bool reset=true) {
      using namespace boost::math::quadrature;
      if (reset) this->setup(t);
      double value = 0.0;
      for (size_t i=0; i<=this->n; i++) {
	auto fn = [&](double x) {
	  return this->f1(x)*this->f2(t-x)*prod_bx(0,i)*
	    prod_beta(i,this->n+this->offset,true); 
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
	       as<std::vector<double>>(input("yi")),
	       as<std::vector<int>>(input("bxi")));
	// Calculate the likelihood contribution for this observation
	if (type == 1) {  // No cancer detected: P_X(t) + P_Y(t)
	  out[i] = like_neg_screening(t);
	} 
	else if (type == 2) {  // Screen-detected cancer: P_Y(t-) * (1 - beta)
	  out[i] = like_screen_detected_cancer(t);
	} 
	else if (type == 3) {  // Interval cancer: I(t)
	  out[i] = like_interval_cancer(t);
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
