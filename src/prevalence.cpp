// [[Rcpp::depends(BH)]]
#include <Rcpp.h>
#include <boost/math/quadrature/gauss_kronrod.hpp>

using namespace boost::math::quadrature;

// Simple wrapper class around findInterval()
class FindInterval {
public:
  std::vector<double> v;
  int ilo, mflag;
  FindInterval(std::vector<double> v) : v(v), ilo(1) {}
  int operator()(double x) {
    ilo = findInterval(&v[0], v.size(), x, FALSE, FALSE, ilo, &mflag);
    return ilo==0 ? 0 : ilo-1;
  }
  size_t size() {
    return v.size();
  }
};

class Tab1Lookup {
public:
  FindInterval brks;
  std::vector<double> data;
  Tab1Lookup(std::vector<double> brks,
	     std::vector<double> data) :
    brks(FindInterval(brks)), data(data) { }
  double operator()(double x) {
    return data[brks(x)];
  }
};

class Tab2Lookup {
public:
  FindInterval rows, cols;
  std::vector<double> data; // filled in by column
  size_t nrows;
  Tab2Lookup(std::vector<double> rows,
	     std::vector<double> cols,
	     std::vector<double> data) :
    rows(FindInterval(rows)), cols(FindInterval(cols)), data(data) {
    nrows = rows.size();
  }
  double operator()(double x, double y) {
    return data[rows(x)+cols(y)*nrows];
  }
};

// Class for modelling an illness-death model (0=Healthy, 1=Illness, 2=OtherDeath, 3=CancerDeath)
template<class T01, class T02, class T13_excess>
class IllnessDeathModel1 {
public:
  double error;
  T01 h01;
  T02 h02;
  T13_excess h13_excess;
  double tol;
  IllnessDeathModel1(T01 h01,  //' hazard for other causes of death by attained age and period
		     T02 h02,  //' hazard for illness onset by attained age and period
		     T13_excess h13_excess, //' excess mortality hazard for the illness state by attained age, period and duration 
		     double tol=1e-6) :
    h01(h01), h02(h02), h13_excess(h13_excess), tol(tol)  { }
  //' Proportion in the healthy state at attained age a and attained
  //' period t assuming everyone starts in the healthy state
  double pHealthy(double a, double t) {
    auto h = [&](double d) { return h01(a-d,t-d)+h02(a-d,t-d); };
    double H = gauss_kronrod<double, 15>::integrate(h, 0, a, 5, tol, &error);
    return exp(-H);
  }
  //' Proportion in the illness state at attained age a, attained
  //' period t and attained duration d assuming everyone starts in the
  //' healthy state. Is this a density rather than a probability?
  double pIllness(double a, double t, double d) {
    auto h = [&](double d) {return h13_excess(a-d,t-d,d)+h02(a-d,t-d); };
    double H = gauss_kronrod<double, 15>::integrate(h, 0, d, 5, tol, &error);
    return pHealthy(a-d,t-d)*h01(a-d,t-d)*exp(-H);
  }
  //' Proportion in the illness state at attained age a and
  //' period t assuming everyone starts in the healthy state.
  double pIllness(double a, double t) {
    auto fn = [&](double d) { return pIllness(a-d,t-d,d); };
    return gauss_kronrod<double, 15>::integrate(fn, 0, a, 5, tol, &error);
  }
  //' Illness prevalence proportion by attained age and period per live population
  double pPrevalence(double a, double t) {
    return pIllness(a,t)/(pIllness(a,t)+pHealthy(a,t));
  }
  //' Illness mortality rate per live population
  double rCancerMortality(double a, double t) {
    auto fn = [&](double d) { return pIllness(a,t,d)*h13_excess(a,t,d); };
    double flow = gauss_kronrod<double, 15>::integrate(fn, 0, a, 5, tol, &error);
    return flow/(pIllness(a,t)+pHealthy(a,t));
  }
  //' Illness incidence rate per live population
  double rIncidence(double a, double t) {
    return pHealthy(a,t)*h01(a,t)/(pIllness(a,t)+pHealthy(a,t));
  }
  double pOtherCausesOfDeath(double a, double t) {
    auto fn = [&](double x) { return h01(a-x,t-x)*(pHealthy(a-x,t-x)+pIllness(a-x,t-x)); };
    return gauss_kronrod<double, 15>::integrate(fn, 0, a, 5, tol, &error);
  }
  double pCancerDeath(double a, double t) {
    return 1.0 - (pHealthy(a,t)+pIllness(a,t)+pOtherCausesOfDeath(a,t));
  }
};

//' TODO: memoise using boost::flyweights
