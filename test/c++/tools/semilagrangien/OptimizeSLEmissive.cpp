#include <iostream>
#include "reflow/core/utils/constant.h"
#include "OptimizeSLEmissive.h"

using namespace reflow;
using namespace Eigen ;
using namespace std ;

// constructor
OptimizeSLEmissive::OptimizeSLEmissive(const double &p_alpha,  const double &p_m, const double &p_sig, const std::function<double(double, double)> &p_PI,
                                       const std::function< double(double, double) >     &p_cBar,  const double   &p_s, const double &p_dt,
                                       const  double &p_lMax, const double &p_lStep, const  std::vector <std::array< double, 2>  >   &p_extrem):
    m_alpha(p_alpha), m_m(p_m), m_sig(p_sig), m_PI(p_PI), m_cBar(p_cBar), m_s(p_s), m_dt(p_dt),
    m_lMax(p_lMax), m_lStep(p_lStep), m_extrem(p_extrem) {}

Array< bool, Dynamic, 1> OptimizeSLEmissive::getDimensionToSplit() const
{
    Array< bool, Dynamic, 1> bDim = Array< bool, Dynamic, 1>::Constant(3, true);
    return  bDim ;
}


// for parallelism
vector< array< double, 2> >  OptimizeSLEmissive::getCone(const  vector<  array< double, 2>  > &p_xInit) const
{
    vector< array< double, 2> > xReached(3);
    xReached[0][0] = p_xInit[0][0]  +   m_alpha * (m_m - m_extrem[0][1]) * m_dt -  m_sig * sqrt(m_dt); // demand "cone" driven by maximal value allowed for demand
    xReached[0][1] = p_xInit[0][1]  +  m_alpha * m_m * m_dt +   m_sig * sqrt(m_dt) ; // low value for demand is taken equal to 0
    xReached[1][0] = p_xInit[1][0]  ;// Q only increases
    xReached[1][1] = p_xInit[1][1]  + m_extrem[0][1] * m_dt  ; // Q increase bounded by  maximal demand
    xReached[2][0] = p_xInit[2][0]  ; // L only increases
    xReached[2][1] = p_xInit[2][1] + m_lMax * m_dt  ;// maximal increase given by the control
    return xReached;
}


// one step in optimization from current point
std::pair< ArrayXd, ArrayXd> OptimizeSLEmissive::stepOptimize(const ArrayXd   &p_point,
        const vector< shared_ptr<SemiLagrangEspCond> > &p_semiLag,
        const double &, const ArrayXd &) const
{
    pair< ArrayXd, ArrayXd> solutionAndControl;
    solutionAndControl.first.resize(2);
    solutionAndControl.second.resize(1);
    ArrayXXd sig = ArrayXXd::Zero(3, 1) ;
    sig(0, 0) = m_sig;
    double vOpt = - reflow::infty;
    double yOpt = 0. ;
    double lOpt = 0 ;
    ArrayXd b(3);
    b(0) =  m_alpha * (m_m - p_point(0)) ; // trend
    b(1) =  max(p_point(0) - p_point(2), 0.);
    // gain already possible to calculate (production and subvention)
    double gainFirst = m_PI(p_point(0), p_point(2)) + m_s * pow(p_point(2), 1. - m_alpha)  ;
    for (int iAl = 0; iAl < m_lMax / m_lStep ; ++iAl) // test all command for investment between 0 and lMax
    {
        double  l = iAl * m_lStep;
        b(2) =  l ;
        pair<double, bool> lagrangY = p_semiLag[1]->oneStep(p_point, b, sig, m_dt); // for the control calculate y
        if (lagrangY.second) // is the control admissible
        {
            pair<double, bool> lagrang = p_semiLag[0]->oneStep(p_point, b, sig, m_dt);  // one step for v
            // gain function
            double gain = m_dt * (gainFirst - lagrangY.first * b(1)  - m_cBar(l, p_point(2)));
            double arbitrage = gain + lagrang.first;
            if (arbitrage > vOpt) // optimality  of the control
            {
                vOpt = arbitrage; // upgrade solution v
                yOpt =  lagrangY.first; // store y
                lOpt = l; // upgrade optimal control
            }
        }
    }

    if (reflow::almostEqual(vOpt, - reflow::infty, 10))
    {
        std::cout << " Reduce time step " << std::endl ;
        abort();
    }
    solutionAndControl.first(0) =  vOpt; // send back v function
    solutionAndControl.first(1) =  yOpt; // send back y function
    solutionAndControl.second(0) =  lOpt; // send back optimal control
    return solutionAndControl;
}

// one step in simulation for current simulation
void OptimizeSLEmissive::stepSimulate(const SpaceGrid   &p_gridNext,
                                      const  std::vector< std::shared_ptr< reflow::SemiLagrangEspCond> > &p_semiLag,
                                      Ref<ArrayXd>  p_state,   int &,
                                      const ArrayXd &p_gaussian,
                                      const ArrayXd &,
                                      Ref<ArrayXd> p_phiInOut) const
{
    ArrayXd state = p_state;
    ArrayXXd sig = ArrayXXd::Zero(3, 1) ; // diffusion matrix for semi Lagrangian
    sig(0, 0) = m_sig;
    double vOpt = - reflow::infty;
    double lOpt = 0 ;
    double yOpt = 0;
    ArrayXd b(3);
    b(0) =  m_alpha * (m_m - p_state(0)) ; // trend for D (independent of control)
    b(1) =  max(p_state(0) - p_state(2), 0.); // trend for Q (independent of control)
    double gainFirst = m_PI(p_state(0), p_state(2)) + m_s * pow(p_state(2), 1. - m_alpha)  ; // gain for production and subvention
    for (int iAl = 0; iAl < m_lMax / m_lStep ; ++iAl) // recalculate the optimal control
    {
        double  l = iAl * m_lStep;
        b(2) =  l ;
        pair<double, bool> lagrangY = p_semiLag[1]->oneStep(p_state, b, sig, m_dt); // calculate y for this control
        if (lagrangY.second)
        {
            pair<double, bool> lagrang = p_semiLag[0]->oneStep(p_state, b, sig, m_dt); // calculate the function value v
            // gain function
            double gain = m_dt * (gainFirst - lagrangY.first * b(1)  - m_cBar(l, p_state(2)));
            double arbitrage = gain + lagrang.first;
            if (arbitrage > vOpt) // arbitrage
            {
                vOpt = arbitrage; // upgrade solution
                yOpt =  lagrangY.first; // upgrade y value
                lOpt = l; // upgrade optimal control
            }
        }
    }
    // gain function
    p_phiInOut(0) += m_dt * (gainFirst - yOpt * b(1) - m_cBar(lOpt, state(2))); // store v value
    p_phiInOut(1) = yOpt; // store y value
    // update state
    state(0) += m_alpha * (m_m - p_state(0)) * m_dt + m_sig * p_gaussian(0) * sqrt(m_dt); // demand (no control)
    state(1) += b(1) * m_dt; //Q
    state(2) += lOpt * m_dt; //L
    // truncate if necessary to stay inside domain.
    p_gridNext.truncatePoint(state);
    p_state = state ;
}


