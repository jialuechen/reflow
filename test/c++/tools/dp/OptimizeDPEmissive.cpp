// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#include "libflow/core/utils/constant.h"
#include "OptimizeDPEmissive.h"

using namespace std ;
using namespace libflow;
using namespace Eigen;


// constructor
OptimizeDPEmissive::OptimizeDPEmissive(const double &p_alpha,
                                       const std::function<double(double, double)> &p_PI,
                                       const std::function< double(double, double) >     &p_cBar,  const double   &p_s, const double &p_lambda,
                                       const double &p_dt,
                                       const double   &p_maturity,
                                       const  double &p_lMax, const double &p_lStep, const  std::vector <std::array< double, 2>  >   &p_extrem):
    m_alpha(p_alpha), m_PI(p_PI),
    m_cBar(p_cBar), m_s(p_s), m_lambda(p_lambda), m_dt(p_dt), m_maturity(p_maturity), m_lMax(p_lMax), m_lStep(p_lStep),
    m_extrem(p_extrem)
{}

Array< bool, Dynamic, 1> OptimizeDPEmissive::getDimensionToSplit() const
{
    Array< bool, Dynamic, 1> bDim = Array< bool, Dynamic, 1>::Constant(2, true);
    return  bDim ;
}

// for parallelism
std::vector< std::array< double, 2> > OptimizeDPEmissive::getCone(const  vector<  std::array< double, 2>  > &p_xInit) const
{
    vector< array< double, 2> > xReached(2);
    xReached[0][0] = p_xInit[0][0]  ; // Q only increases
    xReached[0][1] = m_extrem[0][1]  ; // whole domain due to demand which is unbounded
    xReached[1][0] = p_xInit[1][0]  ;  // L only increases
    xReached[1][1] = p_xInit[1][1] + m_lMax * m_dt  ; // maximal increase given by the control
    return xReached;
}

// one step in optimization from stock point for all simulations
std::pair< ArrayXXd, ArrayXXd> OptimizeDPEmissive::stepOptimize(const   std::shared_ptr< libflow::SpaceGrid> &p_grid, const ArrayXd   &p_stock,
        const  std::vector< ContinuationValue> &p_condEsp,
        const std::vector < std::shared_ptr< ArrayXXd > > &) const
{
    std::pair< ArrayXXd, ArrayXXd> solutionAndControl;
    // to store final solution (here two regimes)
    solutionAndControl.first = ArrayXXd::Constant(m_simulator->getNbSimul(), 2, -libflow::infty);
    solutionAndControl.second =  ArrayXXd::Constant(m_simulator->getNbSimul(), 1, -libflow::infty);
    // demand
    ArrayXd demand = m_simulator->getParticles().array().row(0).transpose();
    // Gain (size number of simulations)
    ArrayXd gain(m_simulator->getNbSimul());
    double gainSubvention =  m_s * pow(p_stock(1), 1. - m_alpha); // subvention for non emissive energy
    for (int is = 0 ; is < m_simulator->getNbSimul(); ++is)
        gain(is) = m_PI(demand(is), p_stock(1)) + gainSubvention ; // gain by production and subvention
    ArrayXd ptStockNext(2);
    // time to maturity
    double timeToMat = m_maturity - m_simulator->getCurrentStep();
    // interpolator at the new step
    for (int is = 0 ; is < m_simulator->getNbSimul(); ++is)
    {
        for (int  iAl = 0; iAl < m_lMax / m_lStep ; ++iAl) // test all command for investment between 0 and lMax
        {
            double  l = iAl * m_lStep;
            // interpolator at the new step
            ptStockNext(0) =  p_stock(0) + std::max(demand(is) - p_stock(1), 0.) * m_dt;
            ptStockNext(1) =  p_stock(1) +  l * m_dt ;
            // first test we are inside the domain
            if (p_grid->isInside(ptStockNext))
            {
                // create an interpolator at the arrival point
                std::shared_ptr<libflow::Interpolator>  interpolator = p_grid->createInterpolator(ptStockNext);
                // calculate Y for this simulation with the optimal control
                double yLoc = p_condEsp[1].getASimulation(is, *interpolator);
                // local gain
                double gainLoc = (gain(is) - yLoc * std::max(demand(is) - p_stock(1), 0.) - m_cBar(l, p_stock(1))) * m_dt;
                //  gain + conditional expectation of future gains
                double condExp = gainLoc + p_condEsp[0].getASimulation(is, *interpolator);
                if (condExp >  solutionAndControl.first(is, 0)) // test optimality of the control
                {
                    solutionAndControl.first(is, 0) = condExp;
                    solutionAndControl.first(is, 1) = yLoc;
                    solutionAndControl.second(is, 0) = l;
                }
            }
        }
        // test if solution acceptable
        if (libflow::almostEqual(solutionAndControl.first(is, 0), - libflow::infty, 10))
        {
            // fix boundary condition
            solutionAndControl.first(is, 0) =  timeToMat * (m_PI(demand(is), p_stock(1)) + m_s * pow(p_stock(1), 1. - m_alpha) - m_lambda * std::max(demand(is) - p_stock(1), 0.));
            solutionAndControl.first(is, 1) = m_lambda ; // Q est maximal !!
            solutionAndControl.second(is, 0) = 0. ; // fix control to zero
        }
    }
    return solutionAndControl;
}

// one step in simulation for current simulation
void OptimizeDPEmissive::stepSimulate(const std::shared_ptr< libflow::SpaceGrid>   &p_grid, const std::vector< libflow::GridAndRegressedValue > &p_continuation,
                                      libflow::StateWithStocks &p_state,
                                      Ref<ArrayXd> p_phiInOut) const
{
    ArrayXd ptStock = p_state.getPtStock();
    ArrayXd ptStockNext(ptStock.size());
    double vOpt = - libflow::infty;
    double gainOpt = 0.;
    double lOpt = 0. ;
    double demand = p_state.getStochasticRealization()(0); // demand for this simulation
    ptStockNext(0) =  ptStock(0) + std::max(demand - ptStock(1), 0.) * m_dt;
    double   gain  = m_PI(demand, ptStock(1)) +  m_s * pow(ptStock(1), 1. - m_alpha) ; // gain from production and subvention
    double yOpt = 0. ;
    for (int  iAl = 0; iAl < m_lMax / m_lStep ; ++iAl) // test all command for investment between 0 and lMax
    {
        double  l = iAl * m_lStep;
        // interpolator at the new step
        ptStockNext(1) =  ptStock(1) +  l * m_dt ;
        // first test we are inside the domain
        if (p_grid->isInside(ptStockNext))
        {
            // calculate Y for this simulation with the control
            double yLoc = p_continuation[1].getValue(ptStockNext, p_state.getStochasticRealization());
            // local gain
            double gainLoc = (gain - yLoc * std::max(demand - ptStock(1), 0.) - m_cBar(l, ptStock(1))) * m_dt;
            //  gain + conditional expectation of future gains
            double condExp = gainLoc +  p_continuation[0].getValue(ptStockNext, p_state.getStochasticRealization());

            if (condExp >  vOpt) // test optimality of the control
            {
                vOpt = condExp;
                gainOpt = gainLoc;
                lOpt = l;
                yOpt = yLoc;
            }
        }
    }
    p_phiInOut(0) += gainOpt; // follow v value
    p_phiInOut(1) = yOpt ; // follow y value
    ptStockNext(1) =  ptStock(1) +  lOpt * m_dt ; // update state due to control
    p_state.setPtStock(ptStockNext);
}
