// Copyright (C) 2016 Fime

// This code is published under the GNU Lesser General Public License (GNU LGPL)
#include <boost/timer/timer.hpp>
#include <Eigen/Dense>
#include "test/c++/tools/simulators/BlackScholesSimulator.h"
#include "test/c++/tools/BasketOptions.h"
#include "libflow/core/utils/types.h"
#include "libflow/regression/SparseRegression.h"

using namespace std;
using namespace Eigen ;
using namespace libflow;


// american option by LongStaff Schwarz
//  p_sim        Monte Carlo simulator
//  p_payOff     Option pay off
//  p_regressor  regressor object
template < class Simulator, class PayOff, class Regressor   >
double resolutionAmericanSparseRegression(Simulator &p_sim, const PayOff &p_payOff,  Regressor &p_regressor)
{
    double step = p_sim.getStep();
    // asset simulated under the neutral risk probability : get the trend of first asset to get interest rate
    double expRate = exp(-step * p_sim.getMu()(0));
    // Terminal
    VectorXd Cash(p_payOff(p_sim.getParticles()));
    for (int iStep = 0; iStep < p_sim.getNbStep(); ++iStep)
    {
        shared_ptr<ArrayXXd> asset(new ArrayXXd(p_sim.stepBackwardAndGetParticles()));
        VectorXd payOffLoc = p_payOff(*asset);
        // conditional espectation
        p_regressor.updateSimulations(((iStep == (p_sim.getNbStep() - 1)) ? true : false), asset);
        // condition espection
        VectorXd condEspec = p_regressor.getAllSimulations(Cash) * expRate;
        // arbitrage
        Cash = (condEspec.array() < payOffLoc.array()).select(payOffLoc, Cash * expRate);
    }
    return Cash.mean();
}

/// \brief Generic test case
/// \param p_nDim                   dimesnion of the problem
/// \param p_nbSimul                number of simulations used
/// \param p_level                  level of the sparse grid
/// \param p_degree                 degree for the intepolation
/// \param p_referenceValue         reference value
/// \param p_bNoRescale             If true avoid rescalin as in Bouchard Warin
void testAmericanSparse(const int &p_nDim, const int &p_nbSimul, const int &p_level,  const int &p_degree,
                        const double &p_referenceValue, bool p_bNoRescale = false)
{
    VectorXd initialValues = ArrayXd::Constant(p_nDim, 1.);
    VectorXd sigma  = ArrayXd::Constant(p_nDim, 0.2);
    VectorXd mu  = ArrayXd::Constant(p_nDim, 0.05);
    MatrixXd corr = MatrixXd::Zero(p_nDim, p_nDim);
    double T = 1. ;
    int nDate = 10 ;
    corr.diagonal().setConstant(1.);
    double strike = 1.;
    // simulator
    BlackScholesSimulator simulator(initialValues, sigma, mu, corr, T, nDate, p_nbSimul, false);
    // payoff
    BasketPut payoff(strike);
    // regressor
    ArrayXd weight = ArrayXd::Constant(p_nDim, 1.);
    boost::timer::auto_cpu_timer t;
    SparseRegression regressor(p_level, weight, p_degree, p_bNoRescale);
    // bermudean value
    double value = resolutionAmericanSparseRegression(simulator, payoff, regressor);
    std::cout << " Value " << value << " error " << std::fabs(value - p_referenceValue) << std::endl ;
}

int main()
{
    {
        // REFERENCE 0.06031 0.03882 0.02947 0.02404 0.02046 0.01831
        int nDim = 1 ;
        int nbSimul = 100000;
        // dimension
        for (int is = 1; is < 4; ++is)
        {
            for (int level = 4; level <= 6; ++level)
            {
                double referenceValue = 0.06031;
                std::cout << " nDim " << nDim << " level " << level << " Nsim " << nbSimul << std::endl;
                std::cout << " LINEAR " << std::endl;
                testAmericanSparse(nDim, nbSimul, level, 1,   referenceValue);
                std::cout << " LINEAR No Rescale " << std::endl;
                testAmericanSparse(nDim, nbSimul, level, 1,   referenceValue, true);
                // std::cout << " QUADRATIC " << std::endl;
                // testAmericanSparse(nDim, nbSimul, level, 2,   referenceValue);
                // std::cout << " CUBIC " << std::endl;
                // testAmericanSparse(nDim, nbSimul, level, 3,   referenceValue);
            }
            nbSimul *= 5;
        }
    }

    {
        // REFERENCE 0.06031 0.03882 0.02947 0.02404 0.02046 0.01831
        int nDim = 2 ;
        int nbSimul = 100000;
        // dimension
        for (int is = 1; is < 4; ++is)
        {
            for (int level = 4; level <= 6; ++level)
            {
                double referenceValue = 0.03882;
                std::cout << " nDim " << nDim << " level " << level << " Nsim " << nbSimul << std::endl;
                std::cout << " LINEAR " << std::endl;
                testAmericanSparse(nDim, nbSimul, level, 1,   referenceValue);
                std::cout << " LINEAR No Rescale " << std::endl;
                testAmericanSparse(nDim, nbSimul, level, 1,   referenceValue, true);
                // std::cout << " QUADRATIC " << std::endl;
                // testAmericanSparse(nDim, nbSimul, level, 2,   referenceValue);
                // std::cout << " CUBIC " << std::endl;
                // testAmericanSparse(nDim, nbSimul, level, 3,   referenceValue);
            }
            nbSimul *= 5;
        }
    }
    {
        // REFERENCE 0.06031 0.03882 0.02947 0.02404 0.02046 0.01831
        int nDim = 3 ;
        int nbSimul = 100000;
        // dimension
        for (int is = 1; is < 4; ++is)
        {
            for (int level = 4; level <= 6; ++level)
            {
                double referenceValue = 0.02947;
                std::cout << " nDim " << nDim << " level " << level << " Nsim " << nbSimul << std::endl;
                std::cout << " LINEAR " << std::endl;
                testAmericanSparse(nDim, nbSimul, level, 1,   referenceValue);
                std::cout << " LINEAR No Rescale " << std::endl;
                testAmericanSparse(nDim, nbSimul, level, 1,   referenceValue, true);
                // std::cout << " QUADRATIC " << std::endl;
                // testAmericanSparse(nDim, nbSimul, level, 2,   referenceValue);
                // std::cout << " CUBIC " << std::endl;
                // testAmericanSparse(nDim, nbSimul, level, 3,   referenceValue);
            }
            nbSimul *= 5;
        }
    }
    {
        // REFERENCE 0.06031 0.03882 0.02947 0.02404 0.02046 0.01831
        int nDim = 4 ;
        int nbSimul = 100000;
        // dimension
        for (int is = 1; is < 4; ++is)
        {
            for (int level = 4; level <= 6; ++level)
            {
                double referenceValue = 0.02404;
                std::cout << " nDim " << nDim << " level " << level << " Nsim " << nbSimul << std::endl;
                std::cout << " LINEAR " << std::endl;
                testAmericanSparse(nDim, nbSimul, level, 1,   referenceValue);
                std::cout << " LINEAR No Rescale " << std::endl;
                testAmericanSparse(nDim, nbSimul, level, 1,   referenceValue, true);
                // std::cout << " QUADRATIC " << std::endl;
                // testAmericanSparse(nDim, nbSimul, level, 2,   referenceValue);
                // std::cout << " CUBIC " << std::endl;
                // testAmericanSparse(nDim, nbSimul, level, 3,   referenceValue);
            }
            nbSimul *= 5;
        }
    }
    {
        // REFERENCE 0.06031 0.03882 0.02947 0.02404 0.02046 0.01831
        int nDim = 5 ;
        int nbSimul = 500000;
        // dimension
        for (int is = 1; is < 4; ++is)
        {
            for (int level = 4; level <= 6; ++level)
            {
                double referenceValue = 0.02046;
                std::cout << " nDim " << nDim << " level " << level << " Nsim " << nbSimul << std::endl;
                std::cout << " LINEAR " << std::endl;
                testAmericanSparse(nDim, nbSimul, level, 1,   referenceValue);
                std::cout << " LINEAR No Rescale " << std::endl;
                testAmericanSparse(nDim, nbSimul, level, 1,   referenceValue, true);
                // std::cout << " QUADRATIC " << std::endl;
                // testAmericanSparse(nDim, nbSimul, level, 2,   referenceValue);
                // std::cout << " CUBIC " << std::endl;
                // testAmericanSparse(nDim, nbSimul, level, 3,   referenceValue);
            }
            nbSimul *= 5;
        }
    }
    {
        // REFERENCE 0.06031 0.03882 0.02947 0.02404 0.02046 0.01831
        int nDim = 6 ;
        int nbSimul = 500000;
        // dimension
        for (int is = 1; is < 4; ++is)
        {
            for (int level = 4; level <= 6; ++level)
            {
                double referenceValue = 0.01831;
                std::cout << " nDim " << nDim << " level " << level << " Nsim " << nbSimul << std::endl;
                std::cout << " LINEAR " << std::endl;
                testAmericanSparse(nDim, nbSimul, level, 1,   referenceValue);
                std::cout << " LINEAR No Rescale " << std::endl;
                testAmericanSparse(nDim, nbSimul, level, 1,   referenceValue, true);
                // std::cout << " QUADRATIC " << std::endl;
                // testAmericanSparse(nDim, nbSimul, level, 2,   referenceValue);
                // std::cout << " CUBIC " << std::endl;
                // testAmericanSparse(nDim, nbSimul, level, 3,   referenceValue);
            }
            nbSimul *= 5;
        }
    }
    return 0;
}
