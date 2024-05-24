#define BOOST_TEST_MODULE testRegressionConvexificationForAmericanOptions
#define BOOST_TEST_DYN_LINK
#include <memory>
#include <iostream>   // std::cout
#include <fstream>
#include <string>     // std::string, std::to_string
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "test/c++/tools/simulators/BlackScholesSimulator.h"
#include "test/c++/tools/BasketOptions.h"
#include "reflow/regression/LocalLinearRegression.h"

using namespace std;
using namespace Eigen ;
using namespace reflow;


// American option by Longstaff Schwartz
//  p_sim           Monte Carlo simulator
//  p_payOff        Option pay off
//  p_regressor     regressor object
//  p_nbIterConvex  maximal number of convexification iterations
template < class Simulator, class PayOff, class Regressor   >
double resolutionAmericanConvexRegression(Simulator &p_sim, const PayOff &p_payOff,  Regressor &p_regressor,
        const int &p_nbIterConvex)
{
    // asset simulated under the neutral risk probability : get the trend of first asset to get interest rate
    double expRate = p_sim.getActuStep();
    // Terminal
    VectorXd Cash(p_payOff(p_sim.getParticles()));
    for (int iStep = 0; iStep < p_sim.getNbStep(); ++iStep)
    {
        ArrayXXd asset = p_sim.stepBackwardAndGetParticles();
        VectorXd payOffLoc = p_payOff(asset);
        // update conditional expectation operator for current Markov state
        p_regressor.updateSimulations(((iStep == (p_sim.getNbStep() - 1)) ? true : false), asset);
        // condition expectation
        VectorXd condEspec = p_regressor.getAllSimulationsConvex(Cash, p_nbIterConvex) * expRate;
        // arbitrage
        Cash = (condEspec.array() < payOffLoc.array()).select(payOffLoc, Cash * expRate);
    }
    return Cash.mean();
}

template< class LocalRegressor   >
void testAmericanConvexLocal(const int &p_nDim, const int &p_nbSimul,
                             const int &p_nMesh, const double &p_referenceValue,
                             const double &p_accuracyEqual,
                             const int &p_nbIterConvex)
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
    // mesh
    ArrayXi nbMesh = ArrayXi::Constant(p_nDim, p_nMesh);
    // regressor
    LocalRegressor regressor(nbMesh);
    // Bermudean value
    double value = resolutionAmericanConvexRegression(simulator, payoff, regressor, p_nbIterConvex);

    cout << " p_referenceValue " << p_referenceValue << " value " << value << std::endl ;
    BOOST_CHECK_CLOSE(value, p_referenceValue, p_accuracyEqual);
}



// Bouchard Warin test cases
// "Monte Carlo Valuation of American options: facts and new algorithm "
BOOST_AUTO_TEST_CASE(testAmericanLinearConvexBasket1D)
{
    // dimension
    int nDim = 1 ;
    int nbSimul = 10000;
    int nbMesh = 8;
    double referenceValue = 0.06031;
    double accuracyEqual = 0.5;
    int nbIterConvex = 5 ;
    testAmericanConvexLocal<LocalLinearRegression>(nDim, nbSimul, nbMesh, referenceValue, accuracyEqual, nbIterConvex);
}

// Bouchard Warin test cases
// "Monte Carlo Valuation of American options: facts and new algorithm "
BOOST_AUTO_TEST_CASE(testAmericanLinearConvexBasket2D)
{
    // dimension
    int nDim = 2 ;
    int nbSimul = 50000;
    int nbMesh = 5;
    double referenceValue = 0.03882;
    double accuracyEqual = 1.;
    int nbIterConvex = 10 ;
    testAmericanConvexLocal<LocalLinearRegression>(nDim, nbSimul, nbMesh, referenceValue, accuracyEqual, nbIterConvex);
}

// Bouchard Warin test cases
// "Monte Carlo Valuation of American options: facts and new algorithm "
BOOST_AUTO_TEST_CASE(testAmericanConvexBasket3D)
{
    // dimension
    int nDim = 3 ;
    int nbSimul = 100000;
    int nbMesh = 5;
    double referenceValue = 0.02947;
    double accuracyEqual = 1.;
    int nbIterConvex = 10 ;
    testAmericanConvexLocal<LocalLinearRegression>(nDim, nbSimul, nbMesh, referenceValue, accuracyEqual, nbIterConvex);
}
