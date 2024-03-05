// Copyright (C) 2016 Fime
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#define BOOST_TEST_MODULE testRegressionForAmericanOptionsForSparseGrids
#define BOOST_TEST_DYN_LINK
#include <memory>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "test/c++/tools/simulators/BlackScholesSimulator.h"
#include "test/c++/tools/BasketOptions.h"
#include "libflow/regression/SparseRegression.h"

using namespace std;
using namespace Eigen ;
using namespace libflow;


/// For Clang < 3.7 (and above ?) to be compatible GCC 5.1 and above
namespace boost
{
namespace unit_test
{
namespace ut_detail
{
string normalize_test_case_name(const_string name)
{
    return (name[0] == '&' ? string(name.begin() + 1, name.size() - 1) : string(name.begin(), name.size()));
}
}
}
}


// American option by Longstaff Schwartz
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
        ArrayXXd asset = p_sim.stepBackwardAndGetParticles();
        VectorXd payOffLoc = p_payOff(asset);
        // conditional expectation
        p_regressor.updateSimulations(((iStep == (p_sim.getNbStep() - 1)) ? true : false), asset);
        // condition expectation
        VectorXd condEspec = p_regressor.getAllSimulations(Cash) * expRate;
        // arbitrage
        Cash = (condEspec.array() < payOffLoc.array()).select(payOffLoc, Cash * expRate);
    }
    return Cash.mean();
}

/// \brief Generic test case
///
/// \param p_bRotate                do we use rotation
/// \param p_nDim                   dimension of the problem
/// \param p_nbSimul                number of simulations used
/// \param p_level                  level of the sparse grid
/// \param p_degree                 degree for the interpolation
/// \param p_referenceValue         reference value
/// \param p_accuracyEqual          accuracy for equality
void testAmericanSparse(const bool &p_bRotate, const int &p_nDim, const int &p_nbSimul, const int &p_level,  const int &p_degree, const double &p_referenceValue, const double &p_accuracyEqual)
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
    SparseRegression regressor(p_level, weight, p_degree, p_bRotate);
    // bermudean value
    double value = resolutionAmericanSparseRegression(simulator, payoff, regressor);
    BOOST_CHECK_CLOSE(value, p_referenceValue, p_accuracyEqual);
}

/// Bouchard Warin test cases
/// "Monte Carlo Valuation of American options: facts and new algorithm "
BOOST_AUTO_TEST_CASE(testAmericanSparseBasket1D)
{
    // dimension
    int nDim = 1 ;
    int nbSimul = 100000;
    int level = 5;
    double referenceValue = 0.06031;
    double accuracyEqual = 0.2;
    // linear
    testAmericanSparse(false, nDim, nbSimul, level, 1, referenceValue, accuracyEqual);
    // quadratic
    testAmericanSparse(false, nDim, nbSimul, level, 2,   referenceValue, accuracyEqual);
    // cubic
    testAmericanSparse(false, nDim, nbSimul, level, 3,   referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanSparseBasket2D)
{
    // dimension
    int nDim = 2 ;
    int nbSimul = 100000;
    int level = 5;
    double referenceValue = 0.03882;
    double accuracyEqual = 0.9;
    // linear
    testAmericanSparse(false, nDim, nbSimul, level, 1, referenceValue, accuracyEqual);
    testAmericanSparse(true, nDim, nbSimul, level, 1, referenceValue, accuracyEqual);
    // quadratic
    testAmericanSparse(false, nDim, nbSimul, level, 2,   referenceValue, accuracyEqual);
    testAmericanSparse(true, nDim, nbSimul, level, 2, referenceValue, accuracyEqual);
    // cubic
    testAmericanSparse(false, nDim, nbSimul, level, 3,   referenceValue, accuracyEqual);
    testAmericanSparse(true, nDim, nbSimul, level, 3,   referenceValue, accuracyEqual);

}

BOOST_AUTO_TEST_CASE(testAmericanSparseBasket3D)
{
    // dimension
    int nDim = 3 ;
    int nbSimul = 100000;
    int level = 5;
    double referenceValue = 0.02947;
    double accuracyEqual = 0.7;
    // linear
    testAmericanSparse(false, nDim, nbSimul, level, 1, referenceValue, accuracyEqual);
    testAmericanSparse(true, nDim, nbSimul, level, 1, referenceValue, accuracyEqual);
    // quadratic
    testAmericanSparse(false, nDim, nbSimul, level, 2,   referenceValue, accuracyEqual);
    testAmericanSparse(true, nDim, nbSimul, level, 2,   referenceValue, accuracyEqual);
    // cubic
    testAmericanSparse(false, nDim, nbSimul, level, 3,   referenceValue, accuracyEqual);
    testAmericanSparse(true, nDim, nbSimul, level, 3,   referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanSparseBasket4D)
{
    // dimension
    int nDim = 4 ;
    int nbSimul = 100000;
    int level = 5;
    double referenceValue = 0.02404;
    double accuracyEqual = 1.;
    // linear
    testAmericanSparse(false, nDim, nbSimul, level, 1, referenceValue, accuracyEqual);
    // quadratic
    testAmericanSparse(false, nDim, nbSimul, level, 2,   referenceValue, accuracyEqual);
    // cubic
    testAmericanSparse(false, nDim, nbSimul, level, 3,   referenceValue, accuracyEqual);
}
