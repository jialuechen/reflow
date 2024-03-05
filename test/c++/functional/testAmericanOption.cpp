#define BOOST_TEST_MODULE testRegressionForAmericanOptions
#define BOOST_TEST_DYN_LINK
#include <memory>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "test/c++/tools/simulators/BlackScholesSimulator.h"
#include "test/c++/tools/BasketOptions.h"
#include "libflow/core/utils/constant.h"
#include "libflow/regression/LocalLinearRegression.h"
#include "libflow/regression/LocalConstRegression.h"
#include "libflow/regression/LocalSameSizeLinearRegression.h"
#include "libflow/regression/LocalSameSizeConstRegression.h"
#include "libflow/regression/LocalGridKernelRegression.h"
#include "libflow/core/utils/Polynomials1D.h"
#include "libflow/regression/GlobalRegression.h"

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
double resolutionAmericanRegression(Simulator &p_sim, const PayOff &p_payOff,  Regressor &p_regressor)
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
        VectorXd condEspec = p_regressor.getAllSimulations(Cash) * expRate;
        // arbitrage
        Cash = (condEspec.array() < payOffLoc.array()).select(payOffLoc, Cash * expRate);
    }
    return Cash.mean();
}

template< class LocalRegressor   >
void testAmericanLocal(const int &p_nDim, const int &p_nbSimul, const int &p_nMesh, const double &p_referenceValue, const double &p_accuracyEqual)
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
    double value = resolutionAmericanRegression(simulator, payoff, regressor);
    BOOST_CHECK_CLOSE(value, p_referenceValue, p_accuracyEqual);
}

template< class ClassFunc1D>
void testAmericanGlobal(const int &p_nDim, const int &p_nbSimul, const int &p_degree, const double &p_referenceValue, const double &p_accuracyEqual)
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
    GlobalRegression<ClassFunc1D> regressor(p_degree, p_nDim);
    // Bermudean value
    double value = resolutionAmericanRegression(simulator, payoff, regressor);
    BOOST_CHECK_CLOSE(value, p_referenceValue, p_accuracyEqual);
}

template< class LocalRegressor   >
void testAmericanSameSizeLocal(const int &p_nDim, const int &p_nbSimul, const int &p_nMesh, const double &p_referenceValue, const double &p_accuracyEqual)
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
    // create the meshing
    double xmin = simulator.getParticles().minCoeff() - libflow::tiny;
    double xmax = simulator.getParticles().maxCoeff() + libflow::tiny;
    // mesh
    ArrayXd lowVal  = ArrayXd::Constant(p_nDim, xmin);
    ArrayXd step = ArrayXd::Constant(p_nDim, (xmax - xmin) / p_nMesh);
    ArrayXi nbMesh = ArrayXi::Constant(p_nDim, p_nMesh);
    // regressor
    LocalRegressor regressor(lowVal, step, nbMesh);
    // Bermudean value
    double value = resolutionAmericanRegression(simulator, payoff, regressor);
    BOOST_CHECK_CLOSE(value, p_referenceValue, p_accuracyEqual);
}

void testAmericanGridKernel(const bool &p_bLin, const int &p_nDim, const int &p_nbSimul, const double &p_coeffBandwith, const double &p_referenceValue, const double &p_accuracyEqual)
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
    LocalGridKernelRegression regressor(p_coeffBandwith, 1., p_bLin);
    // Bermudean value
    double value = resolutionAmericanRegression(simulator, payoff, regressor);
    BOOST_CHECK_CLOSE(value, p_referenceValue, p_accuracyEqual);
}



// Bouchard Warin test cases
// "Monte Carlo Valuation of American options: facts and new algorithm "
BOOST_AUTO_TEST_CASE(testAmericanLinearBasket1D)
{
    // dimension
    int nDim = 1 ;
    int nbSimul = 500000;
    int nbMesh = 16;
    double referenceValue = 0.06031;
    double accuracyEqual = 0.5;
    testAmericanLocal<LocalLinearRegression>(nDim, nbSimul, nbMesh, referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanConstBasket1D)
{
    // dimension
    int nDim = 1 ;
    int nbSimul = 500000;
    int nbMesh = 60;
    double referenceValue = 0.06031;
    double accuracyEqual = 2.;
    testAmericanLocal<LocalConstRegression>(nDim, nbSimul, nbMesh, referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanSameSizeLinearBasket1D)
{
    // dimension
    int nDim = 1 ;
    int nbSimul = 1000000;
    int nbMesh = 16;
    double referenceValue = 0.06031;
    double accuracyEqual = 0.5;
    testAmericanSameSizeLocal<LocalSameSizeLinearRegression>(nDim, nbSimul, nbMesh, referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanSameSizeConstBasket1D)
{
    // dimension
    int nDim = 1 ;
    int nbSimul = 500000;
    int nbMesh = 100;
    double referenceValue = 0.06031;
    double accuracyEqual = 2.;
    testAmericanSameSizeLocal<LocalSameSizeConstRegression>(nDim, nbSimul, nbMesh, referenceValue, accuracyEqual);
}


BOOST_AUTO_TEST_CASE(testAmericanGlobalBasket1D)
{
    // dimension
    int nDim = 1 ;
    int nbSimul = 100000;
    int degree = 3;
    double referenceValue = 0.06031;
    double accuracyEqual = 1.5;
    testAmericanGlobal<Hermite>(nDim, nbSimul, degree, referenceValue, accuracyEqual);
    testAmericanGlobal<Canonical>(nDim, nbSimul, degree, referenceValue, accuracyEqual);
    testAmericanGlobal<Tchebychev>(nDim, nbSimul, degree, referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanLinearBasket2D)
{
    // dimension
    int nDim = 2 ;
    int nbSimul = 1000000;
    int nbMesh = 16;
    double referenceValue = 0.03882;
    double accuracyEqual = 0.4;
    testAmericanLocal<LocalLinearRegression>(nDim, nbSimul, nbMesh, referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanConstBasket2D)
{
    // dimension
    int nDim = 2 ;
    int nbSimul = 1000000;
    int nbMesh = 40;
    double referenceValue = 0.03882;
    double accuracyEqual = 1.;
    testAmericanLocal<LocalConstRegression>(nDim, nbSimul, nbMesh, referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanSameSizeLinearBasket2D)
{
    // dimension
    int nDim = 2 ;
    int nbSimul = 1000000;
    int nbMesh = 30;
    double referenceValue = 0.03882;
    double accuracyEqual = 0.4;
    testAmericanSameSizeLocal<LocalSameSizeLinearRegression>(nDim, nbSimul, nbMesh, referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanSameSizeConstBasket2D)
{
    // dimension
    int nDim = 2 ;
    int nbSimul = 1000000;
    int nbMesh = 100;
    double referenceValue = 0.03882;
    double accuracyEqual = 2.;
    testAmericanSameSizeLocal<LocalSameSizeConstRegression>(nDim, nbSimul, nbMesh, referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanGlobalBasket2D)
{
    // dimension
    int nDim = 2;
    int nbSimul = 500000;
    int degree = 3;
    double referenceValue =  0.03882;
    double accuracyEqual = 1.5;
    testAmericanGlobal<Hermite>(nDim, nbSimul, degree, referenceValue, accuracyEqual);
    testAmericanGlobal<Canonical>(nDim, nbSimul, degree, referenceValue, accuracyEqual);
    testAmericanGlobal<Tchebychev>(nDim, nbSimul, degree, referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanBasket3D)
{
    // dimension
    int nDim = 3 ;
    int nbSimul = 2000000;
    int nbMesh = 10;
    double referenceValue = 0.02947;
    double accuracyEqual = 0.5;
    testAmericanLocal<LocalLinearRegression>(nDim, nbSimul, nbMesh, referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanGlobalBasket3D)
{
    // dimension
    int nDim = 3;
    int nbSimul = 500000;
    int degree = 3;
    double referenceValue =  0.02947;
    double accuracyEqual = 1.5;
    testAmericanGlobal<Hermite>(nDim, nbSimul, degree, referenceValue, accuracyEqual);
    testAmericanGlobal<Canonical>(nDim, nbSimul, degree, referenceValue, accuracyEqual);
    testAmericanGlobal<Tchebychev>(nDim, nbSimul, degree, referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanBasket4D)
{
    // dimension
    int nDim = 4 ;
    int nbSimul = 2000000;
    int nbMesh = 6;
    double referenceValue = 0.02404;
    double accuracyEqual = 1.;
    testAmericanLocal<LocalLinearRegression>(nDim, nbSimul, nbMesh, referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanGridKernelConstBasket1D)
{
    // dimension
    int nDim = 1 ;
    int nbSimul = 10000;
    double referenceValue = 0.06031;
    double coeffBandwith = 0.1;
    double accuracyEqual = 1.5;
    testAmericanGridKernel(false, nDim, nbSimul, coeffBandwith, referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanGridKernelLinearBasket1D)
{
    // dimension
    int nDim = 1 ;
    int nbSimul = 10000;
    double referenceValue = 0.06031;
    double coeffBandwith = 0.1;
    double accuracyEqual = 1.5;
    testAmericanGridKernel(true, nDim, nbSimul, coeffBandwith, referenceValue, accuracyEqual);
}


BOOST_AUTO_TEST_CASE(testAmericanGridKernelConstBasket2D)
{
    // dimension
    int nDim = 2 ;
    int nbSimul = 10000;
    double referenceValue =  0.03882;
    double coeffBandwith = 0.15;
    double accuracyEqual = 1.5;
    testAmericanGridKernel(false, nDim, nbSimul, coeffBandwith, referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanGridKernelLinearBasket2D)
{
    // dimension
    int nDim = 2 ;
    int nbSimul = 10000;
    double referenceValue =  0.03882;
    double coeffBandwith = 0.15;
    double accuracyEqual = 1.5;
    testAmericanGridKernel(true, nDim, nbSimul, coeffBandwith, referenceValue, accuracyEqual);
}

BOOST_AUTO_TEST_CASE(testAmericanGridKernelLinearBasket3D)
{
    // dimension
    int nDim = 3 ;
    int nbSimul = 200000;
    double referenceValue =  0.02947;
    double coeffBandwith = 0.1;
    double accuracyEqual = 3.;
    testAmericanGridKernel(true, nDim, nbSimul, coeffBandwith, referenceValue, accuracyEqual);
}


