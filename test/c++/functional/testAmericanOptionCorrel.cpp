
#define BOOST_TEST_MODULE testRegressionForAmericanOptionsWithCorrelation
#define BOOST_TEST_DYN_LINK
#include <memory>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "test/c++/tools/simulators/BlackScholesSimulator.h"
#include "test/c++/tools/BasketOptions.h"
#include "libflow/core/utils/constant.h"
#include "libflow/regression/LocalLinearRegression.h"
#include "libflow/regression/LocalConstRegression.h"
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

/// \brief test correlation and rotation achived on data

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
double testAmericanLocal(const bool &p_bRotation, const double &p_rho, const int &p_nbSimul, const ArrayXi  &p_nbMesh)
{
    int nDim = 2;
    VectorXd initialValues = ArrayXd::Constant(nDim, 1.);
    VectorXd sigma  = ArrayXd::Constant(nDim, 0.2);
    VectorXd mu  = ArrayXd::Constant(nDim, 0.05);
    MatrixXd corr = MatrixXd::Identity(nDim, nDim);
    corr(0, 1) = p_rho;
    corr(1, 0) = p_rho;
    double T = 1. ;
    int nDate = 10 ;
    double strike = 1.;
    // simulator
    BlackScholesSimulator simulator(initialValues, sigma, mu, corr, T, nDate, p_nbSimul, false);
    // payoff
    BasketPut payoff(strike);
    // regressor
    LocalRegressor regressor(p_nbMesh, p_bRotation);
    // Bermudean value
    return resolutionAmericanRegression(simulator, payoff, regressor);
}


template< class ClassFunc1D>
double  testAmericanGlobal(const bool &p_bRotation, const double &p_rho, const int &p_nbSimul, const int &p_degree)
{
    int nDim = 2;
    VectorXd initialValues = ArrayXd::Constant(nDim, 1.);
    VectorXd sigma  = ArrayXd::Constant(nDim, 0.2);
    VectorXd mu  = ArrayXd::Constant(nDim, 0.05);
    MatrixXd corr = MatrixXd::Identity(nDim, nDim);
    corr(0, 1) = p_rho;
    corr(1, 0) = p_rho;
    double T = 1. ;
    int nDate = 10 ;
    double strike = 1.;
    // simulator
    BlackScholesSimulator simulator(initialValues, sigma, mu, corr, T, nDate, p_nbSimul, false);
    // payoff
    BasketPut payoff(strike);
    // regressor
    GlobalRegression<ClassFunc1D> regressor(p_degree, nDim, p_bRotation);
    // Bermudean value
    double value = resolutionAmericanRegression(simulator, payoff, regressor);
    return value;
}


BOOST_AUTO_TEST_CASE(testAmericCorrel)
{
    // dimension
    int nbSimul = 500000;
    int nbMesh = 16;
    double accuracyEqual = 0.4;
    double rho = 0.9;
    ArrayXi meshArray(2);
    meshArray(0) = nbMesh;
    meshArray(1) = nbMesh;
    double  valAmerLocLinNoRot = testAmericanLocal<LocalLinearRegression>(false, rho, nbSimul, meshArray);
    meshArray(0) = nbMesh;
    meshArray(1) = nbMesh / 2;
    double  valAmerLocLinRot = testAmericanLocal<LocalLinearRegression>(true, rho, nbSimul, meshArray);
    cout << " valAmerLocLinNoRot " << valAmerLocLinNoRot << " valAmerLocLinRot " << valAmerLocLinRot << endl ;
    BOOST_CHECK_CLOSE(valAmerLocLinNoRot, valAmerLocLinRot, accuracyEqual);
    nbMesh = 40;
    meshArray(0) = nbMesh;
    meshArray(1) = nbMesh;
    double  valAmerLocConstNoRot = testAmericanLocal<LocalConstRegression>(false, rho,  nbSimul, meshArray);
    meshArray(0) = nbMesh;
    meshArray(1) = nbMesh / 2;
    double  valAmerLocConstRot = testAmericanLocal<LocalConstRegression>(true, rho,  nbSimul, meshArray);
    cout << " valAmerLocConstNoRot " << valAmerLocConstNoRot << " valAmerLocConstRot " << valAmerLocConstRot << endl ;
    BOOST_CHECK_CLOSE(valAmerLocConstNoRot, valAmerLocConstRot, accuracyEqual);
    int degree = 4;
    double valGlobNoRot = testAmericanGlobal<Hermite>(false, rho, nbSimul, degree);
    double valGlobRot = testAmericanGlobal<Hermite>(true, rho, nbSimul, degree);
    cout << " valGlobNoRot " << valGlobNoRot << " valGlobRot " << valGlobRot << endl ;
    BOOST_CHECK_CLOSE(valGlobNoRot, valGlobRot, accuracyEqual);
}

