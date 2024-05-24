
#ifndef USE_MPI
#define BOOST_TEST_MODULE testSwingOptimSimuWithHedge
#endif
#define BOOST_TEST_DYN_LINK
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <functional>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "reflow/regression/LocalLinearRegression.h"
#include "reflow/core/grids/RegularSpaceGridGeners.h"
#include "reflow/regression/LocalLinearRegressionGeners.h"
#include "test/c++/tools/dp/FinalValueFunction.h"
#include "test/c++/tools/dp/OptimizeSwingWithHedge.h"
#include "test/c++/tools/dp/DynamicProgrammingByRegression.h"
#include "test/c++/tools/dp/SimulateRegressionWithHedge.h"
#include "test/c++/tools/simulators/BlackScholesSimulator.h"
#include "test/c++/tools/BasketOptions.h"


/** \file testSwingOptimSimuWithHedge.cpp
 *  Simple test for swing optimization and simulation add dynamic hedging position during simulations
 * \author Xavier Warin
 */

using namespace std;
using namespace Eigen ;
using namespace reflow;


#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif


double accuracyClose = 0.7;
double accuracyHedge = 0.001;
double ratioHedgeReduction = 0.3;

#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif


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


BOOST_AUTO_TEST_CASE(testSwingOptionOptimSimuWithHedge)
{
#ifdef USE_MPI
    boost::mpi::communicator world;
#endif
    VectorXd initialValues = ArrayXd::Constant(1, 1.);
    VectorXd sigma  = ArrayXd::Constant(1, 0.2);
    VectorXd mu  = ArrayXd::Constant(1, 0.05);
    MatrixXd corr = MatrixXd::Ones(1, 1);
    // number of step
    int nStep = 20;
    // exercise date
    double T = 1. ;
    ArrayXd dates = ArrayXd::LinSpaced(nStep + 1, 0., T);
    int N = 3 ; // number of exercises
    double strike = 1.;
    int nbSimul = 100000;
    int nMesh = 8;
    // payoff
    BasketCall  payoff(strike);
    // mesh
    ArrayXi nbMesh = ArrayXi::Constant(1, nMesh);
    // grid
    ArrayXd lowValues = ArrayXd::Constant(1, 0.);
    ArrayXd step = ArrayXd::Constant(1, 1.);
    ArrayXi nbStep = ArrayXi::Constant(1, N);
    shared_ptr<FullGrid> grid = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
    // final value
    function<double(const int &, const ArrayXd &, const ArrayXd &)>  vFunction = FinalValueFunction<BasketCall>(payoff, N);
    shared_ptr< OptimizeSwingWithHedge<BasketCall, BlackScholesSimulator > >optimizer = make_shared< OptimizeSwingWithHedge<BasketCall, BlackScholesSimulator > >(payoff, N);
    // initial values
    ArrayXd initialStock = ArrayXd::Constant(1, 0.);
    int initialRegime = 0;
    string fileToDump = "CondExpWH";
    // simulator
    int nbSimulSim = 100000;
    shared_ptr<BlackScholesSimulator> simulatorBack(new BlackScholesSimulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimulSim, false));
    // regressor
    shared_ptr< LocalLinearRegression > regressor(new LocalLinearRegression(nbMesh));
    // link the simulations to the optimizer
    optimizer->setSimulator(simulatorBack);
    // Bermudean value
    double valueOptim =  DynamicProgrammingByRegression(grid, optimizer, regressor, vFunction, initialStock, initialRegime, fileToDump
#ifdef USE_MPI
                         , world
#endif

                                                       );
    // simulation value
    shared_ptr<BlackScholesSimulator>  simulatorForward(new BlackScholesSimulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, true));
    optimizer->setSimulator(simulatorForward);
    Array4d  valAndHedge = SimulateRegressionWithHedge(grid, optimizer, vFunction, initialStock, initialRegime, fileToDump
#ifdef USE_MPI
                           , world
#endif

                                                      ) ;
    // check value
    BOOST_CHECK_CLOSE(valueOptim, valAndHedge(0), accuracyClose);
    // check average hedge near 0.
    BOOST_REQUIRE(fabs(valAndHedge(1)) < accuracyHedge);
    // check std diminution
    BOOST_REQUIRE(valAndHedge(3) < ratioHedgeReduction * valAndHedge(2));
}


#ifdef USE_MPI
// (empty) Initialization function. Can't use testing tools here.
bool init_function()
{
    return true;
}

int main(int argc, char *argv[])
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    boost::mpi::environment env(argc, argv);
    return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}

#endif
