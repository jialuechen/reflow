
#ifndef USE_MPI
#define BOOST_TEST_MODULE testSwingOptimSimuND
#endif
#define BOOST_TEST_DYN_LINK
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <functional>
#include <array>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "reflow/core/grids/RegularSpaceGridGeners.h"
#include "reflow/regression/LocalLinearRegressionGeners.h"
#include "test/c++/tools/dp/FinalValueFictitiousFunction.h"
#include "test/c++/tools/dp/OptimizeFictitiousSwing.h"
#include "test/c++/tools/dp/DynamicProgrammingByRegression.h"
#include "test/c++/tools/dp/SimulateRegression.h"
#include "test/c++/tools/BasketOptions.h"
#include "test/c++/tools/simulators/BlackScholesSimulator.h"


/** \file testSwingOptimSimuND.cpp
 *  Simple test for swing optimization and simulation in more than one stock
 * \author Xavier Warin
 */

using namespace std;
using namespace Eigen ;
using namespace reflow;



double accuracyClose = 1.5;


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

BOOST_AUTO_TEST_CASE(testSwingOptionOptimNDSimu)
{
#ifdef USE_MPI
    boost::mpi::communicator world;
#endif

    int ndim = 2 ;
    VectorXd initialValues = ArrayXd::Constant(1, 1.);
    VectorXd sigma  = ArrayXd::Constant(1, 0.2);
    VectorXd mu  = ArrayXd::Constant(1, 0.05);
    MatrixXd corr = MatrixXd::Ones(1, 1);
    // number of step
    int nStep = 20;
    // exercise date
    double T = 1. ;
    ArrayXd dates = ArrayXd::LinSpaced(nStep + 1, 0., T);
    int N = 3 ; // 5 exercise dates
    double strike = 1.;
    int nbSimul = 80000;
    int nMesh = 8;
    // payoff
    BasketCall  payoff(strike);
    // mesh
    ArrayXi nbMesh = ArrayXi::Constant(1, nMesh);

    // simulator
    shared_ptr<BlackScholesSimulator> simulatorBack(new BlackScholesSimulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false));
    // grid
    ArrayXd lowValues = ArrayXd::Constant(ndim, 0.);
    ArrayXd step = ArrayXd::Constant(ndim, 1.);
    // the stock is discretized with values from 0 to N included
    ArrayXi nbStep = ArrayXi::Constant(ndim, N);
    shared_ptr<FullGrid> grid = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
    // final value
    function<double(const int &, const ArrayXd &, const ArrayXd &)>   vFunction = FinalValueFictitiousFunction<BasketCall>(payoff, N);
    // optimizer
    shared_ptr< OptimizeFictitiousSwing<BasketCall, BlackScholesSimulator> >optimizer = make_shared< OptimizeFictitiousSwing<BasketCall, BlackScholesSimulator> >(payoff, N, ndim);
    // initial values
    ArrayXd initialStock = ArrayXd::Constant(ndim, 0.);
    int initialRegime = 0;
    string fileToDump = "CondExpOptND";
    // regressor
    shared_ptr< LocalLinearRegression > regressor(new LocalLinearRegression(nbMesh));
    // link the simulations to the optimizer
    optimizer->setSimulator(simulatorBack);
    double valueOptim = DynamicProgrammingByRegression(grid, optimizer, regressor, vFunction, initialStock, initialRegime, fileToDump
#ifdef USE_MPI
                        , world
#endif
                                                      );
    // simulation value
    int nbSimulSim = 80000;
    shared_ptr<BlackScholesSimulator>  simulatorForward(new BlackScholesSimulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimulSim, true));
    optimizer->setSimulator(simulatorForward);
    double valSimu = SimulateRegression(grid, optimizer, vFunction, initialStock, initialRegime, fileToDump
#ifdef USE_MPI
                                        , world
#endif

                                       ) ;

    BOOST_CHECK_CLOSE(valueOptim, valSimu, accuracyClose);

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
