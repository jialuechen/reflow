
#define BOOST_TEST_DYN_LINK
#include <functional>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include <boost/mpi.hpp>
#include "reflow/regression/LocalLinearRegressionGeners.h"
#include "reflow/core/grids/RegularSpaceGridGeners.h"
#include "test/c++/tools/simulators/BlackScholesSimulator.h"
#include "test/c++/tools/dp/FinalValueFunction.h"
#include "test/c++/tools/dp/OptimizeSwing.h"
#include "test/c++/tools/dp/DynamicProgrammingByRegressionDist.h"
#include "test/c++/tools/dp/SimulateRegressionDist.h"
#include "test/c++/tools/BasketOptions.h"


/** \file testSwingOptimSimuMpi.cpp
 *  Simple test for swing optimization and simulation with MPI
 * \author Xavier Warin
 */

using namespace std;
using namespace Eigen ;
using namespace reflow;

double accuracyClose = 0.7;
double accuracyEqual = 1e-10;


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
std::string normalize_test_case_name(const_string name)
{
    return (name[0] == '&' ? std::string(name.begin() + 1, name.size() - 1) : std::string(name.begin(), name.size()));
}
}
}
}


BOOST_AUTO_TEST_CASE(testSwingOptionOptimSimuDist)
{
    boost::mpi::communicator world;
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
    shared_ptr<RegularSpaceGrid> grid = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
    // final value
    std::function<double(const int &, const ArrayXd &, const ArrayXd &)>   vFunction = FinalValueFunction<BasketCall>(payoff, N);
    // initial values
    ArrayXd initialStock = ArrayXd::Constant(1, 0.);
    int initialRegime = 0;
    string fileToDump = "CondExpOSimMPi" + to_string(world.size());
    // one file version
    double valSimuOneFile ;
    {
        shared_ptr< OptimizeSwing<BasketCall, BlackScholesSimulator> >optimizer = make_shared< OptimizeSwing<BasketCall, BlackScholesSimulator> >(payoff, N);
        bool bOneFile = true;
        // simulator
        shared_ptr<BlackScholesSimulator> simulatorBack(new BlackScholesSimulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false));
        // regressor
        shared_ptr< BaseRegression > regressor(new LocalLinearRegression(nbMesh));
        // Bermudean value
        // link the simulations to the optimizer
        optimizer->setSimulator(simulatorBack);
        double valueOptimDist =  DynamicProgrammingByRegressionDist(grid, optimizer, regressor, vFunction, initialStock, initialRegime, fileToDump, bOneFile, world);
        world.barrier();
        // simulation value
        shared_ptr<BlackScholesSimulator>  simulatorForward(new BlackScholesSimulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, true));
        optimizer->setSimulator(simulatorForward);
        valSimuOneFile = SimulateRegressionDist(grid, optimizer, vFunction, initialStock, initialRegime, fileToDump, bOneFile, world) ;
        if (world.rank() == 0)
            BOOST_CHECK_CLOSE(valueOptimDist, valSimuOneFile, accuracyClose);
    }
    {
        shared_ptr< OptimizeSwing<BasketCall, BlackScholesSimulator> >optimizer = make_shared< OptimizeSwing<BasketCall, BlackScholesSimulator> >(payoff, N);
        bool bOneFile = false;
        // simulator
        shared_ptr<BlackScholesSimulator> simulatorBack(new BlackScholesSimulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false));
        // regressor
        shared_ptr< BaseRegression > regressor(new LocalLinearRegression(nbMesh));
        // link the simulations to the optimizer
        optimizer->setSimulator(simulatorBack);
        // Bermudean value
        double valueOptimDist =  DynamicProgrammingByRegressionDist(grid, optimizer, regressor, vFunction, initialStock, initialRegime, fileToDump, bOneFile, world);
        if (world.rank() == 0)
            cout << " Value in optimization " << valueOptimDist << endl ;
        world.barrier();
        // simulation value
        shared_ptr<BlackScholesSimulator>  simulatorForward(new BlackScholesSimulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, true));
        optimizer->setSimulator(simulatorForward);
        double valSimuMultipleFiles = SimulateRegressionDist(grid, optimizer, vFunction, initialStock, initialRegime, fileToDump, bOneFile, world) ;
        BOOST_CHECK_CLOSE(valSimuOneFile, valSimuMultipleFiles, accuracyEqual);
    }
}

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
