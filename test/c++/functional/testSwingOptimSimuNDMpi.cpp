
#define BOOST_TEST_DYN_LINK
#include <functional>
#include <boost/tuple/tuple.hpp>
#include <boost/test/unit_test.hpp>
#include <array>
#include <Eigen/Dense>
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/regression/LocalLinearRegressionGeners.h"
#include "test/c++/tools/dp/FinalValueFictitiousFunction.h"
#include "test/c++/tools/dp/OptimizeFictitiousSwing.h"
#include "test/c++/tools/dp/DynamicProgrammingByRegressionDist.h"
#include "test/c++/tools/dp/SimulateRegressionDist.h"
#include "test/c++/tools/BasketOptions.h"
#include "test/c++/tools/simulators/BlackScholesSimulator.h"


/** \file testSwingOptimSimuND.cpp
 *  Simple test for swing optimization and simulation in more than one stock
 * \author Xavier Warin
 */

using namespace std;
using namespace Eigen ;
using namespace libflow;

double accuracyClose = 1.5;


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

#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif


/// \param p_ndim        dimension of the swing
/// \param p_bOneFile    Do we use one unique file for continuation values
void testSwingND(const  int   &p_ndim, bool p_bOneFile)
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
    int N = 3 ; // 5 exercise dates
    double strike = 1.;
    int nbSimul = 80000;
    int nMesh = 8;
    // payoff
    BasketCall  payoff(strike);
    // mesh
    ArrayXi nbMesh = ArrayXi::Constant(1, nMesh);

    // simulator
    shared_ptr<BlackScholesSimulator>  simulator(new BlackScholesSimulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false));
    // grid
    ArrayXd lowValues = ArrayXd::Constant(p_ndim, 0.);
    ArrayXd step = ArrayXd::Constant(p_ndim, 1.);
    // the stock is discretized with values from 0 to N included
    ArrayXi nbStep = ArrayXi::Constant(p_ndim, N);
    shared_ptr<RegularSpaceGrid> grid = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
    // final value
    function<double(const int &, const ArrayXd &, const ArrayXd &)>   vFunction = FinalValueFictitiousFunction<BasketCall>(payoff, N);
    // optimizer
    shared_ptr< OptimizeFictitiousSwing<BasketCall, BlackScholesSimulator> >optimizer = make_shared< OptimizeFictitiousSwing<BasketCall, BlackScholesSimulator> >(payoff, N, p_ndim);
    // initial values
    ArrayXd initialStock = ArrayXd::Constant(p_ndim, 0.);
    int initialRegime = 0;
    string fileToDump = "CondExpOptSIMNDPI" + to_string(world.size());
    // regressor
    shared_ptr< BaseRegression > regressor(new LocalLinearRegression(nbMesh));
    // link the simulations to the optimizer
    optimizer->setSimulator(simulator);
    double valueOptimDist = DynamicProgrammingByRegressionDist(grid, optimizer, regressor, vFunction, initialStock, initialRegime, fileToDump, p_bOneFile, world);
    // simulation value
    shared_ptr<BlackScholesSimulator>  simulatorForward(new BlackScholesSimulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, true));
    optimizer->setSimulator(simulatorForward);
    double valSimuDist = SimulateRegressionDist(grid, optimizer, vFunction, initialStock, initialRegime,
                         fileToDump, p_bOneFile, world) ;
    if (world.rank() == 0)
        BOOST_CHECK_CLOSE(valueOptimDist, valSimuDist, accuracyClose);

}

BOOST_AUTO_TEST_CASE(testSwingOptionOptim2DSimuDistOneFile)
{
    int ndim = 2 ;
    bool bOneFile = true ;
    testSwingND(ndim, bOneFile);
}
BOOST_AUTO_TEST_CASE(testSwingOptionOptim2DSimuDistMultipleFile)
{
    int ndim = 2 ;
    bool bOneFile = false ;
    testSwingND(ndim, bOneFile);
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
