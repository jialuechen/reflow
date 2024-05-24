

#ifndef USE_MPI
#define BOOST_TEST_MODULE testLake
#endif
#define BOOST_TEST_DYN_LINK
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#define _USE_MATH_DEFINES
#include <math.h>
#include <memory>
#include <functional>
#include <boost/test/unit_test.hpp>
#include <boost/timer/timer.hpp>
#include <Eigen/Dense>
#include "reflow/core/grids/OneDimRegularSpaceGrid.h"
#include "reflow/core/grids/OneDimData.h"
#include "reflow/regression/LocalLinearRegression.h"
#include "reflow/core/grids/RegularLegendreGridGeners.h"
#include "reflow/regression/LocalLinearRegressionGeners.h"
#include "test/c++/tools/simulators/AR1Simulator.h"
#include "test/c++/tools/dp/DynamicProgrammingByRegression.h"
#include "test/c++/tools/dp/SimulateRegression.h"
#include "test/c++/tools/dp/SimulateRegressionControl.h"
#include "test/c++/tools/dp/OptimizeLake.h"

using namespace std;
using namespace Eigen ;
using namespace reflow;


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

double accuracyClose =  1.5;

class ZeroFunction
{
public:
    ZeroFunction() {}
    double operator()(const int &, const ArrayXd &, const ArrayXd &) const
    {
        return 0. ;
    }
};




/// \brief valorization of a given Lake on a  grid
///        Gain are proportional to what is withdrawn from the storage
///        Only inflows are stochastic
/// \param p_grid             the grid
/// \param p_maxLevelStorage  maximum level
/// \param p_mesh             number of mesh
/// \param p_bCheckClose      Do we check if optimisation and simulations are close
void testLake(shared_ptr< FullGrid> &p_grid, const double &p_maxLevelStorage, const int &p_mesh, const bool   &p_bCheckClose)
{
#ifdef USE_MPI
    boost::mpi::communicator world;
#endif
    // storage
    /////////
    double withdrawalRateStorage = 1000;

    double maturity = 1.;
    size_t nstep = 10;

    // number of simulations
    size_t nbsimulOpt = 8000;

    // inflow model
    double D0 = 50. ; // initial inflow
    double m = D0 ; // average inflow
    double sig = 5. ; // volatility
    double mr  = 5. ; // mean reverting
    // a backward simulator
    ///////////////////////
    bool bForward = false;
    shared_ptr< AR1Simulator> backSimulator = make_shared<AR1Simulator> (D0, m, sig, mr, maturity, nstep, nbsimulOpt, bForward);
    // optimizer
    ///////////
    shared_ptr< OptimizeLake<AR1Simulator> > storage =  make_shared< OptimizeLake<AR1Simulator> >(withdrawalRateStorage);
    // regressor
    ///////////
    ArrayXi nbMesh ;
    if (p_mesh > 0)
        nbMesh = ArrayXi::Constant(1, p_mesh);
    shared_ptr< LocalLinearRegression > regressor =  make_shared< LocalLinearRegression >(nbMesh);
    // final value
    function<double(const int &, const ArrayXd &, const ArrayXd &)>  vFunction = ZeroFunction();

    // initial values
    ArrayXd initialStock = ArrayXd::Constant(1, p_maxLevelStorage);
    int initialRegime = 0; // only one regime

    /// Optimize
#ifdef USE_MPI
    string fileToDump = "CondExpLake" + to_string(world.size());
#else
    string fileToDump = "CondExpLake";
#endif

    double valueOptim ;
    {
        // link the simulations to the optimizer
        storage->setSimulator(backSimulator);
        boost::timer::auto_cpu_timer t;

        valueOptim =  DynamicProgrammingByRegression(p_grid, storage, regressor, vFunction, initialStock, initialRegime, fileToDump
#ifdef USE_MPI
                      , world
#endif
                                                    );
    }

    // a forward simulator
    ///////////////////////
    int nbsimulSim = 8000;
    bForward = true;
    shared_ptr< AR1Simulator> forSimulator = make_shared<AR1Simulator> (D0, m, sig, mr, maturity, nstep, nbsimulSim, bForward);
    double valSimu ;
    {
        // link the simulations to the optimizer
        storage->setSimulator(forSimulator);
        boost::timer::auto_cpu_timer t;
        valSimu = SimulateRegression(p_grid, storage, vFunction, initialStock, initialRegime, fileToDump
#ifdef USE_MPI
                                     , world
#endif
                                    ) ;

    }
    cout << " valSimu  " << valSimu << " valueOptim " << valueOptim << endl ;
    if (p_bCheckClose)
        BOOST_CHECK_CLOSE(valueOptim, valSimu, accuracyClose);

    // a forward simulator
    ///////////////////////
    shared_ptr< AR1Simulator> forSimulator2 = make_shared<AR1Simulator> (D0, m, sig, mr, maturity, nstep, nbsimulSim, bForward);
    double valSimu2 ;
    {
        // link the simulations to the optimizer
        storage->setSimulator(forSimulator2);
        boost::timer::auto_cpu_timer t;
        valSimu2 = SimulateRegressionControl(p_grid, storage, vFunction, initialStock, initialRegime, fileToDump
#ifdef USE_MPI
                                             , world
#endif
                                            ) ;

    }
    cout << " valSimu2  " << valSimu2 << " valueOptim " << valueOptim << endl ;
    if (p_bCheckClose)
        BOOST_CHECK_CLOSE(valueOptim, valSimu2, accuracyClose);
}

// linear interpolation
BOOST_AUTO_TEST_CASE(testSimpleStorageLegendreLinear)
{
    // storage
    /////////
    double maxLevelStorage  = 5000;
    // grid
    //////
    int nGrid = 10;
    ArrayXd lowValues = ArrayXd::Constant(1, 0.);
    ArrayXd step = ArrayXd::Constant(1, maxLevelStorage / nGrid);
    ArrayXi nbStep = ArrayXi::Constant(1, nGrid);
    ArrayXi poly = ArrayXi::Constant(1, 1);
    shared_ptr<FullGrid> grid = make_shared<RegularLegendreGrid>(lowValues, step, nbStep, poly);
    int nbmesh = 4 ;
    testLake(grid, maxLevelStorage, nbmesh, true);
}

// quadratic interpolation on the basis functions
BOOST_AUTO_TEST_CASE(testSimpleStorageLegendreQuad)
{
    // storage
    /////////
    double maxLevelStorage  = 5000;
    // grid
    //////
    int nGrid = 5;
    ArrayXd lowValues = ArrayXd::Constant(1, 0.);
    ArrayXd step = ArrayXd::Constant(1, maxLevelStorage / nGrid);
    ArrayXi nbStep = ArrayXi::Constant(1, nGrid);
    ArrayXi poly = ArrayXi::Constant(1, 2);
    shared_ptr<FullGrid> grid = make_shared<RegularLegendreGrid>(lowValues, step, nbStep, poly);
    int nbmesh = 4 ;
    testLake(grid, maxLevelStorage, nbmesh, true);
}

// forget the AR1 model and suppose that inflows are iid
BOOST_AUTO_TEST_CASE(testSimpleStorageAverageInflows)
{
    // storage
    /////////
    double maxLevelStorage  = 5000;
    // grid
    ////////
    int nGrid = 10;
    ArrayXd lowValues = ArrayXd::Constant(1, 0.);
    ArrayXd step = ArrayXd::Constant(1, maxLevelStorage / nGrid);
    ArrayXi nbStep = ArrayXi::Constant(1, nGrid);
    ArrayXi poly = ArrayXi::Constant(1, 1);
    shared_ptr<FullGrid> grid = make_shared<RegularLegendreGrid>(lowValues, step, nbStep, poly);
    int nbmesh = 0 ;
    testLake(grid, maxLevelStorage, nbmesh, false);
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
