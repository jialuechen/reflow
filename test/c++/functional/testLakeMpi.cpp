// Copyright (C) 2016 Fime

#define BOOST_TEST_DYN_LINK
#include <math.h>
#include <memory>
#include <functional>
#include <boost/mpi.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/timer/timer.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/OneDimRegularSpaceGrid.h"
#include "libflow/core/grids/OneDimData.h"
#include "libflow/regression/LocalLinearRegression.h"
#include "libflow/core/grids/RegularLegendreGridGeners.h"
#include "libflow/regression/LocalLinearRegressionGeners.h"
#include "test/c++/tools/simulators/AR1Simulator.h"
#include "test/c++/tools/dp/DynamicProgrammingByRegressionDist.h"
#include "test/c++/tools/dp/SimulateRegressionDist.h"
#include "test/c++/tools/dp/SimulateRegressionControlDist.h"
#include "test/c++/tools/dp/OptimizeLake.h"

using namespace std;
using namespace Eigen ;
using namespace libflow;

double accuracyClose =  1.5;


/// For Clang 3.6 (and above ?) to be compatible GCC 5.1 and above
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

class ZeroFunction
{
public:
    ZeroFunction() {}
    double operator()(const int &, const ArrayXd &, const ArrayXd &) const
    {
        return 0. ;
    }
};




#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif

/// \brief valorization of a given Lake on a  grid
///        Gain are proportional to what is withdrawn from the storage
///        Only inflows are stochastic
/// \param p_grid             the grid
/// \param p_maxLevelStorage  maximum level
/// \param p_mesh             number of mesh
/// \param p_bCheckClose      Do we check if optimisation and simulations are close
/// \param p_bOneFile         One file to dump control and Bellman values ?
void testLake(shared_ptr< FullGrid> &p_grid, const double &p_maxLevelStorage, const int &p_mesh, const bool   &p_bCheckClose, const bool &p_bOneFile)
{
    boost::mpi::communicator world;
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
    shared_ptr< BaseRegression > regressor =  make_shared< LocalLinearRegression >(nbMesh);
    // final value
    function<double(const int &, const ArrayXd &, const ArrayXd &)>  vFunction = ZeroFunction();

    // initial values
    ArrayXd initialStock = ArrayXd::Constant(1, p_maxLevelStorage);
    int initialRegime = 0; // only one regime
    /// Optimize
    string fileToDump = "CondExpLakeMpi";
    double valueOptimDist ;
    {
        // link the simulations to the optimizer
        storage->setSimulator(backSimulator);
        boost::timer::auto_cpu_timer t;
        valueOptimDist =  DynamicProgrammingByRegressionDist(p_grid, storage, regressor, vFunction, initialStock, initialRegime, fileToDump, p_bOneFile, world);
    }

    world.barrier();

    // a forward simulator
    ///////////////////////
    int nbsimulSim = 8000;
    bForward = true;
    shared_ptr< AR1Simulator> forSimulator = make_shared<AR1Simulator> (D0, m, sig, mr, maturity, nstep, nbsimulSim, bForward);
    double valSimuDist ;
    {
        // link the simulations to the optimizer
        storage->setSimulator(forSimulator);
        boost::timer::auto_cpu_timer t;
        valSimuDist = SimulateRegressionDist(p_grid, storage, vFunction, initialStock, initialRegime, fileToDump, p_bOneFile, world) ;

    }
    if (world.rank() == 0)
    {
        cout << " valSimuDist " << valSimuDist << " valueOptimDist " << valueOptimDist << endl ;
        if (p_bCheckClose)
            BOOST_CHECK_CLOSE(valueOptimDist, valSimuDist, accuracyClose);
    }

    // // a forward simulator
    // ///////////////////////
    shared_ptr< AR1Simulator> forSimulator2 = make_shared<AR1Simulator> (D0, m, sig, mr, maturity, nstep, nbsimulSim, bForward);
    double valSimuDist2 ;
    {
        // link the simulations to the optimizer
        storage->setSimulator(forSimulator2);
        boost::timer::auto_cpu_timer t;
        valSimuDist2 = SimulateRegressionControlDist(p_grid, storage, vFunction, initialStock, initialRegime, fileToDump, p_bOneFile, world) ;

    }
    if (world.rank() == 0)
    {
        cout << " valSimuDist2 " << valSimuDist2 << " valueOptimDist " << valueOptimDist << endl ;
        if (p_bCheckClose)
            BOOST_CHECK_CLOSE(valueOptimDist, valSimuDist2, accuracyClose);
    }
}

// linear interpolation
BOOST_AUTO_TEST_CASE(testSimpleStorageLegendreLinearDist)
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
    testLake(grid, maxLevelStorage, nbmesh, true, true);
}

// linear interpolation
BOOST_AUTO_TEST_CASE(testSimpleStorageLegendreLinearMultipleFilesDist)
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
    testLake(grid, maxLevelStorage, nbmesh, true, false);
}

// quadratic interpolation on the basis functions
BOOST_AUTO_TEST_CASE(testSimpleStorageLegendreQuadDist)
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
    testLake(grid, maxLevelStorage, nbmesh, true, true);
}

// forget the AR1 model and suppose that inflows are iid
BOOST_AUTO_TEST_CASE(testSimpleStorageAverageInflowsDist)
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
    testLake(grid, maxLevelStorage, nbmesh, false, true);
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
