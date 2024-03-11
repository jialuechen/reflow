

#define BOOST_TEST_DYN_LINK
#define _USE_MATH_DEFINES
#include <math.h>
#include <functional>
#include <memory>
#include <boost/test/unit_test.hpp>
#include <boost/timer/timer.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/OneDimRegularSpaceGrid.h"
#include "libflow/core/grids/OneDimData.h"
#include "libflow/regression/LocalLinearRegression.h"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/core/grids/RegularLegendreGridGeners.h"
#include "libflow/regression/LocalLinearRegressionGeners.h"
#include "test/c++/tools/simulators/MeanRevertingSimulator.h"
#include "test/c++/tools/dp/DynamicProgrammingByRegressionVaryingGridsDist.h"
#include "test/c++/tools/dp/SimulateRegressionVaryingGridsDist.h"
#include "test/c++/tools/dp/SimulateRegressionVaryingGridsControlDist.h"
#include "test/c++/tools/dp/OptimizeGasStorage.h"

using namespace std;
using namespace Eigen ;
using namespace libflow;

double accuracyClose = 1.5;
double accuracyEqual = 0.0001;


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

class ZeroFunction
{
public:
    ZeroFunction() {}
    double operator()(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &) const
    {
        return 0. ;
    }
};



#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif




/// \brief valorization of a given gas storage on a  set of grids
///        The characteristics of the cavity change with time.
/// \param p_timeChangeGrid       date for changing grids
/// \param p_grids                grids
/// \param p_maxLevelStorage  maximum level
void testGasStorage(const std::vector<double>    &p_timeChangeGrid,  const std::vector<shared_ptr<FullGrid> >   &p_grids, const double &p_maxLevelStorage)
{
    boost::mpi::communicator world;
    // storage
    /////////
    double injectionRateStorage = 60000;
    double withdrawalRateStorage = 45000;
    double injectionCostStorage = 0.35;
    double withdrawalCostStorage = 0.35;

    double maturity = 1.;
    size_t nstep = 100;
    // define a time grid
    shared_ptr<OneDimRegularSpaceGrid> timeGrid(new OneDimRegularSpaceGrid(0., maturity / nstep, nstep));
    // future values
    shared_ptr<std::vector< double > > futValues(new std::vector<double>(nstep + 1));
    // periodicity factor
    int iPeriod = 52;
    for (size_t i = 0; i < nstep + 1; ++i)
        (*futValues)[i] = 50. + 20 * sin((M_PI * i * iPeriod) / nstep);
    // define the future curve
    shared_ptr<OneDimData<OneDimRegularSpaceGrid, double> > futureGrid(new OneDimData< OneDimRegularSpaceGrid, double> (timeGrid, futValues));
    // one dimensional factors
    int nDim = 1;
    VectorXd sigma = VectorXd::Constant(nDim, 0.94);
    VectorXd mr = VectorXd::Constant(nDim, 0.29);
    // number of simulations
    size_t nbsimulOpt = 20000;

    // no actualization
    double r = 0. ;
    // a backward simulator
    ///////////////////////
    bool bForward = false;
    shared_ptr< MeanRevertingSimulator< OneDimData<OneDimRegularSpaceGrid, double> > > backSimulator(new	  MeanRevertingSimulator< OneDimData<OneDimRegularSpaceGrid, double> > (futureGrid, sigma, mr, r, maturity, nstep, nbsimulOpt, bForward));
    // optimizer
    ///////////
    shared_ptr< OptimizeGasStorage< MeanRevertingSimulator< OneDimData<OneDimRegularSpaceGrid, double> > > > storage(new  OptimizeGasStorage< MeanRevertingSimulator< OneDimData<OneDimRegularSpaceGrid, double> > >(injectionRateStorage, withdrawalRateStorage, injectionCostStorage, withdrawalCostStorage));
    // regressor
    ///////////
    int nMesh = 6;
    ArrayXi nbMesh = ArrayXi::Constant(1, nMesh);
    shared_ptr< BaseRegression > regressor = make_shared<LocalLinearRegression>(nbMesh);
    // final value
    function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>  vFunction = ZeroFunction();

    // initial values
    ArrayXd initialStock = ArrayXd::Constant(1,  p_maxLevelStorage);
    int initialRegime = 0; // only one regime

    // don't use one file to dump
    bool bOneFile = false;

    // Optimize
    ///////////
    std::string fileToDump = "CondExpGasStorageVaryingCavMPI" + to_string(world.size());
    double valueOptimDist ;
    {
        // link the simulations to the optimizer
        storage->setSimulator(backSimulator);
        boost::timer::auto_cpu_timer t;

        valueOptimDist =  DynamicProgrammingByRegressionVaryingGridsDist(p_timeChangeGrid, p_grids, storage, regressor, vFunction, initialStock, initialRegime, fileToDump, bOneFile, world);
        std::cout << " valueOptimDist " << valueOptimDist << std::endl ;
    }

    world.barrier() ;

    // a forward simulator
    ///////////////////////
    int nbsimulSim = 40000;
    bForward = true;
    shared_ptr< MeanRevertingSimulator< OneDimData<OneDimRegularSpaceGrid, double> > > forSimulator(new	  MeanRevertingSimulator< OneDimData<OneDimRegularSpaceGrid, double> > (futureGrid, sigma, mr, r, maturity, nstep, nbsimulSim, bForward));
    double valSimuDist ;
    {
        // link the simulations to the optimizer
        storage->setSimulator(forSimulator);
        boost::timer::auto_cpu_timer t;
        valSimuDist = SimulateRegressionVaryingGridsDist(p_timeChangeGrid, p_grids, storage, vFunction, initialStock, initialRegime, fileToDump, bOneFile, world) ;

    }
    if (world.rank() == 0)
    {
        std::cout << " valSimuDist  " << valSimuDist << " valueOptimDist " << valueOptimDist << std::endl ;
        BOOST_CHECK_CLOSE(valueOptimDist, valSimuDist, accuracyClose);
    }

    // a forward simulator
    ///////////////////////
    shared_ptr< MeanRevertingSimulator< OneDimData<OneDimRegularSpaceGrid, double> > > forSimulator2(new	  MeanRevertingSimulator< OneDimData<OneDimRegularSpaceGrid, double> > (futureGrid, sigma, mr, r, maturity, nstep, nbsimulSim, bForward));
    double valSimuDist2 ;
    {
        // link the simulations to the optimizer
        storage->setSimulator(forSimulator2);
        boost::timer::auto_cpu_timer t;
        valSimuDist2 = SimulateRegressionVaryingGridsControlDist(p_timeChangeGrid, p_grids, storage, vFunction, initialStock, initialRegime, fileToDump, bOneFile, world) ;

    }
    if (world.rank() == 0)
    {
        std::cout << " valSimuDist2  " << valSimuDist2 << " valueOptimDist " << valueOptimDist << std::endl ;
        BOOST_CHECK_CLOSE(valueOptimDist, valSimuDist2, accuracyClose);
    }
}

BOOST_AUTO_TEST_CASE(testSimpleStorageVaryingCavityeDist)
{
    // storage
    /////////
    double maxLevelStorage  = 90000;
    // define changing grids
    std::vector<double> timeChangeGrid;
    timeChangeGrid.reserve(3);
    std::vector<shared_ptr<FullGrid> > grids;
    grids.reserve(3);
    // grids
    ////////
    int nGrid = 10;
    ArrayXd lowValues1 = ArrayXd::Constant(1, 0.);
    ArrayXd step1 = ArrayXd::Constant(1, maxLevelStorage / nGrid);
    ArrayXi nbStep1 = ArrayXi::Constant(1, nGrid);
    // first grid
    timeChangeGrid.push_back(0.);
    grids.push_back(make_shared<RegularSpaceGrid>(lowValues1, step1, nbStep1));
    ArrayXd lowValues2 = ArrayXd::Constant(1, 30000.);
    ArrayXd step2 = ArrayXd::Constant(1, 10000.);
    ArrayXi nbStep2 = ArrayXi::Constant(1, 3);
    // second grid
    timeChangeGrid.push_back(0.3);
    grids.push_back(make_shared<RegularSpaceGrid>(lowValues2, step2, nbStep2));
    ArrayXd lowValues3 = ArrayXd::Constant(1, 0.);
    ArrayXd step3 = ArrayXd::Constant(1, 15000.);
    ArrayXi nbStep3 = ArrayXi::Constant(1, 6);
    timeChangeGrid.push_back(0.7);

    grids.push_back(make_shared<RegularSpaceGrid>(lowValues3, step3, nbStep3));

    testGasStorage(timeChangeGrid, grids, maxLevelStorage);

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
