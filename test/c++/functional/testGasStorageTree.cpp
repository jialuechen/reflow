
#ifndef USE_MPI
#define BOOST_TEST_MODULE testGasStorageTree
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
#include "libflow/core/grids/OneDimRegularSpaceGrid.h"
#include "libflow/core/grids/OneDimData.h"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/core/grids/RegularLegendreGridGeners.h"
#include "libflow/tree/TreeGeners.h"
#include "test/c++/tools/simulators/TrinomialTreeOUSimulator.h"
#include "test/c++/tools/simulators/MeanRevertingSimulatorTree.h"
#include "test/c++/tools/dp/DynamicProgrammingByTree.h"
#include "test/c++/tools/dp/SimulateTree.h"
#include "test/c++/tools/dp/SimulateTreeControl.h"
#include "test/c++/tools/dp/OptimizeGasStorageTree.h"

using namespace std;
using namespace Eigen ;
using namespace libflow;


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



/// \brief valorization of a given gas storage on a  grid with a tree method
/// \param p_grid             the grid
/// \param p_maxLevelStorage  maximum level
void testGasStorageFullGridTree(shared_ptr< FullGrid > &p_grid, const double &p_maxLevelStorage)
{
#ifdef USE_MPI
    boost::mpi::communicator world;
#endif
    // storage
    /////////
    double injectionRateStorage = 60000;
    double withdrawalRateStorage = 45000;
    double injectionCostStorage = 0.35;
    double withdrawalCostStorage = 0.35;

    double maturity = 1.;
    size_t nstep = 100;
    // define  a time grid
    shared_ptr<OneDimRegularSpaceGrid> timeGrid = make_shared<OneDimRegularSpaceGrid>(0., maturity / nstep, nstep);
    // future values
    shared_ptr<vector< double > > futValues = make_shared<vector<double> >(nstep + 1);
    // periodicity factor
    int iPeriod = 52;
    for (size_t i = 0; i < nstep + 1; ++i)
        (*futValues)[i] = 50. + 20 * sin((M_PI * i * iPeriod) / nstep);
    // define the future curve
    shared_ptr<OneDimData<OneDimRegularSpaceGrid, double> > futureGrid = make_shared<OneDimData< OneDimRegularSpaceGrid, double> >(timeGrid, futValues);

    // Create a tree to be used for the simulator
    //*******************************************
    // number to sub-discretization steps
    int nbStepTreePerStep = 2;
    ArrayXd ddates =  ArrayXd::LinSpaced(nstep * nbStepTreePerStep + 1, 0., maturity);
    double  sigma =  0.94;
    double  mr =  0.29;

    TrinomialTreeOUSimulator tree(mr, sigma, ddates);

    // sub array for dates
    ArrayXi indexT(nstep + 1);
    for (size_t i = 0; i < nstep; ++i)
        indexT(i) = i * nbStepTreePerStep;
    indexT(nstep) = ddates.size() - 1;


    // create archive
    string nameTree = "TreeGasStorage";
#ifdef USE_MPI
    nameTree += to_string(world.size());
    if (world.rank() == 0)
#endif
        tree.dump(nameTree, indexT);
#ifdef USE_MPI
    world.barrier();
#endif

    // open archive to create a simualtor
    shared_ptr<gs::BinaryFileArchive> binArxiv = make_shared<gs::BinaryFileArchive>(nameTree.c_str(), "r");


    // a backward simulator
    ///////////////////////
    shared_ptr< MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > backSimulator = make_shared<MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > >(binArxiv, futureGrid, sigma, mr);
    // optimizer
    ///////////
    shared_ptr< OptimizeGasStorageTree< MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > > storage = make_shared< OptimizeGasStorageTree< MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > >(injectionRateStorage, withdrawalRateStorage, injectionCostStorage, withdrawalCostStorage);

    // final value
    function<double(const int &, const ArrayXd &, const ArrayXd &)>  vFunction = ZeroFunction();

    // initial values
    ArrayXd initialStock = ArrayXd::Constant(1, p_maxLevelStorage);
    int initialRegime = 0; // only one regime


    // Optimize
    ///////////
    string fileToDump = "CondTreeGasStorage";
    double valueOptim ;
    {
        // link the simulations to the optimizer
        storage->setSimulator(backSimulator);
        boost::timer::auto_cpu_timer t;

        valueOptim =  DynamicProgrammingByTree(p_grid, storage,  vFunction, initialStock, initialRegime, fileToDump
#ifdef USE_MPI
                                               , world
#endif

                                              );
    }
    cout << "valueOptim " << valueOptim << endl ;
    // a forward simulator
    ///////////////////////
    int nbsimulSim = 50000;
    shared_ptr< MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > forSimulator = make_shared<MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > (binArxiv, futureGrid, sigma, mr, nbsimulSim);
    double valSimu ;
    {
        // link the simulations to the optimizer
        storage->setSimulator(forSimulator);
        boost::timer::auto_cpu_timer t;
        valSimu = SimulateTree(p_grid, storage, vFunction, initialStock, initialRegime, fileToDump
#ifdef USE_MPI
                               , world
#endif

                              ) ;

    }
    cout << " valSimu  " << valSimu << " valueOptim " << valueOptim << endl ;
    BOOST_CHECK_CLOSE(valueOptim, valSimu, accuracyClose);

    /// a second forward simulator
    /////////////////////////////
    shared_ptr< MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > forSimulator2 = make_shared<MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > (binArxiv, futureGrid, sigma, mr, nbsimulSim);

    double valSimu2 ;
    {
        // link the simulations to the optimizer
        storage->setSimulator(forSimulator2);
        boost::timer::auto_cpu_timer t;
        valSimu2 = SimulateTreeControl(p_grid, storage, vFunction, initialStock, initialRegime, fileToDump
#ifdef USE_MPI
                                       , world
#endif

                                      ) ;

    }
    cout << " valSimu2  " << valSimu2 << " valueOptim " << valueOptim << endl ;
    BOOST_CHECK_CLOSE(valueOptim, valSimu2, accuracyClose);
}

BOOST_AUTO_TEST_CASE(testSimpleStorageTree)
{
    // storage
    /////////
    double maxLevelStorage  = 90000;
    // grid
    //////
    int nGrid = 50;
    ArrayXd lowValues = ArrayXd::Constant(1, 0.);
    ArrayXd step = ArrayXd::Constant(1, maxLevelStorage / nGrid);
    ArrayXi nbStep = ArrayXi::Constant(1, nGrid);
    shared_ptr<FullGrid> grid = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);

    testGasStorageFullGridTree(grid, maxLevelStorage);
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
