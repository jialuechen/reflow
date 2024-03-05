// Copyright (C) 2019 Fime
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
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
#include "libflow/core/grids/SparseSpaceGridBound.h"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/core/grids/RegularLegendreGridGeners.h"
#include "test/c++/tools/simulators/TrinomialTreeOUSimulator.h"
#include "test/c++/tools/simulators/MeanRevertingSimulatorTree.h"
#include "test/c++/tools/dp/DynamicProgrammingByTreeDist.h"
#include "test/c++/tools/dp/SimulateTreeDist.h"
#include "test/c++/tools/dp/SimulateTreeControlDist.h"
#include "test/c++/tools/dp/OptimizeGasStorageTree.h"


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

template< class Grid>
double  testGasStorage(shared_ptr< Grid > &p_grid, const double &p_maxLevelStorage, const    bool p_bOneFile, const boost::mpi::communicator &p_world,
                       const string &p_strAdd)
{
    // storage
    /////////
    double injectionRateStorage = 60000;
    double withdrawalRateStorage = 45000;
    double injectionCostStorage = 0.35;
    double withdrawalCostStorage = 0.35;

    double maturity = 1.;
    size_t nstep = 10;
    // define  a time grid
    shared_ptr<OneDimRegularSpaceGrid> timeGrid(new OneDimRegularSpaceGrid(0., maturity / nstep, nstep));
    // future values
    shared_ptr<vector< double > > futValues(new vector<double>(nstep + 1));
    // periodicity factor
    int iPeriod = 52;
    for (size_t i = 0; i < nstep + 1; ++i)
        (*futValues)[i] = 50. + 20 * sin((M_PI * i * iPeriod) / nstep);
    // define the future curve
    shared_ptr<OneDimData<OneDimRegularSpaceGrid, double> > futureGrid(new OneDimData< OneDimRegularSpaceGrid, double> (timeGrid, futValues));

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
    string nameTree = "TreeMpi" + p_strAdd + to_string(p_world.size());
    if (p_world.rank() == 0)
        tree.dump(nameTree, indexT);
    p_world.barrier();

    // open archive to create a simualtor
    shared_ptr<gs::BinaryFileArchive> binArxiv = make_shared<gs::BinaryFileArchive>(nameTree.c_str(), "r");


    // a backward simulator
    ///////////////////////
    shared_ptr< MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > backSimulator = make_shared<MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > >(binArxiv, futureGrid, sigma, mr);
    // optimizer
    ///////////
    shared_ptr< OptimizeGasStorageTree< MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > > storage = make_shared< OptimizeGasStorageTree< MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > >(injectionRateStorage, withdrawalRateStorage, injectionCostStorage, withdrawalCostStorage);

    // initial values
    ArrayXd initialStock = ArrayXd::Constant(1, p_maxLevelStorage);
    int initialRegime = 0; // only one regime

    // final value
    function<double(const int &, const ArrayXd &, const ArrayXd &)>  vFunction = ZeroFunction();

    // Optimize
    ///////////
    string fileToDump = "CondTreeMpiExp" + p_strAdd;
    // link the simulations to the optimizer
    storage->setSimulator(backSimulator);
    double valueOptimDist =  DynamicProgrammingByTreeDist(p_grid, storage,  vFunction, initialStock, initialRegime, fileToDump, p_bOneFile, p_world);

    p_world.barrier();

    // a forward simulator
    ///////////////////////
    int nbsimulSim = 40000;
    shared_ptr< MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > forSimulator = make_shared<MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > (binArxiv, futureGrid, sigma, mr, nbsimulSim);

    // link the simulations to the optimizer
    storage->setSimulator(forSimulator);
    double valSimuDist = SimulateTreeDist(p_grid, storage, vFunction, initialStock, initialRegime, fileToDump, p_bOneFile, p_world) ;

    if (p_world.rank() == 0)
    {
        BOOST_CHECK_CLOSE(valueOptimDist, valSimuDist, accuracyClose);
        cout << " Optim " << valueOptimDist << " valSimuDist " << valSimuDist <<  endl ;
    }

    /// a second forward simulator
    /////////////////////////////
    shared_ptr< MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > forSimulator2 = make_shared<MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > (binArxiv, futureGrid, sigma, mr, nbsimulSim);

    double valSimuDist2 ;
    {
        // link the simulations to the optimizer
        storage->setSimulator(forSimulator2);
        valSimuDist2 = SimulateTreeControlDist(p_grid, storage, vFunction, initialStock, initialRegime, fileToDump, p_bOneFile, p_world) ;

    }
    if (p_world.rank() == 0)
    {
        cout << " valSimuDist2  " << valSimuDist2 << " valueOptimDist " << valueOptimDist << endl ;
        BOOST_CHECK_CLOSE(valueOptimDist, valSimuDist2, accuracyClose);
    }
    return valueOptimDist;
}


BOOST_AUTO_TEST_CASE(testSimpleStorageTreeDist)
{
    boost::mpi::communicator world;
    world.barrier();
    // storage
    /////////
    double maxLevelStorage  = 90000;
    // grid
    //////
    int nGrid = 50;
    ArrayXd lowValues = ArrayXd::Constant(1, 0.);
    ArrayXd step = ArrayXd::Constant(1, maxLevelStorage / nGrid);
    ArrayXi nbStep = ArrayXi::Constant(1, nGrid);
    shared_ptr<RegularSpaceGrid> grid = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
    bool bOneFile = true ;
    double val = testGasStorage(grid, maxLevelStorage, bOneFile, world, "");


    // grid
    //////
    ArrayXi poly = ArrayXi::Constant(1, 1);
    shared_ptr<RegularLegendreGrid> gridL = make_shared<RegularLegendreGrid>(lowValues, step, nbStep, poly);

    double valLegendre = testGasStorage(gridL, maxLevelStorage, bOneFile, world, "");

    if (world.rank() == 0)
    {
        BOOST_CHECK_CLOSE(val, valLegendre, accuracyEqual) ;
    }
}

BOOST_AUTO_TEST_CASE(testSimpleStorageTreeDistCommunicator)
{
    boost::mpi::communicator world;
    if (world.size() > 1)
    {
        // create 2 communicators
        bool bSeparate = (2 * world.rank() < world.size());
        boost::mpi::communicator worldLoc = world.split(bSeparate ? 0 : 1);
        string strToAdd = (bSeparate ? "0" : "1");

        // storage
        /////////
        double maxLevelStorage  = 90000;
        // grid
        //////
        int nGrid = 50;
        ArrayXd lowValues = ArrayXd::Constant(1, 0.);
        ArrayXd step = ArrayXd::Constant(1, maxLevelStorage / nGrid);
        ArrayXi nbStep = ArrayXi::Constant(1, nGrid);
        shared_ptr<RegularSpaceGrid> grid = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
        bool bOneFile = true ;
        double val0 = testGasStorage(grid, maxLevelStorage, bOneFile, worldLoc, strToAdd);
        double val1;
        // sending
        if (2 * world.rank() ==  world.size())
        {
            world.send(0, 0, val0);
        }
        else if (world.rank() == 0)
        {
            world.recv(int(world.size() / 2), 0, val1);
            BOOST_CHECK_CLOSE(val0, val1, accuracyEqual);
        }
    }
}

BOOST_AUTO_TEST_CASE(testSimpleStorageTreeMultipleFileDist)
{
    boost::mpi::communicator world;
    world.barrier();
    // storage
    /////////
    double maxLevelStorage  = 90000;
    // grid
    //////
    int nGrid = 50;
    ArrayXd lowValues = ArrayXd::Constant(1, 0.);
    ArrayXd step = ArrayXd::Constant(1, maxLevelStorage / nGrid);
    ArrayXi nbStep = ArrayXi::Constant(1, nGrid);
    shared_ptr<RegularSpaceGrid> grid = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
    bool bOneFile = false ;
    testGasStorage(grid, maxLevelStorage, bOneFile, world, "");
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
