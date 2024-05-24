
#ifndef USE_MPI
#define BOOST_TEST_MODULE testGasStorageTreeCut
#endif
#define BOOST_TEST_DYN_LINK
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#define _USE_MATH_DEFINES
#include <math.h>
#include <functional>
#include <memory>
#include <boost/test/unit_test.hpp>
#include <boost/timer/timer.hpp>
#include <Eigen/Dense>
#include "reflow/core/grids/OneDimRegularSpaceGrid.h"
#include "reflow/core/grids/OneDimData.h"
#include "reflow/core/grids/RegularSpaceGridGeners.h"
#include "reflow/core/grids/RegularLegendreGridGeners.h"
#include "test/c++/tools/simulators/TrinomialTreeOUSimulator.h"
#include "test/c++/tools/simulators/MeanRevertingSimulatorTree.h"
#include "test/c++/tools/dp/DynamicProgrammingByTreeCut.h"
#include "test/c++/tools/dp/SimulateTreeCut.h"
#include "test/c++/tools/dp/OptimizeGasStorageTreeCut.h"

using namespace std;
using namespace Eigen ;
using namespace reflow;

double accuracyClose = 2.;
double accuracyEqual = 0.0001;


class ZeroFunction
{
private :
    int m_nDim;
public:
    ZeroFunction(const int &p_nDim): m_nDim(p_nDim) {}
    ArrayXd  operator()(const int &, const ArrayXd &, const ArrayXd &) const
    {
        return ArrayXd::Zero(m_nDim) ;
    }
};



#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif

template< class Grid>
void   testGasStorageTreeCut(shared_ptr< Grid > &p_grid, const double &p_maxLevelStorage)
{
    // storage
    /////////
    double injectionRateStorage = 20000;
    double withdrawalRateStorage = 15000;
    double injectionCostStorage = 0.35;
    double withdrawalCostStorage = 0.35;

    double maturity = 1.;
    size_t nstep = 20;
    // define  a time grid
    shared_ptr<OneDimRegularSpaceGrid> timeGrid = make_shared< OneDimRegularSpaceGrid>(0., maturity / nstep, nstep);
    // future values
    shared_ptr<vector< double > > futValues(new vector<double>(nstep + 1));
    // periodicity factor
    int iPeriod = 52;
    for (size_t i = 0; i < nstep + 1; ++i)
        (*futValues)[i] = 50. + 5 * sin((M_PI * i * iPeriod) / nstep);
    // define the future curve
    shared_ptr<OneDimData<OneDimRegularSpaceGrid, double> > futureGrid = make_shared< OneDimData< OneDimRegularSpaceGrid, double> >(timeGrid, futValues);

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
    string nameTree = "Tree";
#ifdef USE_MPI
    boost::mpi::communicator world;
    if (world.rank() == 0)
#endif
        tree.dump(nameTree, indexT);
#ifdef USE_MPI
    world.barrier();
#endif

    // open archive to create a simualtor
    shared_ptr<gs::BinaryFileArchive> binArxiv = make_shared<gs::BinaryFileArchive>(nameTree.c_str(), "r");

    // final value
    function< ArrayXd(const int &, const ArrayXd &, const ArrayXd &)>   vFunction = ZeroFunction(2);

    // a backward simulator
    ///////////////////////
    shared_ptr< MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > backSimulator =   make_shared<MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > (binArxiv, futureGrid, sigma, mr);
    // optimizer
    ///////////
    shared_ptr< OptimizeGasStorageTreeCut< MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> >  > > storage =
        make_shared<  OptimizeGasStorageTreeCut< MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> >  > > (injectionRateStorage, withdrawalRateStorage, injectionCostStorage, withdrawalCostStorage);

    // initial values
    ArrayXd initialStock = ArrayXd::Constant(1, p_maxLevelStorage);
    int initialRegime = 0; // only one regime

    // Optimize
    ///////////
    string fileToDump = "CondCut";
    // link the simulations to the optimizer
    storage->setSimulator(backSimulator);
    double valueOptim =  DynamicProgrammingByTreeCut(p_grid, storage, vFunction, initialStock, initialRegime, fileToDump
#ifdef USE_MPI
                         , world
#endif
                                                    );


    // a forward simulator
    ///////////////////////
    int nbsimulSim = 8000;
    shared_ptr< MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > forSimulator = make_shared< MeanRevertingSimulatorTree< OneDimData<OneDimRegularSpaceGrid, double> > > (binArxiv, futureGrid, sigma, mr, nbsimulSim);

    // link the simulations to the optimizer
    storage->setSimulator(forSimulator);
    double valSimu = SimulateTreeCut(p_grid, storage, vFunction, initialStock, initialRegime, fileToDump
#ifdef USE_MPI
                                     , world
#endif
                                    ) ;

    BOOST_CHECK_CLOSE(valueOptim, valSimu, accuracyClose);
    cout << " Optim " << valueOptim << " valSimu " << valSimu <<  endl ;

}


BOOST_AUTO_TEST_CASE(testSimpleStorageTreeCut)
{
    // storage
    /////////
    double maxLevelStorage  = 40000;
    // grid
    //////
    int nGrid = 40;
    ArrayXd lowValues = ArrayXd::Constant(1, 0.);
    ArrayXd step = ArrayXd::Constant(1, maxLevelStorage / nGrid);
    ArrayXi nbStep = ArrayXi::Constant(1, nGrid);
    shared_ptr<FullGrid> grid = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
    testGasStorageTreeCut(grid, maxLevelStorage);

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
