
#define BOOST_TEST_DYN_LINK
#define _USE_MATH_DEFINES
#include <math.h>
#include <functional>
#include <memory>
#include <boost/test/unit_test.hpp>
#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/OneDimRegularSpaceGrid.h"
#include "libflow/core/grids/OneDimData.h"
#include "libflow/core/grids/RegularSpaceIntGrid.h"
#include "libflow/regression/LocalLinearRegressionGeners.h"
#include "libflow/regression/LocalLinearRegressionGeners.h"
#include "libflow/dp/OptimizerSwitchBase.h"
#include "test/c++/tools/simulators/MeanReverting1DAssetsSimulator.h"
#include "test/c++/tools/dp/OptimizeThermalAsset.h"
#include "test/c++/tools/dp/DynamicProgrammingSwitchingByRegressionDist.h"
#include "test/c++/tools/dp/DynamicProgrammingSwitchingByRegression.h"
#include "test/c++/tools/dp/SimulateRegressionSwitch.h"

using namespace std;
using namespace Eigen ;
using namespace libflow;


#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif

double accuracyClose =  1.;


// Test a Thermal asset in distributed mode and MPI
BOOST_AUTO_TEST_CASE(testAThermalAssetMpiDist)
{
    boost::mpi::communicator world;

    // maturity  in days
    double T = 100.;
    // number of time step  ( step of 4 hours : 6 step per day)
    size_t nStepPerday = 6;
    size_t nbDays = static_cast<int>(T);
    size_t nstep = nStepPerday * nbDays;
    // define  a time grid
    shared_ptr<OneDimRegularSpaceGrid> timeGrid = make_shared<OneDimRegularSpaceGrid>(0., T / nstep, nstep);
    // to store the two assets
    vector< shared_ptr<OneDimData<OneDimRegularSpaceGrid, double> > > futGrids(2);
    vector< double > mr(2) ; // mean reverting here in days
    vector< double > sig(2) ; // volatility in days
    Eigen::MatrixXd  correl = Eigen::MatrixXd::Zero(2, 2);
    // future values  for first asset : gas
    shared_ptr<vector< double > > futValues1 = make_shared<vector<double> >(nstep + 1);
    for (size_t i = 0; i < nstep + 1 ; ++i)
        (*futValues1)[i] = 30. + 5 * cos((2 * M_PI * i) / nstep) + cos(2 * M_PI * i / (7.*nStepPerday)) ;
    // define the first future curve
    futGrids[0] =  make_shared<OneDimData< OneDimRegularSpaceGrid, double> >(timeGrid, futValues1);
    // future values for second asset : electricity
    shared_ptr<vector< double > > futValues2 = make_shared<vector<double> >(nstep + 1);
    for (size_t i = 0; i < nstep + 1 ; ++i)
        (*futValues2)[i] = 30. + 5 * cos((2 * M_PI * i) / nstep) + cos(2 * M_PI * i / (7.*nStepPerday))  + 5.*cos(2 * M_PI * i / nStepPerday) ;
    // define the second future curve
    futGrids[1] =  make_shared<OneDimData< OneDimRegularSpaceGrid, double> >(timeGrid, futValues2);

    // OU parameters
    mr[0] = 0.004;
    mr[1] = 0.01;
    sig[0] = 0.02;
    sig[1] = 0.08;
    correl(0, 1) = 0.8;
    correl(1, 0) = 0.8;

    // Constraints
    int nbMinOn = 6 ; //  minimal number of time step the  thermal asset is on
    int nbMinOff = 4 ; //  maximal number of time step the terhma asset is off
    vector< shared_ptr< RegularSpaceIntGrid > > gridPerReg(2);
    ArrayXi lowValue(1);
    lowValue(0) = 0;
    ArrayXi nbStep1(1);
    nbStep1(0) = nbMinOn - 1;
    // first regime : on
    gridPerReg[0] = make_shared<RegularSpaceIntGrid>(lowValue, nbStep1);
    ArrayXi nbStep2(1);
    nbStep2(0) = nbMinOff - 1;
    // second regime off
    gridPerReg[1] = make_shared<RegularSpaceIntGrid>(lowValue, nbStep2);

    /// create Thermal Asset object
    int switchOffToOnCost = 4;

    typedef  MeanReverting1DAssetsSimulator<OneDimData<OneDimRegularSpaceGrid, double> > SimClass;
    typedef  OptimizeThermalAsset< MeanReverting1DAssetsSimulator<OneDimData<OneDimRegularSpaceGrid, double> > >  Asset ;

    shared_ptr<Asset>  thermalAsset = make_shared<Asset >(switchOffToOnCost);

    // initial positin
    int iReg0 = 0; // on
    ArrayXi stateInit(1);
    stateInit(0) = 0 ; // juste starting time step before 0

    // create teh grids
    // number of simulations
    size_t nbsimul = 50000;
    // name to store simulations
    string toStoreSim = "simulFileMpi" + to_string(world.size());

    // go backward
    bool bForward = false;

    shared_ptr<SimClass > backwardSimulator = make_shared<SimClass > (futGrids, mr, sig, correl, T, nstep, nbsimul, bForward, toStoreSim, world);

    // affect simulator to optimizer object
    thermalAsset->setSimulator(backwardSimulator);

    // barrier
    world.barrier();

    // regressor
    ///////////
    int nMesh = 5;
    ArrayXi nbMesh = ArrayXi::Constant(2, nMesh);
    shared_ptr< LocalLinearRegression > regressor = make_shared<LocalLinearRegression>(nbMesh);


    // to store functon basis reconstruction in each regime
    string fileForExpCond = "funcBasisFileMPI" + to_string(world.size());

    boost::timer::cpu_timer timerOpt;

    // backward resolution
    double  valueOptim = DynamicProgrammingSwitchingByRegressionDist(gridPerReg, thermalAsset, regressor, stateInit, iReg0, fileForExpCond, world);


    timerOpt.stop() ;

    if (world.rank() == 0)
    {
        boost::chrono::duration<double> seconds = boost::chrono::nanoseconds(timerOpt.elapsed().user);
        cout << "  Optim value " << valueOptim <<  " Time "  << seconds.count()  <<  endl ;
    }
    world.barrier();

    // forward simulator
    bForward = true;
    int seed = 36;
    int nbsimulForward = 20000;
    int nbsimulPlot = 100;
    string fileForResTraj = "baseForResSwitchMpi" + to_string(world.size());
    string toStoreSim1 = "simulFileMpiF" + to_string(world.size());

    shared_ptr<SimClass > forwardSimulator = make_shared<SimClass> (futGrids, mr, sig, correl, T, nstep, nbsimulForward, bForward, toStoreSim1, world, seed);

    // affect simulator to optimizer object
    thermalAsset->setSimulator(forwardSimulator);

    boost::timer::cpu_timer timerSim;
    world.barrier();

    // now simulate
    pair< double, ArrayXi > valAndConstraints  = SimulateRegressionSwitch<Asset, SimClass>(gridPerReg, thermalAsset, stateInit, iReg0, fileForExpCond, fileForResTraj, nbsimulPlot, world) ;

    timerSim.stop() ;

    if (world.rank() == 0)
    {
        double valSimu = valAndConstraints.first;
        boost::chrono::duration<double> seconds = boost::chrono::nanoseconds(timerSim.elapsed().user);
        cout << " val in Simulation " << valSimu <<  " Time "  << seconds.count()  <<  endl;
        BOOST_CHECK(nbMinOn <= valAndConstraints.second(0));
        BOOST_CHECK(nbMinOff <= valAndConstraints.second(1));
        BOOST_CHECK_CLOSE(valueOptim, valSimu, accuracyClose);
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
