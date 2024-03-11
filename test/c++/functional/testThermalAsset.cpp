
#ifndef USE_MPI
#define BOOST_TEST_MODULE testThermalAsset
#endif
#define BOOST_TEST_DYN_LINK
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#define _USE_MATH_DEFINES
#include <math.h>
#include <memory>
#include <utility>
#include <functional>
#include <boost/test/unit_test.hpp>
#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/OneDimRegularSpaceGrid.h"
#include "libflow/core/grids/OneDimData.h"
#include "libflow/core/grids/RegularSpaceIntGrid.h"
#include "libflow/regression/LocalLinearRegressionGeners.h"
#include "libflow/regression/LocalLinearRegressionGeners.h"
#include "libflow/dp/OptimizerSwitchBase.h"
#include "test/c++/tools/simulators/MeanReverting1DAssetsSimulator.h"
#include "test/c++/tools/dp/OptimizeThermalAsset.h"
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
double accuracyVarClose =  1.5;


// first test the OU simulator
BOOST_AUTO_TEST_CASE(testMeanReverting1D2Assets)
{
#ifdef USE_MPI
    boost::mpi::communicator world;
#endif

    size_t nstep = 10;
    // maturity
    double T = 365.;
    // define  a time grid
    shared_ptr<OneDimRegularSpaceGrid> timeGrid = make_shared<OneDimRegularSpaceGrid>(0., T / nstep, nstep);
    // to store the two assets
    vector< shared_ptr<OneDimData<OneDimRegularSpaceGrid, double> > > futGrids(2);
    vector< double > mr(2) ; // mean reverting
    vector< double > sig(2) ; // volatility
    Eigen::MatrixXd  correl = Eigen::MatrixXd::Zero(2, 2);
    // future values  for first asset
    shared_ptr<vector< double > > futValues1 = make_shared<vector<double> >(nstep + 1);
    for (size_t i = 0; i < nstep + 1 ; ++i)
        (*futValues1)[i] = 30. + 5 * cos((2 * M_PI * i) / nstep) + cos(2 * M_PI * i / 7.);
    // define the first future curve
    futGrids[0] =  make_shared<OneDimData< OneDimRegularSpaceGrid, double> >(timeGrid, futValues1);
    // future values for second asset
    shared_ptr<vector< double > > futValues2 = make_shared<vector<double> >(nstep + 1);
    for (size_t i = 0; i < nstep + 1 ; ++i)
        (*futValues2)[i] = 28. + 5 * cos((2 * M_PI * i) / nstep) + cos(2 * M_PI * i / 7.);
    // define the second future curve
    futGrids[1] =  make_shared<OneDimData< OneDimRegularSpaceGrid, double> >(timeGrid, futValues2);

    // OU parameters
    mr[0] = 0.01;
    mr[1] = 0.008;
    sig[0] = 0.08;
    sig[1] = 0.06;
    correl(0, 1) = 0.8;
    correl(1, 0) = 0.8;

    // time step
    double step = T / nstep;
    // number of simulations
    size_t nbsimul = 1e5;
    // a forward simulator
    bool bForward = true;
    // name to store simulations
    string toStoreSim = "simulFile";

    MeanReverting1DAssetsSimulator<OneDimData<OneDimRegularSpaceGrid, double> > simulator(futGrids, mr, sig, correl, T, nstep, nbsimul, bForward, toStoreSim
#ifdef USE_MPI
            , world
#endif
                                                                                         );

    for (size_t istep = 0; istep < nstep; ++istep)
    {
        // one step forward
        simulator.stepForward();
        ArrayXXd spots = simulator.getSpot();
        // check average
        double mean1 = spots.row(0).mean();
        double mean2 = spots.row(1).mean();
        BOOST_CHECK_CLOSE(mean1, (*futValues1)[istep + 1], accuracyClose);
        BOOST_CHECK_CLOSE(mean2, (*futValues2)[istep + 1], accuracyClose);
        // check variance
        double var1 = (spots.row(0) - mean1).pow(2).mean();
        double var2 = (spots.row(1) - mean2).pow(2).mean();
        double varOU1 = pow(sig[0], 2.) * (1. - exp(-2 * mr[0] * step * (istep + 1))) / (2 * mr[0]);
        double varOU2 = pow(sig[1], 2.) * (1. - exp(-2 * mr[1] * step * (istep + 1))) / (2 * mr[1]);
        double analVar1 =  pow((*futValues1)[istep + 1], 2.) * (exp(varOU1) - 1);
        double analVar2 =  pow((*futValues2)[istep + 1], 2.) * (exp(varOU2) - 1);
        BOOST_CHECK_CLOSE(var1, analVar1, accuracyVarClose);
        BOOST_CHECK_CLOSE(var2, analVar2, accuracyVarClose);
    }
    //   // backward simulator
    bForward = false;
    simulator.resetDirection(false);
    for (size_t istep = 0; istep < nstep ; ++istep)
    {
        // one step backward
        simulator.stepBackward();
        ArrayXXd spots = simulator.getSpot();
        int iLocstep = nstep - (istep + 1);
        // check average
        double mean1 = spots.row(0).mean();
        double mean2 = spots.row(1).mean();
        BOOST_CHECK_CLOSE(mean1, (*futValues1)[iLocstep], accuracyClose);
        BOOST_CHECK_CLOSE(mean2, (*futValues2)[iLocstep], accuracyClose);
        // check variance
        double var1 = (spots.row(0) - mean1).pow(2).mean();
        double var2 = (spots.row(1) - mean2).pow(2).mean();
        double varOU1 = pow(sig[0], 2.) * (1. - exp(-2 * mr[0] * step * iLocstep)) / (2 * mr[0]);
        double varOU2 = pow(sig[1], 2.) * (1. - exp(-2 * mr[1] * step * iLocstep)) / (2 * mr[1]);
        double analVar1 =  pow((*futValues1)[iLocstep], 2.) * (exp(varOU1) - 1);
        double analVar2 =  pow((*futValues2)[iLocstep], 2.) * (exp(varOU2) - 1);
        BOOST_CHECK_CLOSE(var1, analVar1, accuracyVarClose);
        BOOST_CHECK_CLOSE(var2, analVar2, accuracyVarClose);
    }
}

// Test a Thermal asset
BOOST_AUTO_TEST_CASE(testAThermalAsset)
{
#ifdef USE_MPI
    boost::mpi::communicator world;
#endif

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
    string toStoreSim = "simulFile";

    // go backward
    bool bForward = false;

    shared_ptr<SimClass> backwardSimulator = make_shared<SimClass> (futGrids, mr, sig, correl, T, nstep, nbsimul, bForward, toStoreSim
#ifdef USE_MPI
            , world
#endif
                                                                   );

    // affect simulator to optimizer object
    thermalAsset->setSimulator(backwardSimulator);

    // regressor
    ///////////
    int nMesh = 5;
    ArrayXi nbMesh = ArrayXi::Constant(2, nMesh);
    shared_ptr< LocalLinearRegression > regressor = make_shared<LocalLinearRegression>(nbMesh);


    // to store functon basis reconstruction in each regime
    string fileForExpCond = "funcBasisFile";

    boost::timer::cpu_timer timerOpt;

    // backward resolution
    double valueOptim = DynamicProgrammingSwitchingByRegression(gridPerReg, thermalAsset, regressor, stateInit, iReg0, fileForExpCond
#ifdef USE_MPI
                        , world
#endif

                                                               );

    timerOpt.stop() ;

#ifdef USE_MPI
    if (world.rank() == 0)
#endif
    {
        boost::chrono::duration<double> seconds = boost::chrono::nanoseconds(timerOpt.elapsed().user);
        cout << "  Optim value " << valueOptim <<  " Time "  << seconds.count()  <<  endl ;
    }
#ifdef USE_MPI
    world.barrier();
#endif

    // forward simulator
    bForward = true;
    int seed = 36;
    int nbsimulForward = 20000;
    int nbsimulPlot = 100;
    string fileForResTraj = "baseForResSwitch";

    shared_ptr<SimClass > forwardSimulator = make_shared<SimClass > (futGrids, mr, sig, correl, T, nstep, nbsimulForward, bForward, toStoreSim,
#ifdef USE_MPI
            world,
#endif
            seed);

    // affect simulator to optimizer object
    thermalAsset->setSimulator(forwardSimulator);

    boost::timer::cpu_timer timerSim;

    // now simulate
    pair< double, ArrayXi > valAndConstraints = SimulateRegressionSwitch<Asset, SimClass> (gridPerReg, thermalAsset, stateInit, iReg0, fileForExpCond, fileForResTraj, nbsimulPlot
#ifdef USE_MPI
            , world
#endif
                                                                                          ) ;

    timerSim.stop() ;

#ifdef USE_MPI
    if (world.rank() == 0)
#endif
    {
        double valSimu = valAndConstraints.first;
        boost::chrono::duration<double> seconds = boost::chrono::nanoseconds(timerSim.elapsed().user);
        cout << " val in Simulation " << valSimu <<  " Time "  << seconds.count()  <<  endl;
#ifdef USE_MPI
        // test constraint on minimal number of time step on and off
        BOOST_CHECK(nbMinOn <= valAndConstraints.second(0));
        BOOST_CHECK(nbMinOff <= valAndConstraints.second(1));
        BOOST_CHECK_CLOSE(valueOptim, valSimu, accuracyClose);
#endif
    }
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

