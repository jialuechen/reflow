

#include <iostream>
#include <sstream>
#ifndef USE_MPI
#define BOOST_TEST_MODULE testReservoirWithInflowsSDDP
#endif
#define BOOST_TEST_DYN_LINK
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#define _USE_MATH_DEFINES
#include <math.h>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/OneDimRegularSpaceGrid.h"
#include "libflow/core/grids/OneDimData.h"
#include "libflow/sddp/LocalConstRegressionForSDDPGeners.h"
#include "libflow/sddp/LocalLinearRegressionForSDDPGeners.h"
#include "libflow/sddp/SDDPFinalCut.h"
#include "libflow/sddp/SDDPLocalCut.h"
#include "libflow/sddp/backwardForwardSDDP.h"
#include "test/c++/tools/simulators/SimulatorGaussianSDDP.h"
#include "test/c++/tools/sddp/OptimizeReservoirWithInflowsSDDP.h"

using namespace std;
using namespace Eigen ;
using namespace libflow;

template< class LocalRegressionForSDDP >
void testStorageDemandSDDP(const int &p_nbStorage, const int &p_iterMax,  const int &p_sample,  const int &p_sampleCheck, const double p_accuracyClose, const int &p_nstepIterations,
                           const double &p_sigF,  const double &p_sigD)
{
#ifdef USE_MPI
    boost::mpi::communicator world;
#endif

    double maturity = 40;
    int nstep = 40;

    // optimizer
    //************
    double initLevel = maturity / 10.; // initial level of the stock
    double withdrawalRate = 2 ; // on each time step volume that can be withdrawn
    double sigF = p_sigF; //volatility for inflows
    double sigD = p_sigD * p_nbStorage ; // volatility of demand
    // define  a time grid
    shared_ptr<OneDimRegularSpaceGrid> timeGrid(new OneDimRegularSpaceGrid(0., maturity / nstep, nstep));
    // periodicity factor
    int iPeriod = 52;
    // define average demand
    shared_ptr<vector< double > > demandValues(new vector<double>(nstep + 1));
    for (int i = 0; i < nstep + 1; ++i)
        (*demandValues)[i] = (2. + 0.4 * cos((M_PI * i * iPeriod) / nstep)) * p_nbStorage;
    shared_ptr<OneDimData<OneDimRegularSpaceGrid, double> > demand(new OneDimData< OneDimRegularSpaceGrid, double> (timeGrid, demandValues));
    // define average inflow
    shared_ptr<vector< double > > inflowValues(new vector<double>(nstep + 1));
    for (int i = 0; i < nstep + 1; ++i)
        (*inflowValues)[i] = 1. + 0.2 * (cos((M_PI * i * iPeriod) / nstep) + sin((M_PI * i * iPeriod) / nstep));
    shared_ptr<OneDimData<OneDimRegularSpaceGrid, double> > flowAver(new OneDimData< OneDimRegularSpaceGrid, double> (timeGrid, inflowValues));
    // define  spot price
    shared_ptr<vector< double > > spotValues(new vector<double>(nstep + 1));
    for (int i = 0; i < nstep + 1; ++i)
        (*spotValues)[i] = 50. + 20 * sin((M_PI * i * iPeriod) / nstep);

    shared_ptr<OneDimData<OneDimRegularSpaceGrid, double> > spot(new OneDimData< OneDimRegularSpaceGrid, double> (timeGrid, spotValues));



    // initial state
    ArrayXd initialState = ArrayXd::Constant(p_nbStorage, initLevel);

    /// final cut
    ArrayXXd finalCut =  ArrayXXd::Zero(1 + p_nbStorage, 1);
    SDDPFinalCut finCut(finalCut);

    // no regression here
    ArrayXi nbMesh;

    // backward and forward simulator ( two uncertainties times  nbsimul )
    int nbUncertainties = 1 + p_nbStorage;
    shared_ptr<SimulatorGaussianSDDP>  backSimulator = make_shared<SimulatorGaussianSDDP>(nbUncertainties, p_sample);
    shared_ptr<SimulatorGaussianSDDP> forSimulator = make_shared<SimulatorGaussianSDDP>(nbUncertainties);

    // define the storage
    shared_ptr<OptimizerSDDPBase >  optimizer = make_shared<OptimizeReservoirWithInflowsSDDP<SimulatorGaussianSDDP> >(initLevel, withdrawalRate, p_nbStorage, sigF,   flowAver, sigD,  demand, spot, backSimulator, forSimulator);

    // optimisation dates
    ArrayXd dates = ArrayXd::LinSpaced(nstep + 1, 0., maturity);

    // names for archive
    string nameRegressor = "RegressorReservoir";
    string nameCut = "CutReservoir";
    string nameVisitedStates = "VisitedStateReservoir";
    int nIterMax = p_iterMax;
    double accuracy = p_accuracyClose / 100.;
    ostringstream stringStream; // store intermediate results

    pair<double, double>  values = backwardForwardSDDP<LocalRegressionForSDDP>(optimizer,  p_sampleCheck, initialState,
                                   finCut, dates,  nbMesh, nameRegressor, nameCut, nameVisitedStates, nIterMax,
                                   accuracy,  p_nstepIterations, stringStream
#ifdef USE_MPI
                                   , world
#endif
                                                                              );

#ifdef USE_MPI
    if (world.rank() == 0)
#endif
    {
        cout << stringStream.str() << endl ;
        cout << "Nb storage " << p_nbStorage << " Value Optim " <<  values.first << " and Simulation " << values.second << " Iteration " << nIterMax << endl ;
        BOOST_CHECK(accuracy <= (p_accuracyClose / 100.));
    }

}


BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsSDDP1DDeterminist)
{
    int dim = 1; // number of storage
    int iterMax = 100; /// maximal number of iteration forward/backward
    int   nbSample = 1 ; // number of samples
    int   nbSampleCheck = 1 ; // number of samples for checking convergence
    double  error    = 0.1 ; // percentage between optimization and simulation allowed
    int     nstep = 10 ; /// accuracy is checked every nstep iterations
    double sigF = 0; /// vol for inflows
    double  sigD = 0. ; /// vol for demand
    // 0.5% between optimization and simulation
    testStorageDemandSDDP<LocalLinearRegressionForSDDP>(dim, iterMax,  nbSample, nbSampleCheck, error, nstep, sigF, sigD);
    testStorageDemandSDDP<LocalConstRegressionForSDDP>(dim, iterMax,  nbSample, nbSampleCheck, error, nstep, sigF, sigD);
}

BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsSDDP2DDeterminist)
{
    int dim = 2; // number of storage
    int iterMax = 100; /// maximal number of iteration forward/backward
    int  nbSample = 1 ; // number of samples
    int   nbSampleCheck = 1 ; // number of samples for checking convergence
    double  error    = 0.1 ; // percentage between optimization and simulation allowed
    int     nstep = 10 ; /// accuracy is checked every nstep iterations
    double sigF = 0; /// vol for inflows
    double  sigD = 0. ; /// vol for demand
    // 0.5% between optimization and simulation
    testStorageDemandSDDP<LocalLinearRegressionForSDDP>(dim,  iterMax, nbSample, nbSampleCheck, error, nstep, sigF, sigD);
    testStorageDemandSDDP<LocalConstRegressionForSDDP>(dim,  iterMax, nbSample, nbSampleCheck, error, nstep, sigF, sigD);
}
BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsSDDP5DDeterminist)
{
    int dim = 5; // number of storage
    double iterMax = 100; /// maximal number of iteration forward/backward
    int  nbSample = 1 ; // number of samples
    int   nbSampleCheck = 1 ; // number of samples for checking convergence
    double  error    = 0.1 ; // percentage between optimization and simulation allowed
    int     nstep = 10 ; /// accuracy is checked every nstep iterations
    double sigF = 0; /// vol for inflows
    double  sigD = 0. ; /// vol for demand
    // 0.5% between optimization and simulation
    testStorageDemandSDDP<LocalLinearRegressionForSDDP>(dim,  iterMax, nbSample,  nbSampleCheck, error, nstep, sigF, sigD);
    testStorageDemandSDDP<LocalConstRegressionForSDDP>(dim,  iterMax, nbSample,  nbSampleCheck, error, nstep, sigF, sigD);
}


BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsSDDP1D)
{
    int ndim = 1; // number of storage
    int iterMax = 200; /// maximal number of iteration forward/backward
    int  nbSample = 200 ; // number of samples
    int   nbSampleCheck = 4000; // number of samples for checking convergence
    double  error    = 1.5 ; // percentage between optimization and simulation allowed
    int     nstep = 5 ; /// accuracy is checked every nstep iterations
    double sigF = 0.6; /// vol for inflows
    double  sigD = 0.6; /// vol for demand
    // 0.5% between optimization and simulation
    testStorageDemandSDDP<LocalLinearRegressionForSDDP>(ndim,  iterMax, nbSample, nbSampleCheck, error, nstep, sigF, sigD);
    testStorageDemandSDDP<LocalConstRegressionForSDDP>(ndim,  iterMax, nbSample, nbSampleCheck, error, nstep, sigF, sigD);
}

#ifdef USE_LONG_TEST
BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsSDDP2D)
{
    int ndim = 2; // number of storage
    int  iterMax = 200; /// maximal number of iteration forward/backward
    double  nbSample = 2000 ; // number of samples
    int   nbSampleCheck = 5000; // number of samples for checking convergence
    double  error    =  2; // percentage between optimization and simulation allowed
    int     nstep = 10 ; /// accuracy is checked every nstep iterations
    double sigF = 0.6; /// vol for inflows
    double  sigD = 0.8 ; /// vol for demand
    // 2% between optimization and simulation
    testStorageDemandSDDP<LocalLinearRegressionForSDDP>(ndim, iterMax, nbSample, nbSampleCheck, error, nstep, sigF, sigD);
    testStorageDemandSDDP<LocalConstRegressionForSDDP>(ndim, iterMax, nbSample, nbSampleCheck, error, nstep, sigF, sigD);
}

BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsSDDP5D)
{
    int ndim = 5; // number of storage
    double iterMax = 100; /// maximal number of iteration forward/backward
    double  nbSample = 1000 ; // number of samples
    int   nbSampleCheck = 8000; // number of samples for checking convergence
    double  error    = 2. ; // percentage between optimization and simulation allowed
    int     nstep = 10 ; /// accuracy is checked every nstep iterations
    double sigF = 0.6; /// vol for inflows
    double  sigD = 0.8 ; /// vol for demand
    // 2% between optimization and simulation
    testStorageDemandSDDP<LocalLinearRegressionForSDDP>(ndim,  iterMax, nbSample,  nbSampleCheck, error, nstep, sigF, sigD);
    testStorageDemandSDDP<LocalConstRegressionForSDDP>(ndim,  iterMax, nbSample,  nbSampleCheck, error, nstep, sigF, sigD);
}
#endif

#ifdef USE_MPI
// (empty) Initialization function. Can't use testing tools here.
bool init_function()
{
    return true;
}

int main(int argc, char *argv[])
{
// #if defined   __linux
//     enable_abort_on_floating_point_exception();
// #endif
    boost::mpi::environment env(argc, argv);
    return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
#endif
