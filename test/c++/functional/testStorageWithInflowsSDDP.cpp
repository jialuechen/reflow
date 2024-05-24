

#include <iostream>
#include <sstream>
#ifndef USE_MPI
#define BOOST_TEST_MODULE testStorageWithInflowsSDDP
#endif
#define BOOST_TEST_DYN_LINK
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#define _USE_MATH_DEFINES
#include <math.h>
#include <boost/timer/timer.hpp>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "reflow/core/grids/OneDimRegularSpaceGrid.h"
#include "reflow/core/grids/OneDimData.h"
#include "reflow/sddp/LocalConstRegressionForSDDPGeners.h"
#include "reflow/sddp/LocalLinearRegressionForSDDPGeners.h"
#include "reflow/sddp/SDDPFinalCut.h"
#include "reflow/sddp/SDDPLocalCut.h"
#include "reflow/sddp/backwardForwardSDDP.h"
#include "test/c++/tools/simulators/SimulatorGaussianSDDP.h"
#include "test/c++/tools/sddp/OptimizeStorageWithDemandSDDP.h"

using namespace std;
using namespace Eigen ;
using namespace reflow;


// #if defined   __linux
// #include <fenv.h>
// #define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
// #endif


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

/// Optimize a set of stock of water with a constraint on demand.
/// When demand not fulfilled by water, energy is bought on the market
/// Inflows and demand are represented by AR1 model (mean reverting)
/// \f$  x^{n+1} = \alpha (x^n - \mu_n) + \sigma g +\mu_{n+1}\f$
/// where $g$ is a centred unit Gaussian variable.
// The prices are deterministic.
template< class  LocalRegressionForSDDP>
void testStorageDemandSDDP(const int &p_nbStorage, const int &p_iterMax, const int &p_nbSimul,
                           const int &p_sampleCheck, const double p_accuracyClose, const int &p_nstepIterations,
                           const double &p_sigF,  const double &p_sigD)
{
#ifdef USE_MPI
    boost::mpi::communicator world;
#endif

    double maturity = 40;
    int nstep = 40;

    // optimizer
    //************
    double initLevel = 0.;//maturity / 10.; // initial level of the stock
    double withdrawalRate = 2 ; // on each time step volume that can be withdrawn
    double sigF = p_sigF ; //volatility for inflows
    double meanF = 0.1 ; // mean reverting for inflows
    double sigD = p_sigD * p_nbStorage ; // volatility of demand
    double meanD = 0.2 ; // mean reverting demand
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
    ArrayXd initialState(2 * p_nbStorage + 1);
    initialState.head(p_nbStorage) = ArrayXd::Constant(p_nbStorage, initLevel);
    initialState.segment(p_nbStorage, p_nbStorage) = ArrayXd::Constant(p_nbStorage, (*inflowValues)[0]);
    initialState(2 * p_nbStorage) = (*demandValues)[0];


    /// final cut
    ArrayXXd finalCut =  ArrayXXd::Zero(2 + 2 * p_nbStorage, 1);
    SDDPFinalCut finCut(finalCut);

    // no regression here
    ArrayXi nbMesh;

    // number of samples in optimisation and simulation
    int sampleOptim = p_nbSimul ; // at each time step, each state, number do samples used (optimization)
    int sampleCheckSimul =  p_sampleCheck; // number of simulation to check convergence

    // backward and forward simulator ( two uncertainties times  nbsimul )
    int nbUncertainties = p_nbStorage + 1;
    shared_ptr<SimulatorGaussianSDDP> backSimulator = make_shared<SimulatorGaussianSDDP>(nbUncertainties, sampleOptim);
    shared_ptr<SimulatorGaussianSDDP> forSimulator = make_shared<SimulatorGaussianSDDP>(nbUncertainties);

    // define the storage
    shared_ptr<OptimizerSDDPBase >   optimizer = make_shared<OptimizeStorageWithDemandSDDP<SimulatorGaussianSDDP> > (withdrawalRate, p_nbStorage, sigF, meanF,   flowAver, sigD, meanD, demand, spot, backSimulator, forSimulator);

    // optimisation dates
    ArrayXd dates = ArrayXd::LinSpaced(nstep + 1, 0., maturity);

    // names for archive
    string nameRegressor = "RegressorStorage";
    string nameCut = "CutStorage";
    string nameVisitedStates = "VisitedStateStorage";

    // precision parameter
    int nIterMax =  p_iterMax;
    double accuracy = p_accuracyClose / 100.;
    int nstepIterations = p_nstepIterations ; //// check for convergence between nstepIterations step
    ostringstream stringStream;
    pair<double, double>  values = backwardForwardSDDP<LocalRegressionForSDDP>(optimizer,  sampleCheckSimul, initialState,
                                   finCut, dates,  nbMesh, nameRegressor, nameCut, nameVisitedStates, nIterMax,
                                   accuracy, nstepIterations, stringStream
#ifdef USE_MPI
                                   , world
#endif
                                                                              );

#ifdef USE_MPI
    if (world.rank() == 0)
#endif
    {
        cout << stringStream.str() << endl ;
        cout << " nbStorage "  << p_nbStorage <<  " Value Optim " <<  values.first << " and Simulation " << values.second << " Iteration " << nIterMax << endl ;
        BOOST_CHECK_CLOSE(values.first, values.second, p_accuracyClose);
    }

}


BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsSDDP1DDeterministic)
{
    boost::timer::auto_cpu_timer t;
    int ndim = 1; // number of storage
    int iterMax = 100; /// maximal number of iteration forward/backward
    int  nbSample = 1 ; // number of samples to calculate cut (backward)
    int  nbSampleCheck = 1 ; // number of samples in forward
    double  error    = 0.05 ; // percentage between optimization and simulation allowed
    int     nstep = 10 ; /// accuracy is checked every nstep iterations
    double sigF = 0.; /// vol for inflows
    double  sigD = 0. ; /// vol for demand
    testStorageDemandSDDP<LocalLinearRegressionForSDDP>(ndim, iterMax, nbSample, nbSampleCheck, error, nstep, sigF, sigD);
    testStorageDemandSDDP<LocalConstRegressionForSDDP>(ndim, iterMax, nbSample, nbSampleCheck, error, nstep, sigF, sigD);
}

BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsSDDP2DDeterministic)
{
    boost::timer::auto_cpu_timer t;
    int ndim = 2; // number of storage
    int iterMax = 100; /// maximal number of iteration forward/backward
    int  nbSample = 1 ;  // number of samples to calculate cut (backward)
    int nbSampleCheck = 1; // number of sample to check convergence
    double  error    = 0.5 ; // percentage between optimization and simulation allowed
    int     nstep = 10 ; /// accuracy is checked every nstep iterations
    double sigF = 0.; /// vol for inflows
    double  sigD = 0. ; /// vol for demand
    testStorageDemandSDDP<LocalLinearRegressionForSDDP>(ndim, iterMax, nbSample, nbSampleCheck, error, nstep, sigF, sigD);
    testStorageDemandSDDP<LocalConstRegressionForSDDP>(ndim, iterMax, nbSample, nbSampleCheck, error, nstep, sigF, sigD);
}



BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsSDDP1D)
{
    boost::timer::auto_cpu_timer t;
    int ndim = 1; // number of storage
    int iterMax = 100; /// maximal number of iteration forward/backward
    int  nbSample = 1000 ; // number of samples to calculate cut (backward)
    int nbSampleCheck = 8000; // number of sample to check convergence
    double  error    = 2. ; // percentage between optimization and simulation allowed
    int     nstep = 5 ; /// accuracy is checked every nstep iterations
    double sigF = 0.6; /// vol for inflows
    double  sigD = 0.8 ; /// vol for demand
    testStorageDemandSDDP<LocalLinearRegressionForSDDP>(ndim, iterMax, nbSample,  nbSampleCheck, error, nstep, sigF, sigD);
    testStorageDemandSDDP<LocalConstRegressionForSDDP>(ndim, iterMax, nbSample,  nbSampleCheck, error, nstep, sigF, sigD);
}

#ifdef USE_LONG_TEST
BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsSDDP2D)
{
    int ndim = 2; // number of storage
    int iterMax = 200; /// maximal number of iteration forward/backward
    int  nbSample = 1000 ;  // number of samples to calculate cut (backward)
    int nbSampleCheck = 8000; // number of sample to check convergence
    double  error    = 1. ; // percentage between optimization and simulation allowed
    int     nstep = 10 ; /// accuracy is checked every nstep iterations
    double sigF = 0.6; /// vol for inflows
    double  sigD = 0.8 ; /// vol for demand
    testStorageDemandSDDP<LocalLinearRegressionForSDDP>(ndim, iterMax, nbSample, nbSampleCheck, error, nstep, sigF, sigD);
    testStorageDemandSDDP<LocalConstRegressionForSDDP>(ndim, iterMax, nbSample, nbSampleCheck, error, nstep, sigF, sigD);
}

BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsSDDP5D)
{
    int ndim = 5; // number of storage
    int iterMax = 200; /// maximal number of iteration forward/backward
    int  nbSample = 1000 ;  // number of samples to calculate cut (backward)
    int nbSampleCheck = 10000; // number of sample to check convergence
    double  error    = 2. ; // percentage between optimization and simulation allowed
    int     nstep = 10; /// accuracy is checked every nstep iterations
    double sigF = 0.6 ; /// vol for inflows
    double  sigD = 0.8  ; /// vol for demand
    testStorageDemandSDDP<LocalLinearRegressionForSDDP>(ndim, iterMax, nbSample,  nbSampleCheck, error, nstep, sigF, sigD);
    testStorageDemandSDDP<LocalConstRegressionForSDDP>(ndim, iterMax, nbSample,  nbSampleCheck, error, nstep, sigF, sigD);
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
