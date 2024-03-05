// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#include <iostream>
#include <sstream>
#include <iostream>
#include <sstream>
#ifndef USE_MPI
#define BOOST_TEST_MODULE testStorageWithInflowsAndMarketSDDP
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
#include "libflow/core/grids/OneDimRegularSpaceGrid.h"
#include "libflow/core/grids/OneDimData.h"
#include "libflow/sddp/LocalLinearRegressionForSDDPGeners.h"
#include "libflow/sddp/LocalConstRegressionForSDDPGeners.h"
#include "libflow/sddp/SDDPFinalCut.h"
#include "libflow/sddp/SDDPLocalCut.h"
#include "libflow/sddp/backwardForwardSDDP.h"
#include "test/c++/tools/simulators/MeanRevertingSimulatorSDDP.h"
#include "test/c++/tools/sddp/OptimizeStorageWithDemandAndMarketSDDP.h"

using namespace std;
using namespace Eigen ;
using namespace libflow;

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
/// The market price is given by a mean reverting model
template< class LocalRegressionForSDDP >
void testStorageDemandAndMarketSDDP(const int &p_nbStorage, const int &p_iterMax, const int &p_nMesh,
                                    const int &p_simulReg,  const int &p_nbSample, const int &p_simulRegFor,
                                    const int &p_sampleCheck, const double p_accuracyClose, const int &p_nstepIterations,
                                    const double &p_sigF,  const double &p_sigD, const double &p_sigFuture)
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
    // values for future curve
    shared_ptr<vector< double > > futValues(new vector<double>(nstep + 1));
    for (int i = 0; i < nstep + 1; ++i)
        (*futValues)[i] = 50. + 20 * sin((M_PI * i * iPeriod) / nstep);
    // define the future curve
    shared_ptr<OneDimData<OneDimRegularSpaceGrid, double> > futureGrid(new OneDimData< OneDimRegularSpaceGrid, double> (timeGrid, futValues));
    // one dimensional factors
    int nDim = 1;
    VectorXd sigma = VectorXd::Constant(nDim, p_sigFuture);
    VectorXd mr = VectorXd::Constant(nDim, 0.29);


    // initial state
    ArrayXd initialState(2 * p_nbStorage + 1);
    initialState.head(p_nbStorage) = ArrayXd::Constant(p_nbStorage, initLevel);
    initialState.segment(p_nbStorage, p_nbStorage) = ArrayXd::Constant(p_nbStorage, (*inflowValues)[0]);
    initialState(2 * p_nbStorage) = (*demandValues)[0];


    /// final cut
    ArrayXXd finalCut =  ArrayXXd::Zero(2 + 2 * p_nbStorage, 1);
    SDDPFinalCut finCut(finalCut);

    // no regression here
    ArrayXi nbMesh = ArrayXi::Constant(nDim, p_nMesh);

    // number of samples in optimisation and simulation
    int sampleCheckSimul = p_sampleCheck ; // number of simulation to check convergence

    // backward and forward simulator ( two uncertainties times  nbsimul )
    int nbUncertainties = p_nbStorage + 1;
    // exercize dates
    ArrayXd dates = ArrayXd::LinSpaced(nstep + 1, 0., maturity);

    // a backward simulator
    shared_ptr<MeanRevertingSimulatorSDDP< OneDimData<OneDimRegularSpaceGrid, double> > > backSimulator = make_shared<MeanRevertingSimulatorSDDP< OneDimData<OneDimRegularSpaceGrid, double> > >(futureGrid, sigma, mr, dates, p_simulReg, nbUncertainties, p_nbSample);
    // a forward simulator
    shared_ptr<MeanRevertingSimulatorSDDP< OneDimData<OneDimRegularSpaceGrid, double> > >   forSimulator = make_shared<MeanRevertingSimulatorSDDP< OneDimData<OneDimRegularSpaceGrid, double> > >(futureGrid, sigma, mr, dates, p_simulRegFor, nbUncertainties);

    // define the storage
    shared_ptr<OptimizerSDDPBase >   optimizer = make_shared<OptimizeStorageWithDemandAndMarketSDDP< MeanRevertingSimulatorSDDP< OneDimData<OneDimRegularSpaceGrid, double> > > > (withdrawalRate, p_nbStorage, sigF, meanF,   flowAver, sigD, meanD, demand, backSimulator, forSimulator);

    // names for archive
    string nameRegressor = "RegressorStorageMarket";
    string nameCut = "CutStorageMarket";
    string nameVisitedStates = "VisitedStateStorageMarket";

    // precision parameter
    int nIterMax =  p_iterMax;
    double accuracy = p_accuracyClose / 100.;
    int nstepIterations = p_nstepIterations ; //// check for convergence between nstepIterations step
    ostringstream stringStream;
    pair<double, double>  values = backwardForwardSDDP<LocalRegressionForSDDP>(optimizer,   sampleCheckSimul, initialState,
                                   finCut, dates,  nbMesh, nameRegressor, nameCut, nameVisitedStates, nIterMax,
                                   accuracy, nstepIterations, stringStream,
#ifdef USE_MPI
                                   world,
#endif
                                   false);

#ifdef USE_MPI
    if (world.rank() == 0)
#endif
    {
        cout << stringStream.str() << endl ;
        cout << "Nb storage " <<  p_nbStorage << " Value Optim " <<  values.first << " and Simulation " << values.second << " Iteration " << nIterMax << endl ;
        BOOST_CHECK_CLOSE(values.first, values.second, p_accuracyClose);
    }

}


BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsAndMarketSDDP1DDeterministic)
{
    boost::timer::auto_cpu_timer t;
    int  ndim = 1; // number of storage
    int  iterMax = 100; /// maximal number of iteration forward/backward
    int  nMesh = 1 ; //number of mesh for regressions
    int  simulReg = 3 ; // number of simulation used for regressions
    int nbSampleReg = 1 ; // nb sample used for estimated expectations
    int  simulRegFor = 1 ; // number of simulations used in simulation part of SDDP for uncertainties treated in regression part
    int  nbSampleForCheck = 1 ;  // number of samples to calculate cut (backward)
    double  error    = 0.05 ; // percentage between optimization and simulation allowed
    int     nstep = 10 ; /// accuracy is checked every nstep iterations
    double  sigF = 0.; /// vol for inflows
    double  sigD = 0. ; /// vol for demand
    double  sigFuture = 0.00001 ; // vol future
    testStorageDemandAndMarketSDDP<LocalLinearRegressionForSDDP>(ndim, iterMax, nMesh, simulReg, nbSampleReg,  simulRegFor,  nbSampleForCheck, error, nstep, sigF, sigD, sigFuture);
    testStorageDemandAndMarketSDDP<LocalConstRegressionForSDDP>(ndim, iterMax, nMesh, simulReg, nbSampleReg,  simulRegFor,  nbSampleForCheck, error, nstep, sigF, sigD, sigFuture);
}

BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsAndMarketSDDP2DDeterministic)
{
    boost::timer::auto_cpu_timer t;
    int  ndim = 2; // number of storage
    int  iterMax = 100; /// maximal number of iteration forward/backward
    int  nMesh = 1 ; //number of mesh for regressions
    int  simulReg = 3 ; // number of simulation used for regressions
    int nbSampleReg = 1 ; // nb sample used for estimated expectations
    int  simulRegFor = 1 ; // number of simulations used in simulation part of SDDP for uncertainties treated in regression part
    int  nbSampleForCheck = 1 ;  // number of samples to calculate cut (backward)
    double  error    = 0.05 ; // percentage between optimization and simulation allowed
    int     nstep = 10 ; /// accuracy is checked every nstep iterations
    double  sigF = 0.; /// vol for inflows
    double  sigD = 0. ; /// vol for demand
    double  sigFuture = 0.00001 ; // vol future
    testStorageDemandAndMarketSDDP<LocalLinearRegressionForSDDP>(ndim, iterMax, nMesh, simulReg, nbSampleReg, simulRegFor,   nbSampleForCheck, error, nstep, sigF, sigD, sigFuture);
    testStorageDemandAndMarketSDDP<LocalConstRegressionForSDDP>(ndim, iterMax, nMesh, simulReg, nbSampleReg, simulRegFor,   nbSampleForCheck, error, nstep, sigF, sigD, sigFuture);
}


BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsAndMarketSDDP5DDeterministic)
{
    boost::timer::auto_cpu_timer t;
    int  ndim = 5; // number of storage
    int  iterMax = 100; /// maximal number of iteration forward/backward
    int  nMesh = 1 ; //number of mesh for regressions
    int  simulReg = 3 ; // number of simulation used for regressions
    int nbSampleReg = 1 ; // nb sample used for estimated expectations
    int  simulRegFor = 1 ; // number of simulations used in simulation part of SDDP for uncertainties treated in regression part
    int  nbSampleForCheck = 1 ;  // number of samples to calculate cut (backward)
    double  error    = 0.05 ; // percentage between optimization and simulation allowed
    int     nstep = 10 ; /// accuracy is checked every nstep iterations
    double  sigF = 0.; /// vol for inflows
    double  sigD = 0. ; /// vol for demand
    double  sigFuture = 0.00001 ; // vol future
    testStorageDemandAndMarketSDDP<LocalLinearRegressionForSDDP>(ndim, iterMax, nMesh, simulReg, nbSampleReg, simulRegFor, nbSampleForCheck, error, nstep, sigF, sigD, sigFuture);
    testStorageDemandAndMarketSDDP<LocalConstRegressionForSDDP>(ndim, iterMax, nMesh, simulReg, nbSampleReg, simulRegFor, nbSampleForCheck, error, nstep, sigF, sigD, sigFuture);
}

BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsAndMarketSDDP10DDeterministic)
{
    boost::timer::auto_cpu_timer t;
    int  ndim = 10; // number of storage
    int  iterMax = 200; /// maximal number of iteration forward/backward
    int  nMesh = 1 ; //number of mesh for regressions
    int  simulReg = 3 ; // number of simulation used for regressions
    int nbSampleReg = 1 ; // nb sample used for estimated expectations
    int  simulRegFor = 1 ; // number of simulations used in simulation part of SDDP for uncertainties treated in regression part
    int  nbSampleForCheck = 1 ;  // number of samples to calculate cut (backward)
    double  error    = 0.3 ; // percentage between optimization and simulation allowed
    int     nstep = 10 ; /// accuracy is checked every nstep iterations
    double  sigF = 0.; /// vol for inflows
    double  sigD = 0. ; /// vol for demand
    double  sigFuture = 0.00001 ; // vol future
    testStorageDemandAndMarketSDDP<LocalLinearRegressionForSDDP>(ndim, iterMax, nMesh, simulReg, nbSampleReg, simulRegFor, nbSampleForCheck, error, nstep, sigF, sigD, sigFuture);
    testStorageDemandAndMarketSDDP<LocalConstRegressionForSDDP>(ndim, iterMax, nMesh, simulReg, nbSampleReg, simulRegFor, nbSampleForCheck, error, nstep, sigF, sigD, sigFuture);
}

#ifdef USE_LONG_TEST


BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsAndMarketSDDP1D)
{
    boost::timer::auto_cpu_timer t;
    int  ndim = 1; // number of storage
    int  iterMax = 100; /// maximal number of iteration forward/backward
    int  nMesh = 2 ; //number of mesh for regressions
    int  simulReg = 200 ; // number of simulation used for regressions
    int nbSampleReg = 10 ; // nb sample used for estimated expectations
    int  simulRegFor = 5 ; // number of simulations used in simulation part of SDDP for uncertainties treated in regression part
    int  nbSampleForCheck = 1000 ; // number of samples in forward
    double  error    = 2. ; // percentage between optimization and simulation allowed
    int     nstep = 10 ; /// accuracy is checked every nstep iterations
    double  sigF = 0.3; /// vol for inflows
    double  sigD = 0.4 ; /// vol for demand
    double  sigFuture = 0.6 ; // vol future
    testStorageDemandAndMarketSDDP<LocalLinearRegressionForSDDP>(ndim, iterMax, nMesh, simulReg, nbSampleReg, simulRegFor,  nbSampleForCheck, error, nstep, sigF, sigD, sigFuture);
}



BOOST_AUTO_TEST_CASE(testSimpleStorageWithInflowsAndMarketSDDP5D)
{
    boost::timer::auto_cpu_timer t;
    int  ndim = 5; // number of storage
    int  iterMax = 100; /// maximal number of iteration forward/backward
    int  nMesh = 2 ; //number of mesh for regressions
    int  simulReg = 200 ; // number of simulation used for regressions
    int nbSampleReg = 10 ; // nb sample used for estimated expectations
    int  simulRegFor = 5 ; // number of simulations used in simulation part of SDDP for uncertainties treated in regression part
    int  nbSampleForCheck = 5000 ; // number of samples in forward
    double  error    = 2. ; // percentage between optimization and simulation allowed
    int     nstep = 10 ; /// accuracy is checked every nstep iterations
    double  sigF = 0.3; /// vol for inflows
    double  sigD = 0.4 ; /// vol for demand
    double  sigFuture = 0.6 ; // vol future
    testStorageDemandAndMarketSDDP<LocalLinearRegressionForSDDP>(ndim, iterMax, nMesh, simulReg,  nbSampleReg, simulRegFor,  nbSampleForCheck, error, nstep, sigF, sigD, sigFuture);
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
