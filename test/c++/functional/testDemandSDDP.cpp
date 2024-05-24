

#include <iostream>
#include <sstream>
#ifndef USE_MPI
#define BOOST_TEST_MODULE testDemandSDDP
#endif
#define BOOST_TEST_DYN_LINK
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#define _USE_MATH_DEFINES
#include <math.h>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "reflow/core/grids/OneDimRegularSpaceGrid.h"
#include "reflow/core/grids/OneDimData.h"
#include "reflow/sddp/LocalLinearRegressionForSDDPGeners.h"
#include "reflow/sddp/LocalConstRegressionForSDDPGeners.h"
#include "reflow/sddp/SDDPFinalCut.h"
#include "reflow/sddp/SDDPLocalCut.h"
#include "reflow/sddp/backwardForwardSDDP.h"
#include "test/c++/tools/simulators/SimulatorGaussianSDDP.h"
#include "test/c++/tools/sddp/OptimizeDemandSDDP.h"

using namespace std;
using namespace Eigen ;
using namespace reflow;

#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif

double accuracyClose =  1.;

///  fake optimization (see OptimizeDemandSDDP.h)
/// only calculate the expectancy of demand.
template< class LocalRegressionForSDDP >
void testDemandSDDP(const double   &p_sigD, const int   &p_sampleOptim, const int   &p_sampleCheckSimul)
{
#ifdef USE_MPI
    boost::mpi::communicator world;
#endif

    double maturity = 40;
    int nstep = 40;

    // optimizer
    //************
    double sigD = p_sigD ; //volatility for inflows
    double kappaD = 0.2;
    double spot = 3 ;

    // define a a time grid
    shared_ptr<OneDimRegularSpaceGrid> timeGrid(new OneDimRegularSpaceGrid(0., maturity / nstep, nstep));
    // periodicity factor
    int iPeriod = 52;
    // define average demand
    shared_ptr<vector< double > > demandValues(new vector<double>(nstep + 1));
    for (int i = 0; i < nstep + 1; ++i)
        (*demandValues)[i] = (2. + 0.4 * cos((M_PI * i * iPeriod) / nstep)) ;
    shared_ptr<OneDimData<OneDimRegularSpaceGrid, double> > demand(new OneDimData< OneDimRegularSpaceGrid, double> (timeGrid, demandValues));


    // initial state
    ArrayXd initialState = ArrayXd::Constant(1, demand->get(0));

    /// final cut
    ArrayXXd finalCut =  ArrayXXd::Zero(2, 1);
    SDDPFinalCut finCut(finalCut);

    // no regression here
    ArrayXi nbMesh;

    // number of samples in optimisation and simulation
    int sampleOptim = p_sampleOptim ; // at each time step, each state, number do samples used (optimization)
    int sampleCheckSimul = p_sampleCheckSimul ; // number of simulation to check convergence

    // backward and forward simulator ( one uncertainty times  nbsimul )
    int nbUncertainties = 1;
    shared_ptr<SimulatorGaussianSDDP> backSimulator = make_shared<SimulatorGaussianSDDP>(nbUncertainties, sampleOptim);
    shared_ptr<SimulatorGaussianSDDP> forSimulator  =  make_shared<SimulatorGaussianSDDP>(nbUncertainties);


    // define the storage
    shared_ptr<OptimizerSDDPBase >   optimizer = make_shared<OptimizeDemandSDDP<SimulatorGaussianSDDP> > (sigD, kappaD,  demand, spot, backSimulator, forSimulator);

    // optimisation dates
    ArrayXd dates = ArrayXd::LinSpaced(nstep + 1, 0., maturity);

    // names for archive
    string nameRegressor = "RegressorDemand";
    string nameCut = "CutDemand";
    string nameVisitedStates = "VisitedStateDemand";

    // precision parameter
    int nIterMax = 40;
    double accuracy = accuracyClose / 100;
    int nstepIterations = 4; // check for convergence between nstepIterations step
    ostringstream stringStream;  // store intermediate results
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
        cout << " Value Optim " <<  values.first << " and Simulation " << values.second << " Iteration " << nIterMax << endl ;
        BOOST_CHECK_CLOSE(values.first, values.second, accuracyClose);
    }

}




BOOST_AUTO_TEST_CASE(testDemandSDDP1DDeterministic)
{

    double sig = 0. ;
    int sampleOptim = 1;
    int sampleCheckSimul = 1;
    testDemandSDDP<LocalLinearRegressionForSDDP>(sig, sampleOptim, sampleCheckSimul);
    testDemandSDDP<LocalConstRegressionForSDDP>(sig, sampleOptim, sampleCheckSimul);
}

BOOST_AUTO_TEST_CASE(testDemandSDDP1D)
{
    double sig = 0.6 ;
    int sampleOptim = 500;
    int sampleCheckSimul = 1000;
    testDemandSDDP<LocalLinearRegressionForSDDP>(sig, sampleOptim, sampleCheckSimul);
    testDemandSDDP<LocalConstRegressionForSDDP>(sig, sampleOptim, sampleCheckSimul);
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
