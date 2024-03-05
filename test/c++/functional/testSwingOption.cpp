// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef USE_MPI
#define BOOST_TEST_MODULE testSwingOption
#endif
#define BOOST_TEST_DYN_LINK
#include <functional>
#include <stdint.h>
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <array>
#include <memory>
#include <boost/test/unit_test.hpp>
#include <boost/tuple/tuple.hpp>
#include <Eigen/Dense>
#include "libflow/regression/LocalLinearRegressionGeners.h"
#include "libflow/core/utils/comparisonUtils.h"
#include "libflow/core/grids/GridIterator.h"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/regression/LocalLinearRegression.h"
#include "libflow/regression/LocalGridKernelRegression.h"
#include "libflow/regression/ContinuationValue.h"
#include "test/c++/tools/EuropeanOptions.h"
#include "test/c++/tools/simulators/BlackScholesSimulator.h"
#include "test/c++/tools/BasketOptions.h"
#include "test/c++/tools/dp/OptimizeSwing.h"
#include "test/c++/tools/dp/OptimizeFictitiousSwing.h"
#include "test/c++/tools/dp/FinalValueFunction.h"
#include "test/c++/tools/dp/FinalValueFictitiousFunction.h"
#include "test/c++/tools/dp/DynamicProgrammingByRegression.h"
#ifdef USE_MPI
#include "test/c++/tools/dp/DynamicProgrammingByRegressionDist.h"
#endif

double accuracyClose = 0.2;
double accuracyNearlyEqual = 0.05;

/** \file testSwingOption.cpp
 *  \brief Test the framework for swing options
 *   The payoff  is of type call \f$ q (S- K)\f$ where \f$q\$f is the quantity exercized
 *   N exercises are possible at most between a set of dates
 *   The state is  given by the quantity already exercized and the asset value.
 *   In this case a reference solution is easily given in term of the sum of call options.
 *   A first numerical solution is given without parallelism
 *   A second numerical solution is given using the continuation object
 *    A third numerical solution is given using the parallelism framework distributing grids on processors
 *    A fourth numerical solution is given using the parallelism framework with parallelization and threads without distributing data
 *   At last two test cases in dimension 2 and 3 for the stocks are achieved
 */
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

/// \brief Analytical value
///  \param p_S        asset value
///  \param p_sigma   volatility
///  \param p_r       interest rate
///  \param p_strike  strike
///  \param p_dates     possible exercise dates
///  \return option value
double analyticalValue(const int N, const double &p_S, const double &p_sigma, const double &p_r, const double   &p_strike,
                       const ArrayXd &p_dates)
{
    double analytical = 0.;
    for (int i = p_dates.size() - N; i < p_dates.size(); ++i)
        analytical += CallOption()(p_S, p_sigma,  p_r, p_strike,  p_dates(i));
    return analytical;
}

/// \brief Classical resolution for swing
/// \param p_sim      Monte Carlo simulator
/// \param p_payOff   Option pay off
/// \param p_regressor  regressor object
/// \param p_dates    possible exercise dates
/// \param p_N          number of exercises
template < class Simulator, class PayOff, class Regressor >
double resolutionSwing(Simulator &p_sim, const PayOff &p_payOff, Regressor &p_regressor, const ArrayXd &p_dates, const int &p_N)
{
    assert((p_sim.getNbStep() + 1) == p_dates.size());

    // asset simulated under the neutral risk probability : get the trend of first asset to get interest rate
    // in this example the step between two exercises is given
    double expRate = p_sim.getActuStep();

    // final payoff
    VectorXd finalPayOff(p_payOff.applyVec(p_sim.getParticles()));
    // Terminal function depending on simulations and stock already exercised
    shared_ptr<MatrixXd> cashNext = make_shared<MatrixXd>(finalPayOff.size(), p_N);
    for (int is = 0; is < finalPayOff.size(); ++is)
        cashNext->row(is) = VectorXd::Constant(p_N, finalPayOff(is)).transpose();
    shared_ptr<MatrixXd> cashPrev = make_shared<MatrixXd>(finalPayOff.size(), p_N);
    for (int iStep = p_dates.size() - 2; iStep >= 0; --iStep)
    {
        ArrayXXd asset = p_sim.stepBackwardAndGetParticles();
        VectorXd payOffLoc = p_payOff.applyVec(asset);
        // conditional expectation
        p_regressor.updateSimulations(((iStep == 0) ? true : false), asset);

        // store conditional expectations
        vector< shared_ptr< VectorXd > > vecCondEspec(p_N);
        for (int iStock = 0 ; iStock < p_N; ++iStock)
        {
            vecCondEspec[iStock] = make_shared<VectorXd>(p_regressor.getAllSimulations(cashNext->col(iStock)) * expRate);
        }

        // arbitrage
        for (int iStock = 0 ; iStock < p_N - 1; ++iStock)
            cashPrev->col(iStock) = (payOffLoc.array() + vecCondEspec[iStock + 1]->array() >  vecCondEspec[iStock]->array()).select(payOffLoc + expRate * cashNext->col(iStock + 1),
                                    expRate * cashNext->col(iStock));
        // last stock
        cashPrev->col(p_N - 1) = (payOffLoc.array() >  vecCondEspec[p_N - 1]->array()).select(payOffLoc, expRate * cashNext->col(p_N - 1));
        // switch pointer
        shared_ptr<MatrixXd> tempVec = cashNext;
        cashNext = cashPrev;
        cashPrev = tempVec;
    }
    return cashNext->col(0).mean();
}


/// \brief Same resolution using  Continuation Object to deal with stocks
/// \param p_sim      Monte Carlo simulator
/// \param p_payOff   Option pay off
/// \param p_regressor  regressor object
/// \param p_dates    possible exercise dates
/// \param p_N        number of exercises
template < class Simulator, class PayOff, class RegressorPt >
double resolutionSwingContinuation(Simulator &p_sim, const PayOff &p_payOff, RegressorPt &p_regressor, const ArrayXd &p_dates, const int &p_N)
{
    assert((p_sim.getNbStep() + 1) == p_dates.size());
    // asset simulated under the neutral risk probability : get the trend of first asset to get interest rate
    // in this example the step between two exercises is given
    double expRate = p_sim.getActuStep();
    // regular grid
    ArrayXd lowValues(1), step(1);
    lowValues(0) = 0. ;
    step(0) = 1;
    ArrayXi  nbStep(1);
    nbStep(0) = p_N - 1;
    shared_ptr< RegularSpaceGrid > regular = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
    // extremal values of the grid
    vector <array< double, 2>  > extremeGrid = regular->getExtremeValues();

    // final payoff
    VectorXd finalPayOff(p_payOff.applyVec(p_sim.getParticles()));
    // Terminal function depending on simulations and stock already exercised
    shared_ptr<ArrayXXd> cashNext = make_shared<ArrayXXd>(finalPayOff.size(), regular->getNbPoints());
    for (int is = 0; is < finalPayOff.size(); ++is)
        cashNext->row(is) = ArrayXd::Constant(regular->getNbPoints(), finalPayOff(is)).transpose();
    shared_ptr<ArrayXXd> cashPrev = make_shared<ArrayXXd>(finalPayOff.size(), regular->getNbPoints());
    for (int iStep = p_dates.size() - 2; iStep >= 0; --iStep)
    {
        ArrayXXd asset = p_sim.stepBackwardAndGetParticles();
        ArrayXd payOffLoc = p_payOff.applyVec(asset).array();
        // conditional expectation
        p_regressor->updateSimulations(((iStep == 0) ? true : false), asset);
        // continuation value object dealing with stocks
        ContinuationValue continuation(regular, p_regressor, *cashNext);
        // iterator on grid points
        shared_ptr<GridIterator> iterOnGrid = regular->getGridIterator();
        while (iterOnGrid->isValid())
        {
            ArrayXd CoordStock = iterOnGrid->getCoordinate();
            // use continuation to get realization of condition expectation
            ArrayXd conditionExpecCur = expRate * continuation.getAllSimulations(CoordStock);
            if (isLesserOrEqual(CoordStock(0) + 1, extremeGrid[0][1]))
            {
                ArrayXd conditionExpecNext = expRate * continuation.getAllSimulations(CoordStock + 1);
                cashPrev->col(iterOnGrid->getCount()) = (payOffLoc + conditionExpecNext >  conditionExpecCur).select(payOffLoc + expRate * cashNext->col(iterOnGrid->getCount() + 1),
                                                        expRate * cashNext->col(iterOnGrid->getCount()));
            }
            else
            {
                cashPrev->col(iterOnGrid->getCount()) = (payOffLoc >  conditionExpecCur).select(payOffLoc,	expRate * cashNext->col(iterOnGrid->getCount()));
            }
            iterOnGrid->next();
        }
        // switch pointer
        shared_ptr<ArrayXXd> tempVec = cashNext;
        cashNext = cashPrev;
        cashPrev = tempVec;
    }
    return cashNext->col(0).mean();
}




BOOST_AUTO_TEST_CASE(testSwingOptionInOptimization)
{
#ifdef USE_MPI
    boost::mpi::communicator world;
    world.barrier();
#endif
    VectorXd initialValues = ArrayXd::Constant(1, 1.);
    VectorXd sigma  = ArrayXd::Constant(1, 0.2);
    VectorXd mu  = ArrayXd::Constant(1, 0.05);
    MatrixXd corr = MatrixXd::Ones(1, 1);
    // number of step
    int nStep = 30;
    // exercise date
    double T = 1. ;
    ArrayXd dates = ArrayXd::LinSpaced(nStep + 1, 0., T);
    int N = 3 ; // 3  exercise dates
    double strike = 1.;
    int nbSimul = 200000;
    int nMesh = 16;
    // payoff
    BasketCall  payoff(strike);
    // analytical
#ifdef USE_MPI
    double analytical = ((world.rank() == 0) ? analyticalValue(N, initialValues(0), sigma(0), mu(0), strike, dates) : 0);
#else
    double analytical =  analyticalValue(N, initialValues(0), sigma(0), mu(0), strike, dates) ;
#endif

    // store sequential
    double valueSeq ;
    // mesh
    ArrayXi nbMesh = ArrayXi::Constant(1, nMesh);
#ifdef USE_MPI
    if (world.rank() == 0)
#endif
    {
        // simulator
        BlackScholesSimulator simulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false);
        // regressor
        LocalLinearRegression regressor(nbMesh);
        // Bermudean value
        valueSeq = resolutionSwing(simulator, payoff, regressor, dates, N);
        BOOST_CHECK_CLOSE(valueSeq, analytical, accuracyClose);
    }
#ifdef USE_MPI
    if (world.rank() == 0)
#endif
    {
        // simulator
        BlackScholesSimulator simulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false);
        // regressor
        shared_ptr< LocalLinearRegression > regressor(new LocalLinearRegression(nbMesh));
        // using continuation values
        double valueSeqContinuation   = resolutionSwingContinuation(simulator, payoff, regressor, dates, N);
        BOOST_CHECK_EQUAL(valueSeq, valueSeqContinuation);
    }
#ifdef USE_MPI
    world.barrier();
#endif
    // simulator
    shared_ptr<BlackScholesSimulator>  simulator(new BlackScholesSimulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false)) ;
    // grid
    ArrayXd lowValues = ArrayXd::Constant(1, 0.);
    ArrayXd step = ArrayXd::Constant(1, 1.);
    ArrayXi nbStep = ArrayXi::Constant(1, N);
    shared_ptr<FullGrid> grid = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
    // final value
    function<double(const int &, const ArrayXd &, const ArrayXd &)>  vFunction = FinalValueFunction<BasketCall>(payoff, N);
    // optimizer
    shared_ptr< OptimizeSwing<BasketCall, BlackScholesSimulator> >optimizer = make_shared< OptimizeSwing<BasketCall, BlackScholesSimulator> >(payoff, N);
    // initial values
    ArrayXd initialStock = ArrayXd::Constant(1, 0.);
    int initialRegime = 0;
    string fileToDumpDistr = "CondExpDistr";
#ifdef USE_MPI
    fileToDumpDistr = fileToDumpDistr + to_string(world.size());

    bool bOneFile = false;
    // regressor
    shared_ptr<BaseRegression > regressor(new LocalLinearRegression(nbMesh));
    // link the simulations to the optimizer
    optimizer->setSimulator(simulator);
    double valueParal = DynamicProgrammingByRegressionDist(grid, optimizer, regressor, vFunction, initialStock, initialRegime, fileToDumpDistr, bOneFile, world);
    if (world.rank() == 0)
    {
        BOOST_CHECK_EQUAL(valueSeq, valueParal);
    }
    world.barrier();
#else
    string fileToDump = "CondExp";
    // simulator
    shared_ptr<BlackScholesSimulator>  simulator1(new BlackScholesSimulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false));
    // regressor
    shared_ptr< LocalLinearRegression > regressor1(new LocalLinearRegression(nbMesh));
    // link the simulations to the optimizer
    optimizer->setSimulator(simulator1);
    // using continuation values
    double value =  DynamicProgrammingByRegression(grid, optimizer, regressor1, vFunction, initialStock, initialRegime, fileToDump);
    BOOST_CHECK_EQUAL(valueSeq, value);
#endif
}
// only if 64 bits
#if UINTPTR_MAX == 0xffffffffffffffff
BOOST_AUTO_TEST_CASE(testSwingOptionInOptimizationKernel)
{
#ifdef USE_MPI
    boost::mpi::communicator world;
    world.barrier();
#endif
    VectorXd initialValues = ArrayXd::Constant(1, 1.);
    VectorXd sigma  = ArrayXd::Constant(1, 0.2);
    VectorXd mu  = ArrayXd::Constant(1, 0.05);
    MatrixXd corr = MatrixXd::Ones(1, 1);
    // number of step
    int nStep = 30;
    // exercise date
    double T = 1. ;
    ArrayXd dates = ArrayXd::LinSpaced(nStep + 1, 0., T);
    int N = 3 ; // 3  exercise dates
    double strike = 1.;
    int nbSimul = 100000;
    double bandWidth = 0.15;
    // payoff
    BasketCall  payoff(strike);
    // analytical
#ifdef USE_MPI
    double analytical = ((world.rank() == 0) ? analyticalValue(N, initialValues(0), sigma(0), mu(0), strike, dates) : 0);
#else
    double analytical =  analyticalValue(N, initialValues(0), sigma(0), mu(0), strike, dates) ;
#endif

    // store sequential
    double valueSeq ;

#ifdef USE_MPI
    world.barrier();
    if (world.rank() == 0)
#endif
    {
        // simulator
        BlackScholesSimulator simulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false);
        // regressor
        LocalGridKernelRegression regressor(bandWidth, 1., true);
        // Bermudean value
        valueSeq = resolutionSwing(simulator, payoff, regressor, dates, N);
        BOOST_CHECK_CLOSE(valueSeq, analytical, accuracyClose);
        cout << " analytical " << analytical << " valueSeq " << valueSeq << std::endl ;
    }
#ifdef USE_MPI
    world.barrier();
    if (world.rank() == 0)
#endif
    {
        // simulator
        BlackScholesSimulator simulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false);
        // regressor
        shared_ptr< LocalGridKernelRegression > regressor = make_shared<LocalGridKernelRegression>(bandWidth, 1., true);
        // using continuation values
        double valueSeqContinuation   = resolutionSwingContinuation(simulator, payoff, regressor, dates, N);
        cout << " analytical " << analytical << " Continuation " << valueSeqContinuation << endl ;
        BOOST_CHECK_EQUAL(valueSeq, valueSeqContinuation);
    }
    // simulator
    shared_ptr<BlackScholesSimulator>  simulator(new BlackScholesSimulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false)) ;
    // grid
    ArrayXd lowValues = ArrayXd::Constant(1, 0.);
    ArrayXd step = ArrayXd::Constant(1, 1.);
    ArrayXi nbStep = ArrayXi::Constant(1, N);
    shared_ptr<FullGrid> grid = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
    // final value
    function<double(const int &, const ArrayXd &, const ArrayXd &)>  vFunction = FinalValueFunction<BasketCall>(payoff, N);
    // optimizer
    shared_ptr< OptimizeSwing<BasketCall, BlackScholesSimulator> >optimizer = make_shared< OptimizeSwing<BasketCall, BlackScholesSimulator> >(payoff, N);
    // initial values
    ArrayXd initialStock = ArrayXd::Constant(1, 0.);
    int initialRegime = 0;
    string fileToDumpDistr = "CondExpDistr";
#ifdef USE_MPI
    world.barrier();
    fileToDumpDistr = fileToDumpDistr + to_string(world.size());
    bool bOneFile = false;
    // regressor
    shared_ptr<BaseRegression > regressor = make_shared<LocalGridKernelRegression>(bandWidth, 1., true);
    // link the simulations to the optimizer
    optimizer->setSimulator(simulator);
    double valueParal = DynamicProgrammingByRegressionDist(grid, optimizer, regressor, vFunction, initialStock, initialRegime, fileToDumpDistr, bOneFile, world);
    cout << " By framework" <<  valueParal << " Analytical " << analytical << endl ;
    if (world.rank() == 0)
    {
        BOOST_CHECK_EQUAL(valueSeq, valueParal);
    }
#else
    string fileToDump = "CondExp";
    // simulator
    shared_ptr<BlackScholesSimulator>  simulator1(new BlackScholesSimulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false));
    // regressor
    shared_ptr<BaseRegression > regressor = make_shared<LocalGridKernelRegression>(bandWidth, 1., true);
    // link the simulations to the optimizer
    optimizer->setSimulator(simulator1);
    // using continuation values
    double value =  DynamicProgrammingByRegression(grid, optimizer, regressor, vFunction, initialStock, initialRegime, fileToDump);
    BOOST_CHECK_EQUAL(valueSeq, value);
#endif
}
// only if 64 bits
#endif

#ifdef USE_MPI
/// \brief function to test stock in dimension above 1
void testMultiStock(const int p_ndim)
{
    boost::mpi::communicator world;
    world.barrier();
    VectorXd initialValues = ArrayXd::Constant(1, 1.);
    VectorXd sigma  = ArrayXd::Constant(1, 0.2);
    VectorXd mu  = ArrayXd::Constant(1, 0.05);
    MatrixXd corr = MatrixXd::Ones(1, 1);
    // number of step
    int nStep = 20;
    // exercise date
    double T = 1. ;
    ArrayXd dates = ArrayXd::LinSpaced(nStep + 1, 0., T);
    // exercise dates
    int N = 3 ;
    double strike = 1.;
    int nbSimul = 10000;
    int nMesh = 4;
    // payoff
    BasketCall  payoff(strike);
    // store sequential
    double valueSeq = 0;
    // mesh
    ArrayXi nbMesh = ArrayXi::Constant(1, nMesh);
    if (world.rank() == 0)
    {
        // simulator
        BlackScholesSimulator simulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false);
        // regressor
        LocalLinearRegression regressor(nbMesh);
        // Bermudean value
        valueSeq = resolutionSwing(simulator, payoff, regressor, dates, N);
    }
    world.barrier();
    // simulator
    shared_ptr<BlackScholesSimulator>  simulator(new BlackScholesSimulator(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false));
    // grid
    ArrayXd lowValues = ArrayXd::Constant(p_ndim, 0.);
    ArrayXd step = ArrayXd::Constant(p_ndim, 1.);
    // the stock is discretized with values from 0 to N included
    ArrayXi nbStep = ArrayXi::Constant(p_ndim, N);
    shared_ptr<FullGrid> grid = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
    // final value
    function<double(const int &, const ArrayXd &, const ArrayXd &)>   vFunction = FinalValueFictitiousFunction<BasketCall>(payoff, N);
    // optimizer
    shared_ptr< OptimizeFictitiousSwing<BasketCall, BlackScholesSimulator> >optimizer = make_shared< OptimizeFictitiousSwing<BasketCall, BlackScholesSimulator> >(payoff, N, p_ndim);
    // initial values
    ArrayXd initialStock = ArrayXd::Constant(p_ndim, 0.);
    int initialRegime = 0;
    string fileToDump = "CondExpForMPI" + to_string(world.size());
    // regressor
    shared_ptr< BaseRegression > regressor(new LocalLinearRegression(nbMesh));
    bool bOneFile = false;
    // link the simulations to the optimizer
    optimizer->setSimulator(simulator);
    double  valueParal = DynamicProgrammingByRegressionDist(grid, optimizer, regressor, vFunction, initialStock, initialRegime, fileToDump, bOneFile, world);
    if (world.rank() == 0)
    {
        BOOST_CHECK_CLOSE(valueSeq * p_ndim, valueParal, accuracyNearlyEqual);
    }
    world.barrier();
}

BOOST_AUTO_TEST_CASE(testSwingOption2D)
{
    testMultiStock(2);
}

BOOST_AUTO_TEST_CASE(testSwingOption3D)
{
    testMultiStock(3);
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
#endif
