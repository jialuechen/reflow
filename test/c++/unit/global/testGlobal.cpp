// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#ifndef USE_MPI
#define BOOST_TEST_MODULE testGlobal
#endif
#define BOOST_TEST_DYN_LINK
#ifdef USE_MPI
#include <boost/mpi.hpp>
#endif
#include <fstream>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include <cmath>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "libflow/regression/LocalLinearRegression.h"
#include "libflow/core/grids/RegularSpaceGrid.h"
#include "libflow/core/utils/StateWithStocks.h"
#include "libflow/dp/TransitionStepRegressionDP.h"
#include "libflow/dp/FinalStepDP.h"
#include "libflow/dp/SimulateStepRegression.h"
#include "test/c++/tools/BasketOptions.h"
#include "test/c++/tools/dp/OptimizeSwing.h"
#include "test/c++/tools/dp/FinalValueFunction.h"
#include "test/c++/tools/simulators/BlackScholesSimulator.h"

using namespace std;
using namespace Eigen;
using namespace gs;
using namespace libflow;


/// For Clang < 3.7 (and above ?) to be compatible GCC 5.1 and above
namespace boost
{
namespace unit_test
{
namespace ut_detail
{
std::string normalize_test_case_name(const_string name)
{
    return (name[0] == '&' ? std::string(name.begin() + 1, name.size() - 1) : std::string(name.begin(), name.size()));
}
}
}
}

double accuracyEqual = 1e-10;

#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif

/// test  global components parts for dynamic programming
BOOST_AUTO_TEST_CASE(testGlocalCalc)
{

#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

#ifdef USE_MPI
    boost::mpi::communicator world;
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
    int N = 3 ; // 3  exercises dates
    double strike = 1.;
    int nbSimul = 10;
    // payoff
    BasketCall  payoff(strike);
    // regular grid
    Eigen::ArrayXd lowValues(1), step(1);
    lowValues(0) = 0. ;
    step(0) = 1;
    Eigen::ArrayXi  nbStep(1);
    nbStep(0) = N - 1;
    shared_ptr< RegularSpaceGrid > regular = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
    // final value
    std::function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>  vFunction = FinalValueFunction<BasketCall>(payoff, N);
    // optimizer
    shared_ptr< OptimizeSwing<BasketCall, BlackScholesSimulator> >optimizer = make_shared< OptimizeSwing<BasketCall, BlackScholesSimulator> >(payoff, N);


    // optimization
    //////////////
    {
        // simulator
        shared_ptr<BlackScholesSimulator>  simulator = make_shared< BlackScholesSimulator>(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, false) ;

        // affect simulator to optimizer
        optimizer->setSimulator(simulator);

        // mesh
        int nMesh = 1;
        ArrayXi nbMesh = ArrayXi::Constant(1, nMesh);
        // regressor
        shared_ptr<BaseRegression > regressor = make_shared<LocalLinearRegression>(nbMesh);
        // to dump
        std::shared_ptr<gs::BinaryFileArchive> ar = std::make_shared<gs::BinaryFileArchive>("ToDump", "w");
        // final values
        std::vector< shared_ptr< Eigen::ArrayXXd > >  values = FinalStepDP(regular, optimizer->getNbRegime())(vFunction, simulator->getParticles().array());
        // back
        Eigen::ArrayXXd asset = simulator->stepBackwardAndGetParticles();
        regressor->updateSimulations(0, asset);
        // transition step
        TransitionStepRegressionDP transStep(regular, regular, optimizer
#ifdef USE_MPI
                                             , world
#endif
                                            );
        auto  valuesNext = transStep.oneStep(values, regressor);
        // dump continuation values
        transStep.dumpContinuationValues(ar, "CONT", 1, valuesNext.first, valuesNext.second, regressor);
    }

#ifdef USE_MPI
    world.barrier();
#endif

    // simulation
    /////////////
    {
        // simulator
        shared_ptr<BlackScholesSimulator>  simulator = make_shared< BlackScholesSimulator>(initialValues, sigma, mu, corr, dates(dates.size() - 1), dates.size() - 1, nbSimul, true) ;

        // affect simulator to optimizer
        optimizer->setSimulator(simulator);

        std::cout << "SIMULATION" << std::endl ;
        VectorXd pointStock = ArrayXd::Constant(1, 1.);
        // to dump
        gs::BinaryFileArchive ar("ToDump", "r");
        std::vector< StateWithStocks> states;
        states.reserve(10);
        for (int is = 0; is < nbSimul; ++is)
            states.push_back(StateWithStocks(0, pointStock, Eigen::ArrayXd::Zero(1)));
        // cost function
        Eigen::ArrayXXd costFunction = Eigen::ArrayXXd::Zero(1, 10);
        SimulateStepRegression(ar, 1, "CONT", regular, optimizer
#ifdef USE_MPI
                               , world
#endif
                              ).oneStep(states, costFunction);
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


