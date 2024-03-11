// Copyright (C) 2016 Fime

#ifndef USE_MPI
#define BOOST_TEST_MODULE testSemiLagrangCase3
#endif
#define BOOST_TEST_DYN_LINK
#include <math.h>
#include <memory>
#include <functional>
#include <fstream>
#ifdef USE_MPI
#include "boost/mpi.hpp"
#endif
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/core/grids/SpaceGridGeners.h"
#include "libflow/core/grids/FullGridGeners.h"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/core/grids/RegularLegendreGridGeners.h"
#include "libflow/core/grids/RegularSpaceGrid.h"
#include "libflow/core/grids/RegularLegendreGrid.h"
#include "test/c++/tools/semilagrangien/OptimizeSLCase3.h"
#include "test/c++/tools/semilagrangien/semiLagrangianTime.h"
#include "test/c++/tools/NormalCumulativeDistribution.h"
#include "test/c++/tools/InvNormalCumulativeDistribution.h"
#include "test/c++/tools/semilagrangien/semiLagrangianSimu.h"
#include "test/c++/tools/semilagrangien/semiLagrangianSimuControl.h"

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


// problem solution
class Solution
{

    double m_muSig ;

public :

    Solution(const double &p_mu, const double &p_sig): m_muSig(p_mu / p_sig) {};

    double operator()(const double t, const ArrayXd &x) const
    {
        if (almostEqual(x(0), 0., 10))
            return 0;
        if (almostEqual(x(0), 1., 10))
            return 1;
        return NormalCumulativeDistribution()(InvNormalCumulativeDistribution()(x(0)) - m_muSig * sqrt(t)) ;
    }
    double operator()(const double t, const int &, const ArrayXd &x) const
    {
        return operator()(t, x);
    }
};



/// Optimize semi Lagrangian
/// Stochastic target case: see Choukroun, Elie, Warin
/// The target here is  \$1\f$ in weath.
/// \param p_ndt    number of time steps
/// \param p_grid   interpolation grid
/// \param p_proba  target probability
/// \return max error on the mesh and the simulation result for probability
pair<double, double>  testCase3(const int &p_ndt,  const  shared_ptr<SpaceGrid>   &p_grid, double p_proba)
{
#ifdef USE_MPI
    boost::mpi::communicator world;
#endif
    // the problem
    double sig = 0.2;
    double mu = 0.1 ;
    double T = 1;

    // steps
    double dt = T / p_ndt;
    double alphaMax = 2. ;
    double alphaStep = 0.025;
    // Create optimizer
    shared_ptr<OptimizerSLBase> optimizer = make_shared<OptimizerSLCase3>(mu, sig, dt, alphaMax, alphaStep);

    // initial function value See Bouchard Elie Touzi
    auto f([](const int &, const ArrayXd & x)
    {
        return x(0);
    });
    function<double(const int &, const ArrayXd &)>  initialVal(cref(f));

    // solution
    function<double(const double &, const ArrayXd &)> solution = Solution(mu, sig);

    // boundary
    function<double(const double &,  const int &, const ArrayXd &)> boundary = Solution(mu, sig);

    // point for interpolation
    ArrayXd point = ArrayXd::Constant(1, 0.);
    // only one regime
    int initRegime = 0 ;
    // file to dump
    string fileToDump = "DumpLG3";
    // error at point
    pair<double, double> valAndError = semiLagrangianTime(p_grid, optimizer, initialVal, boundary, dt, p_ndt, point, initRegime, solution, fileToDump
#ifdef USE_MPI
                                       , world
#endif
                                                         );
    cout << " Errmax " <<  valAndError.second << endl ;

    // in simulation send back the probability reached after simulation (should be 0 or 1)
    auto fSimu([](const int &, const Eigen::ArrayXd & x)
    {
        return  x(0);
    });
    function<double(const int &, const Eigen::ArrayXd &)>  finalFunctionValue(cref(fSimu));
    ArrayXd stateInit = Eigen::ArrayXd::Constant(1, p_proba);
    int initialRegime = 0 ; // one regime for this test case
    int nbSimul = 1000; // simulation number
    // following only the probability
    double probaResult = semiLagrangianSimu(p_grid, optimizer, finalFunctionValue, p_ndt, stateInit, initialRegime, nbSimul, fileToDump
#ifdef USE_MPI
                                            , world
#endif
                                           );
    cout <<  " Proba obtained " << probaResult << endl ;
    // now simulate using the optimal control
    double probaResult2 = semiLagrangianSimuControl(p_grid, optimizer, finalFunctionValue, p_ndt, stateInit, initialRegime, nbSimul, fileToDump
#ifdef USE_MPI
                          , world
#endif
                                                   );
    cout <<  " Proba obtained with optimal control" << probaResult2 << endl ;

    return  make_pair(valAndError.second, probaResult) ;
}

BOOST_AUTO_TEST_CASE(TestSemiLagrang3Lin)
{
    // probability target
    double proba = 0.5;
    //  number of time steps
    int ndt = 100 ;
    // mesh number (not converged)
    int nmesh = 100 ;
    // grid
    ArrayXd lowValues = ArrayXd::Constant(1, 0.);
    ArrayXd step  = ArrayXd::Constant(1, 1. / nmesh);
    ArrayXi nstep = ArrayXi::Constant(1, nmesh);
    shared_ptr<SpaceGrid>  grid = make_shared<RegularSpaceGrid>(lowValues, step, nstep);
    pair<double, double>  errorAndProba = testCase3(ndt,  grid, proba);
    BOOST_CHECK(errorAndProba.first < 0.008);
    BOOST_CHECK(fabs(errorAndProba.second - proba) < 0.02);
}

BOOST_AUTO_TEST_CASE(TestSemiLagrang3Quad)
{
    // probability target
    double proba = 0.5;
    //  number of time steps
    int ndt = 100 ;
    // mesh number
    int nmesh = 50 ;
    // grid
    ArrayXd lowValues = ArrayXd::Constant(1, 0.);
    ArrayXd step  = ArrayXd::Constant(1, 1. / nmesh);
    ArrayXi nstep = ArrayXi::Constant(1, nmesh);
    ArrayXi npoly = ArrayXi::Constant(1, 2);
    shared_ptr<SpaceGrid>  grid = make_shared<RegularLegendreGrid>(lowValues, step, nstep, npoly);
    pair<double, double>  errorAndProba = testCase3(ndt,  grid, proba);
    BOOST_CHECK(errorAndProba.first < 0.008);
    BOOST_CHECK(fabs(errorAndProba.second - proba) < 0.02);
}

BOOST_AUTO_TEST_CASE(TestSemiLagrang3Cubic)
{
    // probability target
    double proba = 0.5;
    //  number of time steps
    int ndt = 100 ;
    // mesh number
    int nmesh = 50 ;
    // grid
    ArrayXd lowValues = ArrayXd::Constant(1, 0.);
    ArrayXd step  = ArrayXd::Constant(1, 1. / nmesh);
    ArrayXi nstep = ArrayXi::Constant(1, nmesh);
    ArrayXi npoly = ArrayXi::Constant(1, 3);
    shared_ptr<SpaceGrid>  grid = make_shared<RegularLegendreGrid>(lowValues, step, nstep, npoly);
    pair<double, double>  errorAndProba = testCase3(ndt,  grid, proba);
    BOOST_CHECK(errorAndProba.first < 0.008);
    BOOST_CHECK(fabs(errorAndProba.second - proba) < 0.02);
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
