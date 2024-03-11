

#define BOOST_TEST_DYN_LINK
#include <math.h>
#include <functional>
#include <memory>
#include <fstream>
#include <boost/test/unit_test.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/RegularSpaceGrid.h"
#include "libflow/core/grids/RegularLegendreGrid.h"
#include "libflow/core/grids/SparseSpaceGridBound.h"
#include "test/c++/tools/semilagrangien/OptimizeSLCase3.h"
#include "test/c++/tools/semilagrangien/semiLagrangianTimeDist.h"
#include "test/c++/tools/NormalCumulativeDistribution.h"
#include "test/c++/tools/InvNormalCumulativeDistribution.h"
#include "test/c++/tools/semilagrangien/semiLagrangianSimuDist.h"
#include "test/c++/tools/semilagrangien/semiLagrangianSimuControlDist.h"

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


/// Optimize semi Lagrangian with linear interpolation
/// Stochastic target case: see Choukroun, Elie, Warin
/// \param p_ndt number of time steps
/// \param p_grid  interpolation grid
/// \param p_proba  target probability
/// \param p_bOneFile one file to dump ?
void  testCase3LinearInterpolation(const int &p_ndt,  const  shared_ptr<libflow::FullGrid>   &p_grid, const double &p_proba, const bool &p_bOneFile)
{

    boost::mpi::communicator world;
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

    // initial function value
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
    string fileToDump = "DumpLG3MPI" + to_string(world.size());
    // error at point
    pair<double, double> valAndError = semiLagrangianTimeDist(p_grid, optimizer, initialVal, boundary, dt, p_ndt, point, initRegime, solution, fileToDump, p_bOneFile, world);
    if (world.rank() == 0)
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
    double probaResult = semiLagrangianSimuDist(p_grid, optimizer, finalFunctionValue, p_ndt, stateInit, initialRegime, nbSimul, fileToDump, p_bOneFile, world);
    double probaResult2 = semiLagrangianSimuControlDist(p_grid, optimizer, finalFunctionValue, p_ndt, stateInit, initialRegime, nbSimul, fileToDump, p_bOneFile, world);
    if (world.rank() == 0)
    {
        cout <<  " Proba obtained " << probaResult << " and " <<  probaResult2 << endl ;
        BOOST_CHECK(valAndError.second < 0.008);
        BOOST_CHECK(fabs(probaResult - p_proba) < 0.02);
        BOOST_CHECK(fabs(probaResult2 - p_proba) < 0.03);
    }
}

BOOST_AUTO_TEST_CASE(TestSemiLagrang3LinOneFile)
{
    boost::mpi::communicator world;
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
    shared_ptr<libflow::FullGrid>  grid = make_shared<RegularSpaceGrid>(lowValues, step, nstep);
    // one file tu dump
    bool bOneFile = true;
    testCase3LinearInterpolation(ndt,  grid, proba, bOneFile);

}

BOOST_AUTO_TEST_CASE(TestSemiLagrang3LinMultipleFiles)
{
    boost::mpi::communicator world;
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
    shared_ptr<libflow::FullGrid>  grid = make_shared<RegularSpaceGrid>(lowValues, step, nstep);
    // multiple files to dump
    bool bOneFile = false;
    testCase3LinearInterpolation(ndt,  grid, proba, bOneFile);
}

BOOST_AUTO_TEST_CASE(TestSemiLagrang3Quad)
{
    boost::mpi::communicator world;
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
    shared_ptr<libflow::FullGrid>  grid = make_shared<RegularLegendreGrid>(lowValues, step, nstep, npoly);
    // one file tu dump
    bool bOneFile = true;
    testCase3LinearInterpolation(ndt,  grid, proba, bOneFile);
}

BOOST_AUTO_TEST_CASE(TestSemiLagrang3Cubic)
{
    boost::mpi::communicator world;
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
    shared_ptr<libflow::FullGrid>  grid = make_shared<RegularLegendreGrid>(lowValues, step, nstep, npoly);
    // one file tu dump
    bool bOneFile = true;
    testCase3LinearInterpolation(ndt,  grid, proba, bOneFile);
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
