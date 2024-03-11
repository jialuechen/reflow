// Copyright (C) 2016 Fime

#define BOOST_TEST_DYN_LINK
#define _USE_MATH_DEFINES
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
#include "test/c++/tools/semilagrangien/OptimizeSLCase2.h"
#include "test/c++/tools/semilagrangien/semiLagrangianTimeDist.h"

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

// Initial function value
class Initial
{
public :

    Initial() {};

    double operator()(const int &, const ArrayXd &x) const
    {
        return 2 * sin(x(0)) * sin(x(1));
    }
};

// problem solution
class Solution
{

public :

    Solution() {};

    double operator()(const double &t, const ArrayXd &x) const
    {
        return (2 - t) * sin(x(0)) * sin(x(1));
    }
    double operator()(const double &t, const  int &, const ArrayXd &x) const
    {
        return  operator()(t, x);
    }
};


/// Optimize semi Lagrangian with linear interpolation
/// \param p_ndt number of time steps
/// \param p_grid  interpolation grid
/// \return max error on the mesh
double  testCase2LinearInterpolation(const int &p_ndt,  const  shared_ptr<libflow::FullGrid>   &p_grid)
{

    boost::mpi::communicator world;
    // steps
    double dt = 1. / p_ndt;

    // Create optimizer
    double beta = 0.1;
    shared_ptr<OptimizerSLBase> optimizer = make_shared<OptimizerSLCase2>(beta, dt);

    // initial function value
    function<double(const int &, const ArrayXd &)>  initialVal = Initial();

    // solution
    function<double(const double &, const ArrayXd &)> solution = Solution();

    // boundary
    function<double(const double &,  const int &, const ArrayXd &)> boundary = Solution();

    // point for interpolation
    ArrayXd point = ArrayXd::Constant(2, 0.);
    // only one regime
    int initRegime = 0 ;
    // file to dump
    string fileToDump = "DumpLG2Mpi" + to_string(world.size());
    // in one file
    bool bOneFile = true;
    // error at point
    pair<double, double> valAndError = semiLagrangianTimeDist(p_grid, optimizer, initialVal, boundary, dt, p_ndt, point, initRegime, solution, fileToDump, bOneFile, world);
    return  valAndError.second ;
}

BOOST_AUTO_TEST_CASE(TestSemiLagrang2Lin)
{
    boost::mpi::communicator world;
    //  number of time steps
    int ndt = 100 ;
    // mesh number (not converged)
    int nmesh = 80 ;
    // grid
    ArrayXd lowValues = ArrayXd::Constant(2, -2 * M_PI);
    ArrayXd step  = ArrayXd::Constant(2, 4 * M_PI / nmesh);
    ArrayXi nstep = ArrayXi::Constant(2, nmesh);
    shared_ptr<libflow::FullGrid>  grid = make_shared<RegularSpaceGrid>(lowValues, step, nstep);
    double error = testCase2LinearInterpolation(ndt,  grid);
    if (world.rank() == 0)
        BOOST_CHECK(error < 0.25);
    world.barrier();
}

BOOST_AUTO_TEST_CASE(TestSemiLagrang2Quad)
{
    boost::mpi::communicator world;
    //  number of time steps
    int ndt = 100 ;
    // mesh number
    int nmesh = 20 ;
    // grid
    ArrayXd lowValues = ArrayXd::Constant(2, -2 * M_PI);
    ArrayXi npoly = ArrayXi::Constant(2, 2);
    ArrayXd step  = ArrayXd::Constant(2, 4 * M_PI / nmesh);
    ArrayXi nstep = ArrayXi::Constant(2, nmesh);
    shared_ptr<libflow::FullGrid>  grid = make_shared<RegularLegendreGrid>(lowValues, step, nstep, npoly);
    double error = testCase2LinearInterpolation(ndt,  grid);
    if (world.rank() == 0)
        BOOST_CHECK(error < 0.01);
    world.barrier();
}

BOOST_AUTO_TEST_CASE(TestSemiLagrang2Cubic)
{
    boost::mpi::communicator world;
    //  number of time steps
    int ndt = 100 ;
    // mesh number
    int nmesh = 20 ;
    // grid
    ArrayXd lowValues = ArrayXd::Constant(2, -2 * M_PI);
    ArrayXi npoly = ArrayXi::Constant(2, 3);
    ArrayXd step  = ArrayXd::Constant(2, 4 * M_PI / nmesh);
    ArrayXi nstep = ArrayXi::Constant(2, nmesh);
    shared_ptr<libflow::FullGrid>  grid = make_shared<RegularLegendreGrid>(lowValues, step, nstep, npoly);
    double error = testCase2LinearInterpolation(ndt,  grid);
    if (world.rank() == 0)
        BOOST_CHECK(error < 0.02);
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
