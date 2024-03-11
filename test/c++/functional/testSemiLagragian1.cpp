

#ifndef USE_MPI
#define BOOST_TEST_MODULE testSemiLagrangCase1
#endif
#define BOOST_TEST_DYN_LINK
#define _USE_MATH_DEFINES
#include <math.h>
#include <functional>
#include <memory>
#include <fstream>
#ifdef USE_MPI
#include "boost/mpi.hpp"
#endif
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "libflow/core/grids/RegularSpaceGrid.h"
#include "libflow/core/grids/RegularLegendreGrid.h"
#include "libflow/core/grids/SparseSpaceGridBound.h"
#include "test/c++/tools/semilagrangien/OptimizeSLCase1.h"
#include "test/c++/tools/semilagrangien/semiLagrangianTime.h"

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
        if (x(0) < 0)
            return sin(x(1) / 2.) * sin(x(0) / 2.);
        else
            return sin(x(1) / 2.) * sin(x(0) / 4.);
    }
};

// problem solution
class Solution
{

public :

    Solution() {};

    double operator()(const double &t, const ArrayXd &x) const
    {
        if (x(0) < 0)
            return (1. + t) * sin(x(1) / 2.) * sin(x(0) / 2.);
        else
            return (1 + t) * sin(x(1) / 2.) * sin(x(0) / 4.);
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
double  testCase1LinearInterpolation(const int &p_ndt,  const  shared_ptr<libflow::SpaceGrid>   &p_grid)
{
#ifdef USE_MPI
    boost::mpi::communicator world;
#endif
    // steps
    double dt = 1. / p_ndt;

    // Create optimizer
    shared_ptr<OptimizerSLBase> optimizer = make_shared<OptimizerSLCase1>(dt);

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
    string fileToDump = "DumpLG1";
    // error at point
    pair<double, double> valAndError = semiLagrangianTime(p_grid, optimizer, initialVal, boundary, dt, p_ndt, point, initRegime, solution, fileToDump
#ifdef USE_MPI
                                       , world
#endif
                                                         );
    cout << " Errmax " <<  valAndError.second << endl ;
    return  valAndError.second ;
}

BOOST_AUTO_TEST_CASE(TestSemiLagrang1Lin)
{
    //  number of time steps
    int ndt = 100 ;
    // mesh number (not converged)
    int nmesh = 80 ;
    // grid
    ArrayXd lowValues = ArrayXd::Constant(2, -2 * M_PI);
    ArrayXd step  = ArrayXd::Constant(2, 4 * M_PI / nmesh);
    ArrayXi nstep = ArrayXi::Constant(2, nmesh);
    shared_ptr<libflow::SpaceGrid>  grid = make_shared<RegularSpaceGrid>(lowValues, step, nstep);
    double error = testCase1LinearInterpolation(ndt,  grid);
    BOOST_CHECK(error < 0.06);
}

BOOST_AUTO_TEST_CASE(TestSemiLagrang1Quad)
{
    //  number of time steps
    int ndt = 100 ;
    // mesh number
    int nmesh = 40 ;
    // grid
    ArrayXd lowValues = ArrayXd::Constant(2, -2 * M_PI);
    ArrayXi npoly = ArrayXi::Constant(2, 2);
    ArrayXd step  = ArrayXd::Constant(2, 4 * M_PI / nmesh);
    ArrayXi nstep = ArrayXi::Constant(2, nmesh);
    shared_ptr<libflow::SpaceGrid>  grid = make_shared<RegularLegendreGrid>(lowValues, step, nstep, npoly);
    double error = testCase1LinearInterpolation(ndt,  grid);
    BOOST_CHECK(error < 0.006);

}

BOOST_AUTO_TEST_CASE(TestSemiLagrang1Cubic)
{
    //  number of time steps
    int ndt = 100 ;
    // mesh number
    int nmesh = 20 ;
    // grid
    ArrayXd lowValues = ArrayXd::Constant(2, -2 * M_PI);
    ArrayXi npoly = ArrayXi::Constant(2, 3);
    ArrayXd step  = ArrayXd::Constant(2, 4 * M_PI / nmesh);
    ArrayXi nstep = ArrayXi::Constant(2, nmesh);
    shared_ptr<libflow::SpaceGrid>  grid = make_shared<RegularLegendreGrid>(lowValues, step, nstep, npoly);
    double error = testCase1LinearInterpolation(ndt,  grid);
    BOOST_CHECK(error < 0.006);
}

BOOST_AUTO_TEST_CASE(TestSemiLagrang1SparseQuad)
{
    //  number of time steps
    int ndt = 100 ;
    // grid
    ArrayXd lowValues = ArrayXd::Constant(2, -2 * M_PI);
    ArrayXd sizeDomain =  ArrayXd::Constant(2, 4 * M_PI);
    ArrayXd weight =  ArrayXd::Constant(2, 1.);
    int levelMax = 8;
    int degree = 2 ; // quadratic
    shared_ptr<libflow::SpaceGrid>  grid = make_shared<SparseSpaceGridBound>(lowValues, sizeDomain, levelMax, weight, degree);
    double error = testCase1LinearInterpolation(ndt,  grid);
    BOOST_CHECK(error < 0.006);
}

// define max function : for interpolation adaptation, the criterium is to take max of interpolated values
class functionMaxLevel
{
public :

    double operator()(const SparseSet::const_iterator &p_iterLevel,
                      const ArrayXd &values) const
    {
        double smax = -infty;
        for (SparseLevel::const_iterator iterIndex = p_iterLevel->second.begin(); iterIndex != p_iterLevel->second.end(); ++iterIndex)
        {
            smax = max(smax, fabs(values(iterIndex->second)));
        }
        return smax;
    }

    double operator()(const vector< double> &p_vec) const
    {
        assert(p_vec.size() > 0);
        double smax = p_vec[0];
        for (size_t i = 1; i < p_vec.size(); ++i)
            smax = max(smax, p_vec[i]);
        return smax;
    }
};

BOOST_AUTO_TEST_CASE(TestSemiLagrang1SparseQuadAdapt)
{
    //  number of time steps
    int ndt = 100 ;
    // grid
    ArrayXd lowValues = ArrayXd::Constant(2, -2 * M_PI);
    ArrayXd sizeDomain =  ArrayXd::Constant(2, 4 * M_PI);
    ArrayXd weight =  ArrayXd::Constant(2, 1.);
    int levelMax = 1;
    int degree = 2 ; // quadratic
    double precision = 1e-3;
    shared_ptr<libflow::SparseSpaceGridBound>  grid = make_shared<SparseSpaceGridBound>(lowValues, sizeDomain, levelMax, weight, degree);

    // adapt the grid
    function< double(const SparseSet::const_iterator &,  const ArrayXd &)> fErrorOneLevel = functionMaxLevel();
    function< double(const vector< double> &)> fErrorLevels = functionMaxLevel();

    // initial function value
    function<double(const int &, const ArrayXd &)>  initialVal = Initial();
    function<double(const ArrayXd &)> funcForRaf = bind(initialVal, 0, std::placeholders::_1);

    // first hierarchize initial level
    shared_ptr<GridIterator > iterGrid = grid->getGridIterator();

    // first hierachization at low level
    ArrayXd valuesFunction(grid->getNbPoints());
    while (iterGrid->isValid())
    {
        ArrayXd pointCoord = iterGrid->getCoordinate();
        valuesFunction(iterGrid->getCount()) = funcForRaf(pointCoord);
        iterGrid->next();
    }
    // Hieriarchize
    ArrayXd hierarValues = valuesFunction;
    grid->toHierarchize(hierarValues);

    // now refine the grid
    grid->refine(precision, funcForRaf, fErrorOneLevel, fErrorLevels, valuesFunction, hierarValues);

    double error = testCase1LinearInterpolation(ndt,  grid);
    BOOST_CHECK(error < 0.006);
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
