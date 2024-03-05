
#define BOOST_TEST_MODULE testLinearInterpolator
#define BOOST_TEST_DYN_LINK
#include <array>
#include <memory>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "libflow/core/utils/constant.h"
#include "libflow/core/grids/GeneralSpaceGrid.h"
#include "libflow/core/grids/RegularSpaceGrid.h"
#include "libflow/core/grids/GridIterator.h"
#include "libflow/core/grids/LinearInterpolator.h"
#include "libflow/core/grids/LinearInterpolatorSpectral.h"

using namespace std;
using namespace Eigen;
using namespace libflow;

double accuracyEqual = 1e-10;

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
std::string normalize_test_case_name(const_string name)
{
    return (name[0] == '&' ? std::string(name.begin() + 1, name.size() - 1) : std::string(name.begin(), name.size()));
}
}
}
}

void testLinearInterpolatorND(int nDim)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXd lowValues(nDim);
    for (int i = 0; i < nDim; ++i)
        lowValues(i) = static_cast<double>(i);
    ArrayXd step(nDim);
    for (int i = 0; i < nDim; ++i)
        step(i) = 0.6 * (i + 1);
    ArrayXi  nbStep(nDim);
    for (int i = 0; i < nDim; ++i)
        nbStep(i) = 3 * nDim * (i + 1);

    // regular
    RegularSpaceGrid regGrid(lowValues, step, nbStep);

    vector<shared_ptr<ArrayXd> > meshPerDimension(nDim);
    for (int i = 0; i < nDim; ++i)
    {
        meshPerDimension[i] = make_shared< ArrayXd >(nbStep(i) + 1);
        (*meshPerDimension[i]) = ArrayXd::LinSpaced(nbStep(i) + 1, lowValues(i), lowValues(i) + nbStep(i) * step(i));
    }

    // general Grid
    GeneralSpaceGrid genGrid(meshPerDimension);

    // Data
    ArrayXXd data(2, regGrid.getNbPoints());
    shared_ptr<GridIterator> iterRegGrid  =  regGrid.getGridIterator();
    while (iterRegGrid->isValid())
    {
        ArrayXd pointCoord = iterRegGrid->getCoordinate();
        data(0, iterRegGrid->getCount()) = exp(pointCoord.sum() / nDim);
        data(1, iterRegGrid->getCount()) = log1p(pointCoord.sum());
        iterRegGrid->next();
    }

    // create iterator on general grid
    shared_ptr<GridIterator> iterGenGrid = genGrid.getGridIterator();
    while (iterGenGrid->isValid())
    {
        ArrayXd point = iterGenGrid->getCoordinate();
        LinearInterpolator  regLin(&regGrid, point);
        LinearInterpolator  genLin(&genGrid, point);
        ArrayXd interpReg = regLin.applyVec(data);
        ArrayXd interpGen = genLin.applyVec(data);
        BOOST_CHECK_CLOSE(data(0, iterGenGrid->getCount()), interpReg(0), accuracyEqual);
        BOOST_CHECK_CLOSE(data(1, iterGenGrid->getCount()), interpReg(1), accuracyEqual);
        BOOST_CHECK_CLOSE(data(0, iterGenGrid->getCount()), interpGen(0), accuracyEqual);
        BOOST_CHECK_CLOSE(data(1, iterGenGrid->getCount()), interpGen(1), accuracyEqual);
        iterGenGrid->next();
    }
}

void testLinearInterpolatorNDSecond(int nDim)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXd lowValues = ArrayXd::Zero(nDim);
    ArrayXd step = ArrayXd::Constant(nDim, 1.);
    ArrayXi  nbStep = ArrayXi::Constant(nDim, 1);

    // regular
    RegularSpaceGrid regGrid(lowValues, step, nbStep);

    // create an array to store the values a function
    ArrayXd data(regGrid.getNbPoints());

    shared_ptr<GridIterator> iterRegGrid  =  regGrid.getGridIterator();
    while (iterRegGrid->isValid())
    {
        ArrayXd pointCoord = iterRegGrid->getCoordinate();
        data(iterRegGrid->getCount()) = 1; // the value is stored in data at place iterRegGrid->getCount()
        for (int id = 0; id < nDim; ++id)
            data(iterRegGrid->getCount()) *= pointCoord(id);
        iterRegGrid->next();
    }

    // point where to interpolate
    ArrayXd point = ArrayXd::Constant(nDim, 1. / 3.);
    // create the interpolator
    LinearInterpolator  regLin(&regGrid, point);
    // get back the interpolated value
    double interpReg = regLin.apply(data);

    BOOST_CHECK_CLOSE(interpReg, std::pow(1. / 3, static_cast<double>(nDim)), accuracyEqual);

    // test spectral interpolator
    LinearInterpolatorSpectral regSpectral(&regGrid, data);
    BOOST_CHECK_CLOSE(interpReg, regSpectral.apply(point), accuracyEqual);
}




BOOST_AUTO_TEST_CASE(testLinearInterpolator1D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    testLinearInterpolatorND(1);
}
BOOST_AUTO_TEST_CASE(testLinearInterpolator2D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    testLinearInterpolatorND(2);
}
BOOST_AUTO_TEST_CASE(testLinearInterpolator3D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    testLinearInterpolatorND(3);
}
BOOST_AUTO_TEST_CASE(testLinearInterpolator1DSecond)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    testLinearInterpolatorNDSecond(1);
}
BOOST_AUTO_TEST_CASE(testLinearInterpolator2DSecond)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    testLinearInterpolatorNDSecond(2);
}
BOOST_AUTO_TEST_CASE(testLinearInterpolator4DSecond)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    testLinearInterpolatorNDSecond(4);
}
BOOST_AUTO_TEST_CASE(testLinearInterpolatorArnaudBug)
{
    ArrayXd lowValues(1);
    lowValues(0) = 0.0;
    ArrayXd step(1);
    step(0) = 9.9999999999999662;
    ArrayXi nstep(1);
    nstep(0) = 5;
    RegularSpaceGrid grid(lowValues, step, nstep);
    ArrayXd point(1);
    point(0) = 29.999999999999719;
    LinearInterpolator interp(&grid, point);
    ArrayXd data(grid.getNbPoints());
    data(0) = 100.0;
    data(1) = 100.0;
    data(2) = 100.0;
    data(3) = 100.0;
    data(4) = -infty;
    double result = interp.apply(data);
    BOOST_CHECK_CLOSE(result, 100.0, accuracyEqual);
}
