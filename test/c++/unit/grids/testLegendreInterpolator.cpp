
#define BOOST_TEST_MODULE testLegendreInterpolator
#define BOOST_TEST_DYN_LINK
#include <fstream>
#include <memory>
#define _USE_MATH_DEFINES
#include <cmath>
#include <boost/lexical_cast.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/random.hpp>
#include <boost/random/uniform_01.hpp>
#include <Eigen/Dense>
#include "libflow/core/utils/constant.h"
#include "libflow/core/grids/RegularLegendreGrid.h"
#include "libflow/core/grids/LegendreInterpolator.h"
#include "libflow/core/grids/LegendreInterpolatorSpectral.h"

using namespace std;
using namespace Eigen;
using namespace libflow;

double accuracyEqual = 1e-10;
double accuracyNearEqual = 1e-5;


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

void testLegendreInterpolatorND(ArrayXi &nPol)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    int nDim = nPol.size();
    ArrayXd lowValues(nDim);
    for (int i = 0; i < nDim; ++i)
        lowValues(i) = static_cast<double>(i) + 1.;
    ArrayXd step(nDim);
    for (int i = 0; i < nDim; ++i)
        step(i) = 0.6 * (i + 1);
    ArrayXi  nbStep(nDim);
    for (int i = 0; i < nDim; ++i)
        nbStep(i) = 3 * (i + 1);

    // regular
    RegularLegendreGrid regGrid(lowValues, step, nbStep, nPol);


    // Data
    ArrayXXd data(2, regGrid.getNbPoints());
    shared_ptr<GridIterator> iterRegGrid  =  regGrid.getGridIterator();
    while (iterRegGrid->isValid())
    {
        ArrayXd pointCoord = iterRegGrid->getCoordinate();
        data(0, iterRegGrid->getCount()) = exp(pointCoord.sum() / nDim);
        data(1, iterRegGrid->getCount()) = 1. + log1p(pointCoord.sum());
        iterRegGrid->next();
    }

    shared_ptr<GridIterator> iterRegGrid2  =  regGrid.getGridIterator();
    while (iterRegGrid2->isValid())
    {
        ArrayXd point = iterRegGrid2->getCoordinate();
        LegendreInterpolator  regLin(&regGrid, point);
        ArrayXd interpReg = regLin.applyVec(data);
        BOOST_CHECK_CLOSE(data(0, iterRegGrid2->getCount()), interpReg(0), accuracyEqual);
        BOOST_CHECK_CLOSE(data(1, iterRegGrid2->getCount()), interpReg(1), accuracyEqual);
        iterRegGrid2->next();
    }
}

void testLegendreInterpolatorNDSecond(int nDim, int nPol)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXd lowValues = ArrayXd::Zero(nDim);
    ArrayXd step = ArrayXd::Constant(nDim, 1.);
    ArrayXi  nbStep = ArrayXi::Constant(nDim, 1);
    ArrayXi npoly = ArrayXi::Constant(nDim, nPol);

    // regular
    RegularLegendreGrid regGrid(lowValues, step, nbStep, npoly);


    ArrayXd data(regGrid.getNbPoints());

    shared_ptr<GridIterator> iterRegGrid  =  regGrid.getGridIterator();
    while (iterRegGrid->isValid())
    {
        ArrayXd pointCoord = iterRegGrid->getCoordinate();
        data(iterRegGrid->getCount()) = 1;
        for (int id = 0; id < nDim; ++id)
            data(iterRegGrid->getCount()) *= pointCoord(id);
        iterRegGrid->next();
    }

    ArrayXd point = ArrayXd::Constant(nDim, 1. / 3.);
    LegendreInterpolator regLin(&regGrid, point);

    double interpReg = regLin.apply(data);

    BOOST_CHECK_CLOSE(interpReg, std::pow(1. / 3, static_cast<double>(nDim)), accuracyEqual);

}

/// check exact interpolation of polynomials
void testLegendreInterpolatorExact(ArrayXi &nPol)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    int nDim = nPol.size();
    ArrayXd lowValues = ArrayXd::Zero(nDim);
    ArrayXd step  = ArrayXd::Constant(nDim, 1.);
    ArrayXi  nbStep = ArrayXi::Constant(nDim, 1);

    // regular
    RegularLegendreGrid regGrid(lowValues, step, nbStep, nPol);


    // Data
    ArrayXd data(regGrid.getNbPoints());
    shared_ptr<GridIterator> iterRegGrid  =  regGrid.getGridIterator();
    while (iterRegGrid->isValid())
    {
        ArrayXd pointCoord = iterRegGrid->getCoordinate();
        data(iterRegGrid->getCount()) = 1. + pointCoord(0);
        for (int id = 1; id < nDim; ++id)
            data(iterRegGrid->getCount()) *= pow(1. + pointCoord(id), static_cast<double>(nPol(id)));
        iterRegGrid->next();
    }

    boost::mt19937 generator;
    boost::uniform_01<double> alea;
    boost::variate_generator<boost::mt19937 &, boost::uniform_01<double> > uniform(generator, alea);

    int nsimul = 1000;
    for (int is = 0; is < nsimul; ++is)
    {
        ArrayXd pointCoord(nDim);
        for (int id = 0; id < nDim; ++id)
            pointCoord(id) = step(id) * uniform();
        double val = 1. + pointCoord(0);
        for (int id = 1; id < nDim; ++id)
            val *= pow(1. + pointCoord(id), static_cast<double>(nPol(id)));

        LegendreInterpolator regLin(&regGrid, pointCoord);
        double vInterp = regLin.apply(data);
        // check exact  interpolation for polynoms
        BOOST_CHECK_CLOSE(vInterp, val,  accuracyNearEqual);

    }
}

/// check exact interpolation of polynomials
void testLegendreInterpolatorBisExact(ArrayXi &nPol)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    int nDim = nPol.size();
    ArrayXd lowValues = ArrayXd::Constant(nDim, 1.); // bottom of the domain
    ArrayXd step  = ArrayXd::Constant(nDim, 1.); // size of the mesh
    ArrayXi  nbStep = ArrayXi::Constant(nDim, 5); // number of mesh in each direction

    // regular
    RegularLegendreGrid regGrid(lowValues, step, nbStep, nPol);


    // Data
    ArrayXd data(regGrid.getNbPoints());
    shared_ptr<GridIterator> iterRegGrid  =  regGrid.getGridIterator();
    while (iterRegGrid->isValid())
    {
        ArrayXd pointCoord = iterRegGrid->getCoordinate();
        data(iterRegGrid->getCount()) = 1. + pointCoord(0);
        for (int id = 1; id < nDim; ++id)
            data(iterRegGrid->getCount()) *= pow(1. + pointCoord(id), static_cast<double>(nPol(id)));
        iterRegGrid->next();
    }

    // spectral interpolator
    LegendreInterpolatorSpectral interpolator(&regGrid, data);

    boost::mt19937 generator;
    boost::uniform_01<double> alea;
    boost::variate_generator<boost::mt19937 &, boost::uniform_01<double> > uniform(generator, alea);

    int nsimul = 1000;
    for (int is = 0; is < nsimul; ++is)
    {
        ArrayXd pointCoord(nDim);
        for (int id = 0; id < nDim; ++id)
            pointCoord(id) = lowValues(id) + nbStep(id) * step(id) * uniform();
        double val = 1. + pointCoord(0);
        for (int id = 1; id < nDim; ++id)
            val *= pow(1. + pointCoord(id), static_cast<double>(nPol(id)));

        double vInterp = interpolator.apply(pointCoord);
        // check exact  interpolation for polynoms
        BOOST_CHECK_CLOSE(vInterp, val,  accuracyNearEqual);

    }
}

// Runge  function in 1D
void testLegendreInterpolator1DAndPlot(int p_nPol)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXd lowValues = ArrayXd::Constant(1, -1.); // corner  point
    ArrayXd step =  ArrayXd::Constant(1, 2.); // size of the meshes
    ArrayXi  nbStep = ArrayXi::Constant(1, 1); // number of mesh in each direction
    ArrayXi nPol = ArrayXi::Constant(1, p_nPol); // polynomial approximation
    // regular Legrendre
    RegularLegendreGrid regGrid(lowValues, step, nbStep, nPol);

    // Data array to store values on the grid points
    ArrayXd data(regGrid.getNbPoints());
    shared_ptr<GridIterator> iterRegGrid  =  regGrid.getGridIterator();
    while (iterRegGrid->isValid())
    {
        ArrayXd pointCoord = iterRegGrid->getCoordinate();
        data(iterRegGrid->getCount()) = 1. / (1. + 25 * pointCoord(0) * pointCoord(0)); // store runge function
        iterRegGrid->next();
    }
    // point
    ArrayXd point(1);
    int nbp =  1000;
    double dx = 2. / nbp;
    // file creation
    std::string fileP = "RungeInterpolation" + boost::lexical_cast<std::string>(p_nPol);
    std::fstream fileInterp(fileP.c_str(), std::fstream::out);
    for (int ip = 0; ip <= nbp; ++ip)
    {
        point(0) = -1 + ip * dx;
        // create interpolator
        shared_ptr<Interpolator> interp = regGrid.createInterpolator(point);
        double  interpReg = interp->apply(data); // interpolated value
        fileInterp << point(0) << " " << interpReg << std::endl ;
    }
    fileInterp.close();
}


BOOST_AUTO_TEST_CASE(testLegendreInterpolator1DPol1)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(1, 1);
    testLegendreInterpolatorND(npoly);
}
BOOST_AUTO_TEST_CASE(testLegendreInterpolator1DPol2)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(1, 2);
    testLegendreInterpolatorND(npoly);
}
BOOST_AUTO_TEST_CASE(testLegendreInterpolator1DPol3)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(1, 3);
    testLegendreInterpolatorND(npoly);
}
BOOST_AUTO_TEST_CASE(testLegendreInterpolator1DPol5)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(1, 5);
    testLegendreInterpolatorND(npoly);
}
BOOST_AUTO_TEST_CASE(testLegendreInterpolator2DPol1)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(2, 1);
    testLegendreInterpolatorND(npoly);
}
BOOST_AUTO_TEST_CASE(testLegendreInterpolator2DPol4)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(2, 4);
    testLegendreInterpolatorND(npoly);
}

BOOST_AUTO_TEST_CASE(testLegendreInterpolator2DPol7)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(2, 7);
    testLegendreInterpolatorND(npoly);
}

BOOST_AUTO_TEST_CASE(testLegendreInterpolator3DPol1)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(3, 1);
    testLegendreInterpolatorND(npoly);
}

BOOST_AUTO_TEST_CASE(testLegendreInterpolator3DPol2)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    ArrayXi npoly = ArrayXi::Constant(3, 2);
    testLegendreInterpolatorND(npoly);
}

BOOST_AUTO_TEST_CASE(testLegendreInterpolator3DPol1and2)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    ArrayXi npoly(3) ;
    npoly(0) = 1;
    npoly(1) = 2;
    npoly(2) = 1;
    testLegendreInterpolatorND(npoly);
}

BOOST_AUTO_TEST_CASE(testLegendreInterpolator1DSecondPol1)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    testLegendreInterpolatorNDSecond(1, 1);
}
BOOST_AUTO_TEST_CASE(testLegendreInterpolator2DSecondPol5)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    testLegendreInterpolatorNDSecond(2, 5);
}
BOOST_AUTO_TEST_CASE(testLegendreInterpolator4DSecondPol2)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    testLegendreInterpolatorNDSecond(4, 2);
}


BOOST_AUTO_TEST_CASE(testLegendreExactInterpolator1DPol1)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(1, 1);
    testLegendreInterpolatorExact(npoly);
}
BOOST_AUTO_TEST_CASE(testLegendreExactInterpolator1DPol7)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(1, 7);
    testLegendreInterpolatorExact(npoly);
}

BOOST_AUTO_TEST_CASE(testLegendreExactInterpolator2DPol2)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(2, 2);
    testLegendreInterpolatorExact(npoly);
}
BOOST_AUTO_TEST_CASE(testLegendreExactInterpolator2DPol3)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(2, 3);
    testLegendreInterpolatorExact(npoly);
}
BOOST_AUTO_TEST_CASE(testLegendreExactInterpolator2DPol4)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(2, 4);
    testLegendreInterpolatorExact(npoly);
}


BOOST_AUTO_TEST_CASE(testLegendreExactInterpolator3DPol3)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(3, 3);
    testLegendreInterpolatorExact(npoly);
}


BOOST_AUTO_TEST_CASE(testLegendreExactInterpolatorBis1DPol1)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(1, 1);
    testLegendreInterpolatorBisExact(npoly);
}

BOOST_AUTO_TEST_CASE(testLegendreExactInterpolatorBis1DPol7)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(1, 7);
    testLegendreInterpolatorBisExact(npoly);
}

BOOST_AUTO_TEST_CASE(testLegendreExactInterpolatorBis2DPol2)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(2, 2);
    testLegendreInterpolatorBisExact(npoly);
}
BOOST_AUTO_TEST_CASE(testLegendreExactInterpolatorBis2DPol3)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(2, 3);
    testLegendreInterpolatorBisExact(npoly);
}
BOOST_AUTO_TEST_CASE(testLegendreExactInterpolatorBis2DPol4)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(2, 4);
    testLegendreInterpolatorBisExact(npoly);
}


BOOST_AUTO_TEST_CASE(testLegendreExactInterpolatorBis3DPol3)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi npoly = ArrayXi::Constant(3, 3);
    testLegendreInterpolatorBisExact(npoly);
}

BOOST_AUTO_TEST_CASE(testRungeFunction)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    for (int npol = 1; npol <= 10 ; ++npol)
        testLegendreInterpolator1DAndPlot(npol);
}



void testLegendreInterpolatorPlot(int p_nDim, int p_nPol, int p_nStep, std::function< double(const ArrayXd &)> &p_func, const std::string   &p_file, int p_nStepLoc)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXd lowValues = ArrayXd::Zero(p_nDim);
    ArrayXd step = ArrayXd::Constant(p_nDim, 1. / p_nStep);
    ArrayXi nbStep = ArrayXi::Constant(p_nDim, p_nStep);
    ArrayXi npoly = ArrayXi::Constant(p_nDim, p_nPol);

    // regular
    RegularLegendreGrid regGrid(lowValues, step, nbStep, npoly);


    ArrayXd data(regGrid.getNbPoints());

    shared_ptr<GridIterator> iterRegGrid  =  regGrid.getGridIterator();
    while (iterRegGrid->isValid())
    {
        ArrayXd pointCoord = iterRegGrid->getCoordinate();
        data(iterRegGrid->getCount()) = p_func(pointCoord);
        iterRegGrid->next();
    }

    // creation a thin grid
    ArrayXd lowValuesLoc = ArrayXd::Zero(p_nDim);
    ArrayXd stepLoc = ArrayXd::Constant(p_nDim, 1. / p_nStepLoc);
    ArrayXi nbStepLoc = ArrayXi::Constant(p_nDim, p_nStepLoc);
    ArrayXi npolyLoc = ArrayXi::Constant(p_nDim, 1);

    std::fstream File(p_file.c_str(), std::fstream::out);
    RegularLegendreGrid regGridLoc(lowValuesLoc, stepLoc, nbStepLoc, npolyLoc);
    shared_ptr<GridIterator> iterRegGridLoc  =  regGridLoc.getGridIterator();
    while (iterRegGridLoc->isValid())
    {
        ArrayXd pointCoord = iterRegGridLoc->getCoordinate();
        LegendreInterpolator regLin(&regGrid, pointCoord);
        for (int i = 0; i < pointCoord.size(); ++i)
            File   << pointCoord(i) << " " ;
        File <<  regLin.apply(data) - p_func(pointCoord) << std::endl ;
        iterRegGridLoc->next();
    }

    File.close();

}

BOOST_AUTO_TEST_CASE(testLinear1DSin)
{
    class fSin
    {
    public :
        fSin() {}

        double operator()(const  ArrayXd &x) const
        {
            double ret = 1 ;
            for (int i = 0; i < x.size(); ++i)
                ret *= sin(M_PI * 2 * x(i));
            return ret;
        }
    };

    std::function< double(const ArrayXd &)> func = fSin();

    int nbStep = 4 ;
    int nDim = 1 ;
    int nStepLoc = 10000;
    for (int iPol = 1; iPol < 7; ++iPol)
    {
        int iStepDeb = 4;
        for (int istep = 1; istep < nbStep; ++istep)
        {

            std::string filePrint = "testInterpol1DSin" + boost::lexical_cast<std::string>(iStepDeb) + "Pol" + boost::lexical_cast<std::string>(iPol) ;
            testLegendreInterpolatorPlot(nDim, iPol, iStepDeb,  func, filePrint, nStepLoc);
            iStepDeb *= 2;
        }
    }
}

BOOST_AUTO_TEST_CASE(testLinear2DSin)
{
    class fSin
    {
    public :
        fSin() {}

        double operator()(const  ArrayXd &x) const
        {
            double ret = 1 ;
            for (int i = 0; i < x.size(); ++i)
                ret *= sin(M_PI * 2 * x(i));
            return ret;
        }
    };

    std::function< double(const ArrayXd &)> func = fSin();

    int nbStep = 4 ;
    int nDim = 2 ;
    int nStepLoc = 200;
    for (int iPol = 1; iPol < 7; ++iPol)
    {
        int iStepDeb = 4;
        for (int istep = 1; istep < nbStep; ++istep)
        {

            std::string filePrint = "testInterpol2DSin" + boost::lexical_cast<std::string>(iStepDeb) + "Pol" + boost::lexical_cast<std::string>(iPol) ;
            testLegendreInterpolatorPlot(nDim, iPol, iStepDeb,  func, filePrint, nStepLoc);
            iStepDeb *= 2;
        }
    }
}
BOOST_AUTO_TEST_CASE(testLegendreInterpolatorArnaudBug)
{
    ArrayXd lowValues(1);
    lowValues(0) = 0.0;
    ArrayXd step(1);
    step(0) = 9.9999999999999662;
    ArrayXi nstep(1);
    nstep(0) = 5;
    ArrayXi npolyLoc = ArrayXi::Constant(1, 1);
    RegularLegendreGrid grid(lowValues, step, nstep, npolyLoc);
    ArrayXd point(1);
    point(0) = 29.999999999999719;
    LegendreInterpolator interp(&grid, point);
    ArrayXd data(grid.getNbPoints());
    data(0) = 100.0;
    data(1) = 100.0;
    data(2) = 100.0;
    data(3) = 100.0;
    data(4) = -infty;
    double result = interp.apply(data);
    BOOST_CHECK_CLOSE(result, 100.0, accuracyEqual);
}
