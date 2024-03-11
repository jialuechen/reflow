// Copyright (C) 2017 EDF

#define BOOST_TEST_MODULE testGridKernelRegression
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/timer/timer.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "libflow/regression/gridKernelIndexHelper.h"
#include "libflow/regression/gridKernelRegression.h"
#include "libflow/regression/LocalGridKernelRegression.h"
#include "libflow/regression/LocalGridKernelRegressionGeners.h"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/regression/ContinuationValueGeners.h"
#include "libflow/core/utils/constant.h"

using namespace std;
using namespace Eigen;
using namespace libflow;
using namespace gs;

// utilities developed because VS2010 doesn't support auto


double accuracyEqual = 1e-6;

#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif


// functions
auto m1([](const double &x)
{
    return x + exp(-16.0 * x * x);
});
auto m2([](const double &x)
{
    return sin(2.0 * x) + 2.*exp(-16.*x * x);
});
auto m3([](const double &x)
{
    return 0.3 * exp(-4.*pow(x - 1., 2.)) + 0.7 * exp(-16.0 * pow(x - 1, 2));
});


// test in dimension 1
void testDimension1D(const bool &p_bLin, const int &p_nbSimul,  double p_prop, const double &p_epsilon)
{

    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    // particles
    ArrayXXd x(1, p_nbSimul);
    // to regress
    ArrayXd y(p_nbSimul);
    for (int is = 0; is < p_nbSimul; ++is)
    {
        x(0, is) = 0.6 * normal_random();
        y(is) = m1(x(0, is)) + 0.5 * normal_random();
    }
    // test local NAIVE
    ArrayXd regressedNaive = locAdapRegNaive(x, y, p_prop, p_bLin);

    // test regression object
    double q = 1; // coeff for the number of grid points used
    LocalGridKernelRegression kernelReg(false, x, p_prop,  q, p_bLin);

    ArrayXd regressed = kernelReg.getAllSimulations(y);

    // difference
    BOOST_CHECK_SMALL((regressed - regressedNaive).abs().maxCoeff(), p_epsilon);
}

// test in dimension 2
void testDimension2D(const bool &p_bLin, const int &p_nbSimul,  double p_prop, const double &p_epsilon)
{

    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    // particles
    ArrayXXd x(2, p_nbSimul);
    // to regress
    ArrayXd y(p_nbSimul);
    for (int is = 0; is < p_nbSimul; ++is)
    {
        for (int id = 0; id < 2; ++id)
            x(id, is) = 0.6 * normal_random();
        y(is) = m2(x(0, is)) + m2(x(1, is)) + 0.5 * normal_random();
    }
    // test local NAIVE
    ArrayXd regressedNaive = locAdapRegNaive(x, y, p_prop, p_bLin);

    // test regression object
    double q = 1; // coeff for the number of grid points used
    LocalGridKernelRegression kernelReg(false, x, p_prop, q, p_bLin);

    ArrayXd regressed = kernelReg.getAllSimulations(y);

    // difference
    BOOST_CHECK_SMALL((regressed - regressedNaive).abs().maxCoeff(), p_epsilon);
}


// test in dimension 3
void testDimension3D(const bool &p_bLin, const int &p_nbSimul,  double p_prop, const double &p_epsilon)
{

    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    // particles
    ArrayXXd x(3, p_nbSimul);

    // to regress
    ArrayXd y(p_nbSimul);
    for (int is = 0; is < p_nbSimul; ++is)
    {
        for (int id = 0; id < 3; ++id)
            x(id, is) = 0.7 * normal_random();
        y(is) = m2(x(0, is)) + m2(x(1, is)) + m3(x(2, is)) + 0.5 * normal_random();
    }
    // test local NAIVE
    ArrayXd regressedNaive = locAdapRegNaive(x, y, p_prop, p_bLin);

    // test regression object
    double q = 1; // coeff for the number of grid points used
    LocalGridKernelRegression kernelReg(false, x, p_prop, q, p_bLin);

    ArrayXd regressed = kernelReg.getAllSimulations(y);

    // difference
    BOOST_CHECK_SMALL((regressed - regressedNaive).abs().maxCoeff(), p_epsilon);
}

// test fonctionality in 2D
void testDimensionFonctionality1D(const bool &p_bLin, const int &p_nbSimul,  double p_prop, const double &p_epsilon)
{

    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    // particles
    ArrayXXd x(1, p_nbSimul);
    // to regress
    ArrayXd y(p_nbSimul);
    for (int is = 0; is < p_nbSimul; ++is)
    {
        x(0, is) = 0.6 * normal_random();
        y(is) = m2(x(0, is)) + 0.5 * normal_random();
    }
    // test regression object
    double q = 1; // coeff for the number of grid points used
    LocalGridKernelRegression kernelReg(false, x, p_prop, q, p_bLin);

    ArrayXd regressed = kernelReg.getAllSimulations(y);


    // multiple regressions
    ArrayXXd yy(1, p_nbSimul);
    for (int is = 0; is < p_nbSimul; ++is)
        yy(0, is) = y(is);
    ArrayXXd regressedM = kernelReg.getAllSimulationsMultiple(yy);

    // difference
    BOOST_CHECK_SMALL((regressed - regressedM.row(0).transpose()).abs().maxCoeff(), p_epsilon);

    // reconstruction by coeff fucntions
    ArrayXd  regressedGrid = kernelReg.getCoordBasisFunction(y);
    ArrayXd regressed2 = kernelReg.reconstruction(regressedGrid);
    BOOST_CHECK_SMALL((regressed - regressed2).maxCoeff(), p_epsilon);

    // reconstruction by coeff fucntions
    ArrayXXd  regressedGrid1 = kernelReg.getCoordBasisFunctionMultiple(yy);
    ArrayXXd regressed3 = kernelReg.reconstructionMultiple(regressedGrid1);
    BOOST_CHECK_SMALL((regressed - regressed3.row(0).transpose()).abs().maxCoeff(), p_epsilon);

    // test reconstruction par simu
    for (int is = 0; is < p_nbSimul / 10; ++is)
    {
        double regressAPoint = kernelReg.reconstructionASim(is, regressedGrid);
        BOOST_CHECK_SMALL(fabs(regressAPoint - regressed(is)), p_epsilon);
    }
    // archive
    {
        BinaryFileArchive ar("archiveGK", "w");
        ar << Record(kernelReg, "Regressor", "Top") ;
        ar.flush();
    }
    {
        BinaryFileArchive ar("archiveGK", "r");
        LocalGridKernelRegression regAr;
        Reference<LocalGridKernelRegression> (ar, "Regressor", "Top").restore(0, &regAr);
        for (int is = 0; is < p_nbSimul / 10; ++is)
        {
            double regressAPoint = regAr.getValue(x.col(is), regressedGrid);
            BOOST_CHECK_SMALL(fabs(regressAPoint - regressed(is)), p_epsilon);
        }
    }
}



// test fonctionality in 2D
void testDimensionFonctionality2D(const bool &p_bLin, const int &p_nbSimul,  double p_prop, const double &p_epsilon)
{

    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    // particles
    ArrayXXd x(2, p_nbSimul);
    // to regress
    ArrayXd y(p_nbSimul);
    for (int is = 0; is < p_nbSimul; ++is)
    {
        for (int id = 0; id < 2; ++id)
            x(id, is) = 0.6 * normal_random();
        y(is) = m2(x(0, is)) + m2(x(1, is)) + 0.5 * normal_random();
    }
    // test regression object
    double q = 1; // coeff for the number of grid points used
    LocalGridKernelRegression kernelReg(false, x, p_prop, q, p_bLin);

    ArrayXd regressed = kernelReg.getAllSimulations(y);


    // multiple regressions
    ArrayXXd yy(1, p_nbSimul);
    for (int is = 0; is < p_nbSimul; ++is)
        yy(0, is) = y(is);
    ArrayXXd regressedM = kernelReg.getAllSimulationsMultiple(yy);

    // difference
    BOOST_CHECK_SMALL((regressed - regressedM.row(0).transpose()).abs().maxCoeff(), p_epsilon);

    // reconstruction by coeff fucntions
    ArrayXd  regressedGrid = kernelReg.getCoordBasisFunction(y);
    ArrayXd regressed2 = kernelReg.reconstruction(regressedGrid);
    BOOST_CHECK_SMALL((regressed - regressed2).maxCoeff(), p_epsilon);

    // reconstruction by coeff fucntions
    ArrayXXd  regressedGrid1 = kernelReg.getCoordBasisFunctionMultiple(yy);
    ArrayXXd regressed3 = kernelReg.reconstructionMultiple(regressedGrid1);
    BOOST_CHECK_SMALL((regressed - regressed3.row(0).transpose()).abs().maxCoeff(), p_epsilon);

    // test reconstruction par simu
    for (int is = 0; is < p_nbSimul / 10; ++is)
    {
        double regressAPoint = kernelReg.reconstructionASim(is, regressedGrid);
        BOOST_CHECK_SMALL(fabs(regressAPoint - regressed(is)), p_epsilon);
    }

}

// test serialization for continuation values
void TestContinuationValue(const bool &p_bLin, const int &p_nDim, const int &p_nbSimul, const double &p_bandWidth, const double &p_accuracyEqual, const double &p_accuracyInterp)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // nb stock points
    int sizeForStock  = 4;
    // generator Mersene Twister
    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    ArrayXXd x = ArrayXXd::Random(p_nDim, p_nbSimul);        // test archive
    ArrayXXd  regressedValues;
    {

        // second member to regress with one stock
        ArrayXXd toRegress(sizeForStock, p_nbSimul);
        for (int is  = 0;  is < p_nbSimul; ++is)
        {
            double prod = m1(x(0, is));
            for (int id = 1 ; id < p_nDim ; ++id)
                prod *=  m1(x(id, is));
            double uncertainty = 4 * normal_random();
            for (int j = 0; j < sizeForStock; ++j)
            {
                toRegress(j, is) = prod * (j + 1) +  uncertainty;
            }
        }
        // grid for stock
        Eigen::ArrayXd lowValues(1), step(1);
        lowValues(0) = 0. ;
        step(0) = 1;
        Eigen::ArrayXi  nbStep(1);
        nbStep(0) = sizeForStock - 1;
        // grid
        shared_ptr< RegularSpaceGrid > regular = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);
        // conditional expectation
        shared_ptr<LocalGridKernelRegression> localRegressor = make_shared<LocalGridKernelRegression>(false, x, p_bandWidth, 1., p_bLin);

        // regress  directly with regressor
        regressedValues = localRegressor->getAllSimulationsMultiple(toRegress);
        // creation continuation value object
        ContinuationValue  continuation(regular, localRegressor,  toRegress.transpose());

        // regress with continuation value object
        ArrayXd ptStock(1) ;
        ptStock(0) = sizeForStock / 2;

        ArrayXd regressedByContinuation(p_nbSimul);
        {
            cout << " Get all simulations" << endl ;
            boost::timer::auto_cpu_timer t;
            regressedByContinuation = continuation.getAllSimulations(ptStock);
        }
        ArrayXd regressedByContinuationSecond(p_nbSimul);
        {
            cout << " Get all simulation one by one " << endl ;
            boost::timer::auto_cpu_timer t;
            for (int is  = 0;  is < p_nbSimul; ++is)
                regressedByContinuationSecond(is) = continuation.getValue(ptStock, x.col(is));
        }
        for (int is  = 0;  is < p_nbSimul; ++is)
        {
            if (fabs(regressedByContinuation(is)) > accuracyEqual)
            {
                if (fabs(regressedByContinuation(is)) > tiny)
                {
                    BOOST_CHECK_CLOSE(regressedByContinuation(is), regressedValues(sizeForStock / 2, is), p_accuracyEqual);
                    BOOST_CHECK_CLOSE(regressedByContinuation(is), regressedByContinuationSecond(is), p_accuracyInterp);
                }
            }
        }
        // default non compression
        BinaryFileArchive ar("archiveGK1", "w");
        ar << Record(continuation, "FirstContinuation", "Top") ;

    }
    {
        // read archive
        BinaryFileArchive ar("archiveGK1", "r");
        ContinuationValue contRead;
        Reference< ContinuationValue >(ar, "FirstContinuation", "Top").restore(0, &contRead);
        ArrayXd ptStock(1) ;
        ptStock(0) = sizeForStock / 2;
        boost::timer::auto_cpu_timer t;
        for (int is  = 0;  is < p_nbSimul; ++is)
        {
            if (fabs(regressedValues(sizeForStock / 2, is)) > tiny)
                BOOST_CHECK_CLOSE(contRead.getValue(ptStock, x.col(is)), regressedValues(sizeForStock / 2, is), p_accuracyInterp);
        }
    }
}


template<int NDIM>
void testHelper()
{
    vector<array<size_t, 4> >  indexB;
    indexB.reserve(3 * ((NDIM * (NDIM + 1)) / 2) + 1);
    boost::multi_array < int, 2> tabToIndexB(boost::extents [1 + NDIM] [1 + 2 * NDIM]);
    createLinearIndexB<NDIM>(indexB, tabToIndexB);

    BOOST_CHECK_EQUAL(indexB.size(), 3 * ((NDIM * (NDIM + 1)) / 2) + 1);

    vector<array<size_t, 6> >  indexA;
    indexA.reserve(1 + 3 * NDIM * NDIM + NDIM + 2 * (NDIM * (NDIM - 1) * (NDIM - 2) / 3));
    boost::multi_array < int, 3> tabToIndexA(boost::extents [1 + NDIM] [1 + NDIM] [1 + 2 * NDIM]);
    createLinearIndexA<NDIM>(indexA, tabToIndexA);
    BOOST_CHECK_EQUAL(indexA.size(),  1 + 4 * NDIM + 3 * NDIM * (NDIM - 1) + 2 * (NDIM * (NDIM - 1) * (NDIM - 2)) / 3) ;

}
// Certainly a compilo bug on mips
#if !defined(mips) && !defined(__mips__) && !defined(__mips)

BOOST_AUTO_TEST_CASE(testGridKernelHelper)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    // dimension 1
    testHelper<1>();
    // dimension 2
    testHelper<2>();
    // dimension 3
    testHelper<3>();
    // dimension 4
    testHelper<4>();
    // dimension 5
    testHelper<5>();
    // dimension 6
    testHelper<6>();

}

BOOST_AUTO_TEST_CASE(testGridKernel1D1)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    // proportion of particule to ,define bandwidth
    double prop = 0.25;

    int nbSimul = 100;

    double epsilon1 = 1e-6;
    bool bLin = true;
    // testDimension1D(bLin, nbSimul, prop, epsilon1);
    bLin = false;
    testDimension1D(bLin, nbSimul, prop, epsilon1);

}

BOOST_AUTO_TEST_CASE(testGridKernel2D1)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    // proportion of particule to ,define bandwidth
    double prop = 0.14;

    int nbSimul = 1000;

    double epsilon1 = 1e-6;
    bool bLin = true;
    // testDimension2D(bLin,nbSimul, prop, epsilon1);
    bLin = false;
    testDimension2D(bLin, nbSimul, prop, epsilon1);

}

BOOST_AUTO_TEST_CASE(testGridKernel3D1)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    // proportion of particule to ,define bandwidth
    double prop = 0.14;

    int nbSimul = 2000;

    double epsilon1 = 1e-6;
    bool bLin = true;
    // testDimension3D(bLin,nbSimul, prop, epsilon1);
    bLin = false;
    testDimension3D(bLin, nbSimul, prop, epsilon1);

}

BOOST_AUTO_TEST_CASE(testGridKernelFunc1D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    // proportion of particule to ,define bandwidth
    double prop = 0.14;

    int nbSimul = 200;

    double epsilon1 = 2e-6;
    bool bLin = true;
    testDimensionFonctionality1D(bLin, nbSimul, prop, epsilon1);
    bLin = false;
    testDimensionFonctionality1D(bLin, nbSimul, prop, epsilon1);

}

BOOST_AUTO_TEST_CASE(testGridKernelFunc2D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    // proportion of particule to ,define bandwidth
    double prop = 0.14;

    int nbSimul = 1000;

    double epsilon1 = 1e-6;
    bool bLin = true;
    testDimensionFonctionality2D(bLin, nbSimul, prop, epsilon1);
    bLin = false;
    testDimensionFonctionality2D(bLin, nbSimul, prop, epsilon1);
}


BOOST_AUTO_TEST_CASE(testContinuationKernel1D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    bool bLin = true;
    TestContinuationValue(bLin, 1, 200, 0.1, 1e-9, 0.0001);
    bLin = false;
    TestContinuationValue(bLin, 1, 200, 0.1, 1e-9, 0.0001);

}

BOOST_AUTO_TEST_CASE(testContinuationKernel2D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    bool bLin = true;
    TestContinuationValue(bLin, 2, 200, 0.1, 1e-9, 0.0001);
    bLin = false;
    TestContinuationValue(bLin, 2, 200, 0.1, 1e-9, 0.0001);
}

BOOST_AUTO_TEST_CASE(testContinuationKernel3D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    bool bLin = true;
    TestContinuationValue(bLin, 3, 500, 0.1, 1e-9, 0.0001);
    bLin = false;
    TestContinuationValue(bLin, 3, 500, 0.1, 1e-9, 0.0001);
}
BOOST_AUTO_TEST_CASE(testContinuationKernel4D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    bool bLin = true;
    TestContinuationValue(bLin, 4, 10000, 0.1, 1e-9, 0.0001);
    bLin = false;
    TestContinuationValue(bLin, 4, 10000, 0.1, 1e-9, 0.0001);
}

#else
BOOST_AUTO_TEST_CASE(testContinuationKernel1D)
{

}
#endif
