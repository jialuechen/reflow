// Copyright (C) 2017 EDF

// This code is published under the GNU Lesser General Public License (GNU LGPL)
#define BOOST_TEST_MODULE testLaplacianConstKernelRegression
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/timer/timer.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "libflow/regression/LaplacianConstKernelRegression.h"
#include "libflow/regression/LaplacianConstKernelRegressionGeners.h"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/regression/ContinuationValueGeners.h"

using namespace std;
using namespace Eigen;
using namespace libflow;
using namespace gs;

// utilities developed because VS2010 doesn't support auto


double accuracyEqual = 1e-9;

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

double KVal(const ArrayXd &x1, const ArrayXd &x2,  const ArrayXd   &h)
{
    double sum = 0;
    for (int i = 0; i < x1.size(); ++i)
        sum += fabs(x1(i) - x2(i)) / h(i);
    return exp(- sum);
}


ArrayXd naiveKernel(const ArrayXXd &p_x,  const ArrayXd &p_y, const ArrayXd &p_h)
{
    int nbSim = p_x.cols();
    ArrayXd  kernValY(nbSim);
    ArrayXd  kernVal(nbSim);
    for (int i = 0 ; i < nbSim ; ++i)
    {
        kernVal(i) = 0;
        kernValY(i) = 0;
        for (int j = 0; j < nbSim; ++j)
        {
            double kern = KVal(p_x.col(i), p_x.col(j), p_h);
            kernValY(i) += kern * p_y(j);
            kernVal(i) += kern;
        }
    }
    return kernValY / kernVal;
}

// test in dimension 1
void testDimension1D(const int &p_nbSimul, const ArrayXd &p_h,  const double &p_epsilon)
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
    ArrayXd regressedNaive = naiveKernel(x, y, p_h);

    // test regression object
    LaplacianConstKernelRegression kernelReg(false, x, p_h);

    ArrayXd regressed = kernelReg.getAllSimulations(y);

    // difference
    BOOST_CHECK_SMALL((regressed - regressedNaive).abs().maxCoeff(), p_epsilon);
}

// test in dimension 2
void testDimension2D(const int &p_nbSimul,  const ArrayXd &p_h, const double &p_epsilon)
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
    ArrayXd regressedNaive = naiveKernel(x, y, p_h);

    // test regression object
    LaplacianConstKernelRegression kernelReg(false, x, p_h);

    ArrayXd regressed = kernelReg.getAllSimulations(y);

    // difference
    BOOST_CHECK_SMALL((regressed - regressedNaive).abs().maxCoeff(), p_epsilon);
}


// test in dimension 3
void testDimension3D(const int &p_nbSimul,  const ArrayXd   &p_h, const double &p_epsilon)
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
    ArrayXd regressedNaive = naiveKernel(x, y, p_h);

    // test regression object
    LaplacianConstKernelRegression kernelReg(false, x, p_h);

    ArrayXd regressed = kernelReg.getAllSimulations(y);

    // difference
    BOOST_CHECK_SMALL((regressed - regressedNaive).abs().maxCoeff(), p_epsilon);
}

// test fonctionality in 1D
void testDimensionFonctionality1D(const int &p_nbSimul,  const ArrayXd &p_h, const double &p_epsilon)
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
    LaplacianConstKernelRegression kernelReg(false, x, p_h);

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
        BinaryFileArchive ar("archiveCKR", "w");
        ar << Record(kernelReg, "Regressor", "Top") ;
        ar.flush();
    }
    {
        BinaryFileArchive ar("archiveCKR", "r");
        LaplacianConstKernelRegression regAr;
        Reference<LaplacianConstKernelRegression> (ar, "Regressor", "Top").restore(0, &regAr);
        for (int is = 0; is < p_nbSimul / 10; ++is)
        {
            double regressAPoint = regAr.getValue(x.col(is), regressedGrid);
            BOOST_CHECK_SMALL(fabs(regressAPoint - regressed(is)), p_epsilon);
        }
    }
}



// test fonctionality in 2D
void testDimensionFonctionality2D(const int &p_nbSimul,  const ArrayXd &p_h, const double &p_epsilon)
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
    LaplacianConstKernelRegression kernelReg(false, x, p_h);

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
void TestContinuationValue(const int &p_nbSimul, const ArrayXd &p_h, const double &p_accuracyEqual, const double &p_accuracyInterp)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    int nDim = p_h.size();
    // nb stock points
    int sizeForStock  = 4;
    // generator Mersene Twister
    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    ArrayXXd x = ArrayXXd::Random(nDim, p_nbSimul);        // test archive
    ArrayXXd  regressedValues;
    {

        // second member to regress with one stock
        ArrayXXd toRegress(sizeForStock, p_nbSimul);
        for (int is  = 0;  is < p_nbSimul; ++is)
        {
            double prod = m1(x(0, is));
            for (int id = 1 ; id < nDim ; ++id)
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
        shared_ptr<LaplacianConstKernelRegression> localRegressor = make_shared<LaplacianConstKernelRegression>(false, x, p_h);

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
            BOOST_CHECK_CLOSE(regressedByContinuation(is), regressedValues(sizeForStock / 2, is), p_accuracyEqual);
            BOOST_CHECK_CLOSE(regressedByContinuation(is), regressedByContinuationSecond(is), p_accuracyInterp);
        }
        // default non compression
        BinaryFileArchive ar("archiveCKR1", "w");
        ar << Record(continuation, "FirstContinuation", "Top") ;
        ar.flush();
    }
    {
        // read archive
        BinaryFileArchive ar("archiveCKR1", "r");
        ContinuationValue contRead;
        Reference< ContinuationValue >(ar, "FirstContinuation", "Top").restore(0, &contRead);
        ArrayXd ptStock(1) ;
        ptStock(0) = sizeForStock / 2;
        boost::timer::auto_cpu_timer t;
        for (int is  = 0;  is < p_nbSimul; ++is)
        {
            BOOST_CHECK_CLOSE(contRead.getValue(ptStock, x.col(is)), regressedValues(sizeForStock / 2, is), p_accuracyInterp);
        }
    }
}


BOOST_AUTO_TEST_CASE(testLaplacianConstKernel1D1)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSimul = 100;

    double epsilon1 = 1e-6;

    ArrayXd h = ArrayXd::Constant(1, 0.1) ; // bandwidth

    testDimension1D(nbSimul, h, epsilon1);

}

BOOST_AUTO_TEST_CASE(testLaplacianConstKernel2D1)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSimul = 1000;

    double epsilon1 = 1e-6;

    ArrayXd h = ArrayXd::Constant(2, 0.1) ; // bandwidth

    testDimension2D(nbSimul, h, epsilon1);

}

BOOST_AUTO_TEST_CASE(testLaplacianConstKernel3D1)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSimul = 2000;

    double epsilon1 = 1e-6;

    ArrayXd h = ArrayXd::Constant(3, 0.1) ; // bandwidth

    testDimension3D(nbSimul, h, epsilon1);

}

BOOST_AUTO_TEST_CASE(testLaplacianConstKernelFunc1D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif


    int nbSimul = 200;

    double epsilon1 = 2e-6;

    ArrayXd h = ArrayXd::Constant(1, 0.1) ; // bandwidth

    testDimensionFonctionality1D(nbSimul, h, epsilon1);

}

BOOST_AUTO_TEST_CASE(testLaplacianConstKernelFunc2D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif


    int nbSimul = 1000;

    double epsilon1 = 1e-6;

    ArrayXd h = ArrayXd::Constant(2, 0.1) ; // bandwidth

    testDimensionFonctionality2D(nbSimul, h, epsilon1);
}


BOOST_AUTO_TEST_CASE(testContinuationLaplacianConstKernel1D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    ArrayXd h = ArrayXd::Constant(1, 0.1) ; // bandwidth
    TestContinuationValue(200, h, 1e-9, 0.0001);

}

BOOST_AUTO_TEST_CASE(testContinuationLaplacianConstKernel2D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    ArrayXd h = ArrayXd::Constant(2, 0.1) ; // bandwidth
    TestContinuationValue(200, h, 1e-9, 0.0001);
}

BOOST_AUTO_TEST_CASE(testContinuationLaplacianConstKernel3D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    ArrayXd h = ArrayXd::Constant(3, 0.1) ; // bandwidth
    TestContinuationValue(500, h, 1e-9, 0.0001);
}

