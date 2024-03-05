// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#define BOOST_TEST_MODULE testLocalSameSizeConstRegression
#define BOOST_TEST_DYN_LINK
#include <memory>
#include <functional>
#include <boost/test/unit_test.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "libflow/core/utils/constant.h"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/core/grids/LinearInterpolator.h"
#include "libflow/regression/LocalSameSizeConstRegressionGeners.h"
#include "libflow/regression/ContinuationValueGeners.h"

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

// utilities developed because VS2010 doesn't support auto
class Func1
{
public :
    Func1() {}

    double operator()(const double &p_x) const
    {
        return (1 + 0.1 * p_x) * (1 + 0.1 * p_x) + p_x + 2.;
    }
};

class Func2
{

public :

    Func2() {}

    double operator()(const int p_mem, const double &p_x) const
    {
        return  p_x * p_x + p_x * (p_mem + 1) + 1.;
    }
};

// test for different  dimensions
void testDimension(const int &p_nDim, const int &p_nbSimul, const int &p_nMesh, const double &p_accuracyRegression)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    // generator Mersene Twister
    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    std::function< double(double) > f = Func1();
    ArrayXXd x =  ArrayXXd::Random(p_nDim, p_nbSimul);
    // create the mesh
    double xMin = x.minCoeff() - tiny;
    double xMax = x.maxCoeff() + tiny;
    ArrayXd  lowValues = ArrayXd::Constant(p_nDim, xMin);
    ArrayXd  step = ArrayXd::Constant(p_nDim, (xMax - xMin) / p_nMesh);
    ArrayXi  nbStep = ArrayXi::Constant(p_nDim, p_nMesh);

    // second member to regress
    ArrayXd toRegress(p_nbSimul);
    ArrayXd toReal(p_nbSimul);
    for (int is  = 0;  is < p_nbSimul; ++is)
    {
        double prod = f(x(0, is));
        for (int id = 1 ; id < p_nDim ; ++id)
            prod *=  f(x(id, is));
        toRegress(is) = prod +  4 * normal_random();
        toReal(is) = prod;
    }
    // check if regression in t=0
    {
        // constructor if time t  = 0
        bool bZeroDate = true;
        // constructor
        LocalSameSizeConstRegression localRegressor(lowValues, step, nbStep);
        localRegressor.updateSimulations(bZeroDate, x);
        ArrayXd  regressedValues = localRegressor.getAllSimulations(toRegress);

        double average = toRegress.mean();
        for (int is = 0; is < p_nbSimul; ++is)
            BOOST_CHECK_CLOSE(regressedValues(is), average, accuracyEqual);
    }

    {
        // constructor not zero
        bool bZeroDate = 0;

        // constructor
        LocalSameSizeConstRegression localRegressor(lowValues, step, nbStep);
        localRegressor.updateSimulations(bZeroDate, x);
        // first regression
        ArrayXd  regressedValues = localRegressor.getAllSimulations(toRegress);
        // then just calculate function basis coefficient
        ArrayXd regressedFuntionCoeff = localRegressor.getCoordBasisFunction(toRegress);
        for (int is = 0; is < p_nbSimul; ++is)
        {
            Map<ArrayXd> xloc(x.col(is).data(), p_nDim);
            BOOST_CHECK_CLOSE(regressedValues(is), localRegressor.getValue(xloc, regressedFuntionCoeff), accuracyEqual);
        }
        // Check that regression is correct
        double erMax = 0. ;
        for (int is = 0; is < p_nbSimul; ++is)
        {
            erMax = std::max(erMax, std::fabs((regressedValues(is) - toReal(is)) / toReal(is)));
        }

        BOOST_CHECK_CLOSE(1. + erMax, 1., p_accuracyRegression);
    }
}


// 1D check single second member regression
BOOST_AUTO_TEST_CASE(testLocalRegression1D)
{

#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 1 ;
    // Number of simulations and parameters for regression.
    int nMesh = 16 ;
    ArrayXi  nbMesh = ArrayXi::Constant(nDim, nMesh);
    int nbSimul = 1000000;
    // accuracy
    double accuracy = 6; // percentage error
    testDimension(nDim, nbSimul, nMesh, accuracy);
}

// 2D check single second member
BOOST_AUTO_TEST_CASE(testLocalRegression2D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 2 ;
    // Number of simulations and parameters for regression.
    int nMesh = 30 ;

    int nbSimul = 5000000;
    // accuracy
    double accuracy = 10; // percentage error
    testDimension(nDim, nbSimul, nMesh, accuracy);
}


// 1D check multiple second members
BOOST_AUTO_TEST_CASE(testLocalRegression1DMultipleSecondMember)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 1 ;
    // number of second member
    int nMember = 5 ;
    // Number of simulations and parameters for regression.
    int nMesh = 20 ;
    // create the mesh
    ArrayXd  lowValues = ArrayXd::Constant(nDim, 0.);
    ArrayXd  step = ArrayXd::Constant(nDim, 1. / nMesh);
    ArrayXi  nbStep = ArrayXi::Constant(nDim, nMesh);
    int nbSimul = 10000;
    // generator Mersene Twister
    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    std::function< double(int, double) >  f = Func2();
    ArrayXXd x = ArrayXXd::Random(nDim, nbSimul);
    // second member to regress
    ArrayXXd toRegress(nMember, nbSimul);
    for (int imem = 0; imem < nMember; ++imem)
        for (int is  = 0;  is < nbSimul; ++is)
            toRegress(imem, is) = f(imem, x(0, is)) +  normal_random();

    // check if regression in t=0
    {
        // constructor if time t  = 0
        bool bZeroDate = true;
        // constructor
        LocalSameSizeConstRegression localRegressor(lowValues, step, nbStep);
        localRegressor.updateSimulations(bZeroDate, x);
        // regression
        ArrayXXd  regressedValues = localRegressor.getAllSimulationsMultiple(toRegress);

        for (int imem = 0; imem < nMember; ++imem)
        {
            double average = toRegress.row(imem).mean();
            for (int is = 0; is < nbSimul; ++is)
                BOOST_CHECK_CLOSE(regressedValues(imem, is), average, accuracyEqual);
        }
    }
    // t is not zero
    {
        // constructor not zero
        bool bZeroDate = 0;
        // constructor
        LocalSameSizeConstRegression localRegressor(bZeroDate, x, lowValues, step, nbStep);
        // regression
        ArrayXXd  regressedValues = localRegressor.getAllSimulationsMultiple(toRegress);
        // regression function coefficient
        ArrayXXd regressedFuntionCoeff = localRegressor.getCoordBasisFunctionMultiple(toRegress);

        // now regress all second member and test
        for (int imem = 0 ; imem <  nMember; ++imem)
        {
            ArrayXd toRegress1D(nbSimul) ;
            toRegress1D = toRegress.row(imem);
            ArrayXd regressedValues1D = localRegressor.getAllSimulations(toRegress1D);
            for (int is = 0; is < nbSimul; ++is)
                BOOST_CHECK_CLOSE(regressedValues(imem, is), regressedValues1D(is), accuracyEqual);
            ArrayXd regressedFuntionCoeff1D = localRegressor.getCoordBasisFunction(toRegress1D);
            for (int im = 0; im < regressedFuntionCoeff1D.size(); ++im)
                BOOST_CHECK_CLOSE(regressedFuntionCoeff(imem, im), regressedFuntionCoeff1D(im), accuracyEqual);
        }
    }
}




// test serialization LocalSameSizeConstRegression
void  TestLocalRegressionSerialization(const int &p_nDim, const int &p_nbSimul, const int &p_nbMesh)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // create the mesh
    ArrayXd  lowValues = ArrayXd::Constant(p_nDim, 0.);
    ArrayXd  step = ArrayXd::Constant(p_nDim, 1. / p_nbMesh);
    ArrayXi  nbStep = ArrayXi::Constant(p_nDim, p_nbMesh);

    // generator Mersene Twister
    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    std::function< double(double) > f = Func1();
    ArrayXXd  x = ArrayXXd::Random(p_nDim, p_nbSimul);
    // to store basis function
    ArrayXd basisFunction, regressedValues	;
    // test archive
    {
        // second member to regress with one stock
        ArrayXd toRegress(p_nbSimul);
        for (int is  = 0;  is < p_nbSimul; ++is)
        {
            double prod = f(x(0, is));
            for (int id = 1 ; id < p_nDim ; ++id)
                prod *=  f(x(id, is));
            double uncertainty = 4 * normal_random();
            toRegress(is) = prod +  uncertainty;
        }
        // conditional expectation
        shared_ptr<LocalSameSizeConstRegression> localRegressor = make_shared<LocalSameSizeConstRegression>(false, x, lowValues, step, nbStep);

        // regress  directly with regressor
        regressedValues = localRegressor->getAllSimulations(toRegress);

        // basis functions
        basisFunction =  localRegressor->getCoordBasisFunction(toRegress);

        // test
        for (int is  = 0; is < p_nbSimul; ++is)
        {
            double regVal = localRegressor->reconstructionASim(is, basisFunction);
            BOOST_CHECK_CLOSE(regressedValues(is), regVal, accuracyEqual);
        }

        // archive
        BinaryFileArchive ar("archiveLSS", "w");
        ar << Record(*localRegressor, "Regressor", "Top") ;
    }
    {
        BinaryFileArchive ar("archiveLSS", "r");
        LocalSameSizeConstRegression reg;
        Reference< LocalSameSizeConstRegression> (ar, "Regressor", "Top").restore(0, &reg);
        for (int is = 0; is < x.size(); ++is)
            BOOST_CHECK_CLOSE(regressedValues(is), reg.getValue(x.col(is), basisFunction), accuracyEqual);

    }
}

BOOST_AUTO_TEST_CASE(testLocalRegressionSerialization)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    TestLocalRegressionSerialization(1, 10, 1);
}

// test serialization for continuation values
void TestContinuationValue(const int &p_nDim, const int &p_nbSimul, const int &p_nbMesh)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // nb stock points
    int sizeForStock  = 4;
    // create the mesh
    ArrayXd  lowValuesStoc = ArrayXd::Constant(p_nDim, 0.);
    ArrayXd  stepStoc = ArrayXd::Constant(p_nDim, 1. / p_nbMesh);
    ArrayXi  nbStepStoc = ArrayXi::Constant(p_nDim, p_nbMesh);

    // generator Mersene Twister
    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    std::function< double(double) > f = Func1();
    ArrayXXd x =  ArrayXXd::Random(p_nDim, p_nbSimul);     // test archive
    ArrayXXd  regressedValues;
    {

        // second member to regress with one stock
        ArrayXXd toRegress(sizeForStock, p_nbSimul);
        for (int is  = 0;  is < p_nbSimul; ++is)
        {
            double prod = f(x(0, is));
            for (int id = 1 ; id < p_nDim ; ++id)
                prod *=  f(x(id, is));
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
        shared_ptr<LocalSameSizeConstRegression> localRegressor = make_shared<LocalSameSizeConstRegression>(false, x, lowValuesStoc, stepStoc, nbStepStoc);

        // regress  directly with regressor
        regressedValues = localRegressor->getAllSimulationsMultiple(toRegress);
        // creation continuation value object
        ContinuationValue  continuation(regular, localRegressor,  toRegress.transpose());

        // regress with continuation value object
        ArrayXd ptStock(1) ;
        ptStock(0) = sizeForStock / 2;
        ArrayXd regressedByContinuation = continuation.getAllSimulations(ptStock);
        for (int is  = 0;  is < p_nbSimul; ++is)
            BOOST_CHECK_CLOSE(regressedByContinuation(is), regressedValues(sizeForStock / 2, is), accuracyEqual);

        // default non compression
        BinaryFileArchive ar("archiveLSS1", "w");
        ar << Record(continuation, "FirstContinuation", "Top") ;

    }
    {
        // read archive
        BinaryFileArchive ar("archiveLSS1", "r");
        ContinuationValue contRead;
        Reference< ContinuationValue >(ar, "FirstContinuation", "Top").restore(0, &contRead);
        ArrayXd ptStock(1) ;
        ptStock(0) = sizeForStock / 2;
        for (int is  = 0;  is < p_nbSimul; ++is)
            BOOST_CHECK_CLOSE(contRead.getValue(ptStock, x.col(is)), regressedValues(sizeForStock / 2, is), accuracyEqual);
    }
}



BOOST_AUTO_TEST_CASE(testContinuation)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    TestContinuationValue(1, 10, 1);
}


BOOST_AUTO_TEST_CASE(testCloneLocal)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // create the mesh
    int nbMesh = 2;
    // create the mesh
    ArrayXd  lowValues = ArrayXd::Constant(2, 0.);
    ArrayXd  step = ArrayXd::Constant(2, 1. / nbMesh);
    ArrayXi  nbStep = ArrayXi::Constant(2, nbMesh);
    // generator Mersene Twister
    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    std::function< double(double) > f = Func1();
    ArrayXXd x = ArrayXXd::Random(2, 100);
    // second member to regress
    ArrayXd toRegress(100);
    ArrayXd toReal(100);
    for (int is  = 0;  is < 100; ++is)
    {
        double prod = f(x(0, is));
        for (int id = 1 ; id < 2 ; ++id)
            prod *=  f(x(id, is));
        toRegress(is) = prod +  4 * normal_random();
        toReal(is) = prod;
    }
    // check if regression in t=0
    {
        // constructor if time t  = 0
        bool bZeroDate = true;
        // constructor
        LocalSameSizeConstRegression localRegressor(lowValues, step, nbStep);
        localRegressor.updateSimulations(bZeroDate, x);
        ArrayXd  regressedValues = localRegressor.getAllSimulations(toRegress);

        // now clone
        shared_ptr< BaseRegression> cloneRegressor = localRegressor.clone();
        ArrayXd  regressedValuesClone = cloneRegressor->getAllSimulations(toRegress);

        for (int is = 0; is < 100; ++is)
            BOOST_CHECK_CLOSE(regressedValues(is), regressedValuesClone(is), accuracyEqual);
    }

    {
        // constructor not zero
        bool bZeroDate = 0;

        // constructor
        LocalSameSizeConstRegression localRegressor(lowValues, step, nbStep);
        localRegressor.updateSimulations(bZeroDate, x);
        // first regression
        ArrayXd  regressedValues = localRegressor.getAllSimulations(toRegress);
        // now clone
        shared_ptr< BaseRegression> cloneRegressor = localRegressor.clone();
        ArrayXd  regressedValuesClone = cloneRegressor->getAllSimulations(toRegress);

        for (int is = 0; is < 100; ++is)
            BOOST_CHECK_CLOSE(regressedValues(is), regressedValuesClone(is), accuracyEqual);


    }
}
