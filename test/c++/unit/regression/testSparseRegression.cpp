
#define BOOST_TEST_MODULE testSparseRegression
#define BOOST_TEST_DYN_LINK
#include <array>
#include <memory>
#include <boost/test/unit_test.hpp>
#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/function.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "reflow/core/grids/RegularSpaceGridGeners.h"
#include "reflow/regression/SparseRegressionGeners.h"
#include "reflow/regression/ContinuationValueGeners.h"

using namespace std;
using namespace Eigen;
using namespace gs;
using namespace reflow;



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

double accuracyEqual = 1e-7;

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

/// test for different  dimensions
/// \param p_nDim                   dimension of the problem
/// \param p_nbSimul                number of simulations used
/// \param p_level                  level of the sparse grid
/// \param p_degree                 degree for the interpolation
/// \param p_accuracyRegression     accuracy
void testDimension(const int &p_nDim, const int &p_nbSimul, const int &p_level,  const int &p_degree, const double &p_accuracyRegression)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXd weight = ArrayXd::Constant(p_nDim, 1.);
    // generator Mersene Twister
    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    std::function< double(double) > f = Func1();
    ArrayXXd x =  ArrayXXd::Random(p_nDim, p_nbSimul);
    // second member to regress
    ArrayXd toRegress(p_nbSimul);
    ArrayXd toReal(p_nbSimul);
    for (int is  = 0;  is < p_nbSimul; ++is)
    {
        double prod = f(x(0, is));
        for (int id = 1 ; id < p_nDim ; ++id)
            prod *=  f(x(id, is));
        toRegress(is) = prod +   normal_random();
        toReal(is) = prod;
    }
    // check if regression in t=0
    {
        // constructor if time t  = 0
        bool bZeroDate = true;
        // constructor
        SparseRegression sparseRegressor(p_level, weight, p_degree);
        sparseRegressor.updateSimulations(bZeroDate, x);
        ArrayXd  regressedValues = sparseRegressor.getAllSimulations(toRegress);

        double average = toRegress.mean();
        for (int is = 0; is < p_nbSimul; ++is)
            BOOST_CHECK_CLOSE(regressedValues(is), average, accuracyEqual);
    }

    {
        // constructor not zero
        bool bZeroDate = 0;

        // constructor
        SparseRegression sparseRegressor(p_level, weight, p_degree);
        sparseRegressor.updateSimulations(bZeroDate, x);
        // first regression
        ArrayXd  regressedValues = sparseRegressor.getAllSimulations(toRegress);
        // then just calculate function basis coefficient
        ArrayXd regressedFuntionCoeff = sparseRegressor.getCoordBasisFunction(toRegress);
        for (int is = 0; is < p_nbSimul; ++is)
        {
            Map<ArrayXd> xloc(x.col(is).data(), p_nDim);
            BOOST_CHECK_CLOSE(regressedValues(is), sparseRegressor.getValue(xloc, regressedFuntionCoeff), accuracyEqual);
        }
        // Check that regression is correct
        double erMax = 0. ;
        for (int is = 0; is < p_nbSimul; ++is)
            erMax = std::max(erMax, std::fabs((regressedValues(is) - toReal(is)) / toReal(is)));
        BOOST_CHECK_CLOSE(1. + erMax, 1., p_accuracyRegression);

    }
}


// 1D check single second member regression
BOOST_AUTO_TEST_CASE(testSparseRegression1D)
{

#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 1 ;
    int nbSimul = 100000;
    // accuracy
    double accuracy = 5; // percentage error
    // level 5
    int level = 5;
    // degree
    int degree = 1;
    testDimension(nDim, nbSimul, level, degree, accuracy);
    degree = 2;
    testDimension(nDim, nbSimul, level, degree, accuracy);
    degree = 3;
    testDimension(nDim, nbSimul, level, degree, accuracy);
}

/// 2D check single second member
BOOST_AUTO_TEST_CASE(testSparseRegression2D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 2 ;
    int nbSimul = 100000;
    // accuracy
    double accuracy = 5; // percentage error
    // level 5
    int level = 4;
    // degree
    int degree = 1;
    testDimension(nDim, nbSimul, level, degree, accuracy);
    degree = 2;
    testDimension(nDim, nbSimul, level, degree, accuracy);
    degree = 3;
    testDimension(nDim, nbSimul, level, degree, accuracy);

}

/// 3D check single second member
BOOST_AUTO_TEST_CASE(testSparseRegression3D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 3 ;
    // Number of simulations and parameters for regression.
    int nbSimul = 100000;
    // accuracy
    double accuracy = 5; // percentage error
    // level 5
    int level = 4;
    // degree
    int degree = 1;
    testDimension(nDim, nbSimul, level, degree, accuracy);
    degree = 2;
    testDimension(nDim, nbSimul, level, degree, accuracy);
    degree = 3;
    testDimension(nDim, nbSimul, level, degree, accuracy);

}

// 1D check multiple second members
BOOST_AUTO_TEST_CASE(testSparseRegression1DMultipleSecondMember)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 1 ;
    // number of second member
    int nMember = 5 ;
    // Number of simulations and parameters for regression.
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

    int level = 4;
    ArrayXd weight  = ArrayXd::Constant(1, 1.);
    int degree = 1;
    // check if regression in t=0
    {
        // constructor if time t  = 0
        bool bZeroDate = true;
        // constructor
        SparseRegression sparseRegressor(level, weight, degree);
        sparseRegressor.updateSimulations(bZeroDate, x);
        // regression
        ArrayXXd  regressedValues = sparseRegressor.getAllSimulationsMultiple(toRegress);

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
        SparseRegression sparseRegressor(bZeroDate, x, level, weight, degree);
        // regression
        ArrayXXd  regressedValues = sparseRegressor.getAllSimulationsMultiple(toRegress);
        // regression function coefficient
        ArrayXXd regressedFuntionCoeff = sparseRegressor.getCoordBasisFunctionMultiple(toRegress);

        // now regress all second member and test
        for (int imem = 0 ; imem <  nMember; ++imem)
        {
            ArrayXd toRegress1D(nbSimul) ;
            toRegress1D = toRegress.row(imem);
            ArrayXd regressedValues1D = sparseRegressor.getAllSimulations(toRegress1D);
            for (int is = 0; is < nbSimul; ++is)
                BOOST_CHECK_CLOSE(regressedValues(imem, is), regressedValues1D(is), accuracyEqual);
            ArrayXd regressedFuntionCoeff1D = sparseRegressor.getCoordBasisFunction(toRegress1D);
            for (int im = 0; im < regressedFuntionCoeff1D.size(); ++im)
                BOOST_CHECK_CLOSE(regressedFuntionCoeff(imem, im), regressedFuntionCoeff1D(im), accuracyEqual);
        }
    }
}




// test serialization Sparse Regression
/// \param p_nDim                   dimension of the problem
/// \param p_nbSimul                number of simulations used
/// \param p_level                  level of the sparse grid
/// \param p_degree                 degree for the interpolation
void  TestSparseRegressionSerialization(const int &p_nDim, const int &p_nbSimul,  const int &p_level,  const int &p_degree)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // generator Mersene Twister
    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    std::function< double(double) > f = Func1();
    ArrayXXd x = ArrayXXd::Random(p_nDim, p_nbSimul);
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
        ArrayXd weight = ArrayXd::Constant(p_nDim, 1.);
        shared_ptr<SparseRegression> sparseRegressor = make_shared<SparseRegression>(false, x, p_level, weight, p_degree);

        // regress  directly with regressor
        regressedValues = sparseRegressor->getAllSimulations(toRegress);

        // basis functions
        basisFunction =  sparseRegressor->getCoordBasisFunction(toRegress);

        // test
        for (int is  = 0; is < p_nbSimul; ++is)
        {
            double regVal = sparseRegressor->reconstructionASim(is, basisFunction);
            BOOST_CHECK_CLOSE(regressedValues(is), regVal, accuracyEqual);
        }

        // archive
        BinaryFileArchive ar("archiveSR", "w");
        ar << Record(*sparseRegressor, "Regressor", "Top") ;
    }
    {
        BinaryFileArchive ar("archiveSR", "r");
        SparseRegression reg;
        Reference< SparseRegression> (ar, "Regressor", "Top").restore(0, &reg);
        for (int is = 0; is < x.size(); ++is)
            BOOST_CHECK_CLOSE(regressedValues(is), reg.getValue(x.col(is), basisFunction), accuracyEqual);

    }
}

BOOST_AUTO_TEST_CASE(testSparseRegressionSerialization)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    TestSparseRegressionSerialization(1, 1000, 4, 1);
}

// test serialization for continuation values
/// \param p_nDim                   dimesnion of the problem
/// \param p_nbSimul                number of simulations used
/// \param p_level                  level of the sparse grid
/// \param p_degree                 degree for the interpolation
void TestContinuationValue(const int &p_nDim, const int &p_nbSimul, const int &p_level,  const int &p_degree)
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
    std::function< double(double) > f = Func1();
    ArrayXXd x = ArrayXXd::Random(p_nDim, p_nbSimul);     // test archive
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
        ArrayXd weight = ArrayXd::Constant(p_nDim, 1.);
        shared_ptr<SparseRegression> sparseRegressor = make_shared<SparseRegression>(false, x, p_level, weight, p_degree);

        // regress  directly with regressor
        regressedValues = sparseRegressor->getAllSimulationsMultiple(toRegress);
        // creation continuation value object
        ContinuationValue  continuation(regular, sparseRegressor,  toRegress.transpose());

        // regress with continuation value object
        ArrayXd ptStock(1) ;
        ptStock(0) = sizeForStock / 2;
        ArrayXd regressedByContinuation = continuation.getAllSimulations(ptStock);
        for (int is  = 0;  is < p_nbSimul; ++is)
            BOOST_CHECK_CLOSE(regressedByContinuation(is), regressedValues(sizeForStock / 2, is), accuracyEqual);

        // default non compression
        BinaryFileArchive ar("archiveSR1", "w");
        ar << Record(continuation, "FirstContinuation", "Top") ;
    }
    {
        // read archive
        BinaryFileArchive ar("archiveSR1", "r");
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

    TestContinuationValue(1, 10, 4, 1);
}

BOOST_AUTO_TEST_CASE(testCloneSparseRegression)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXd weight = ArrayXd::Constant(2, 1.);
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
        toRegress(is) = prod +   normal_random();
        toReal(is) = prod;
    }
    // check if regression in t=0
    {
        // constructor if time t  = 0
        bool bZeroDate = true;
        // constructor
        SparseRegression sparseRegressor(2, weight, 1);
        sparseRegressor.updateSimulations(bZeroDate, x);
        ArrayXd  regressedValues = sparseRegressor.getAllSimulations(toRegress);

        // now clone
        shared_ptr< BaseRegression> cloneRegressor = sparseRegressor.clone();
        ArrayXd  regressedValuesClone = cloneRegressor->getAllSimulations(toRegress);

        for (int is = 0; is < 100; ++is)
            BOOST_CHECK_CLOSE(regressedValues(is), regressedValuesClone(is), accuracyEqual);
    }

    {
        // constructor not zero
        bool bZeroDate = 0;

        // constructor
        SparseRegression sparseRegressor(2, weight, 1);
        sparseRegressor.updateSimulations(bZeroDate, x);
        // first regression
        ArrayXd  regressedValues = sparseRegressor.getAllSimulations(toRegress);

        // now clone
        shared_ptr< BaseRegression> cloneRegressor = sparseRegressor.clone();
        ArrayXd  regressedValuesClone = cloneRegressor->getAllSimulations(toRegress);

        for (int is = 0; is < 100; ++is)
            BOOST_CHECK_CLOSE(regressedValues(is), regressedValuesClone(is), accuracyEqual);

    }
}
