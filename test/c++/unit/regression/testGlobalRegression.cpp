
#define BOOST_TEST_MODULE testGlobalRegression
#define BOOST_TEST_DYN_LINK
#include <memory>
#include <boost/test/unit_test.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/function.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "reflow/core/utils/Polynomials1D.h"
#include "reflow/core/utils/constant.h"
#include "reflow/regression/GlobalRegressionGeners.h"
#include "reflow/regression/ContinuationValueGeners.h"

using namespace std;
using namespace Eigen;
using namespace gs;
using namespace reflow;


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
void testDimension(const bool &p_bRotate, const int &p_nDim, const int &p_nbSimul, const int &p_degree, const double &p_accuracyRegression)
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
        GlobalRegression<Hermite> globalRegressor(p_degree, p_nDim, p_bRotate);
        globalRegressor.updateSimulations(bZeroDate, x);
        ArrayXd  regressedValues = globalRegressor.getAllSimulations(toRegress);

        double average = toRegress.mean();
        for (int is = 0; is < p_nbSimul; ++is)
            if ((fabs(regressedValues(is)) > tiny) && (fabs(average) > tiny))
                BOOST_CHECK(fabs(regressedValues(is) - average) <  accuracyEqual * max(fabs(average), fabs(regressedValues(is))));
    }

    {
        // constructor not zero
        bool bZeroDate = 0;

        // constructor
        GlobalRegression<Hermite> globalRegressor(p_degree, p_nDim, p_bRotate);
        globalRegressor.updateSimulations(bZeroDate, x);
        // first regression
        ArrayXd  regressedValues = globalRegressor.getAllSimulations(toRegress);
        // then just calculate function basis coefficient
        ArrayXd regressedFuntionCoeff = globalRegressor.getCoordBasisFunction(toRegress);
        for (int is = 0; is < p_nbSimul; ++is)
        {
            Map<ArrayXd> xloc(x.col(is).data(), p_nDim);
            if (fabs(regressedValues(is)) > tiny)
                BOOST_CHECK_CLOSE(regressedValues(is), globalRegressor.getValue(xloc, regressedFuntionCoeff), accuracyEqual);
        }
        // Check that regression is correct
        double erMax = 0. ;
        for (int is = 0; is < p_nbSimul; ++is)
            erMax = std::max(erMax, std::fabs((regressedValues(is) - toReal(is)) / toReal(is)));
        BOOST_CHECK_CLOSE(1. + erMax, 1., p_accuracyRegression);
    }
}


// 1D check single second member regression
BOOST_AUTO_TEST_CASE(testGlobalRegression1D)
{

#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 1 ;
    // degree for polynomial approximation
    int nDegree = 3;
    int nbSimul = 1000000;
    // accuracy
    double accuracy = 5; // percentage error
    testDimension(false, nDim, nbSimul, nDegree, accuracy);
}

// 2D check single second member
BOOST_AUTO_TEST_CASE(testGlobalRegression2D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 2 ;
    // degree for polynomial approximation
    int nDegree = 3;
    int nbSimul = 1000000;
    // accuracy
    double accuracy = 5; // percentage error
    testDimension(false, nDim, nbSimul, nDegree, accuracy);
    // Certainly a compiler bug on mips
#if !defined(mips) && !defined(__mips__) && !defined(__mips)
    testDimension(true, nDim, nbSimul, nDegree, accuracy);
#endif
}

// 1D check multiple second members
BOOST_AUTO_TEST_CASE(testGlobalRegression1DMultipleSecondMember)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 1 ;
    // number of second member
    int nMember = 5 ;
    // degree
    int degree = 3 ;
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
        GlobalRegression<Hermite> globalRegressor(degree, nDim);
        globalRegressor.updateSimulations(bZeroDate, x);
        // regression
        ArrayXXd  regressedValues = globalRegressor.getAllSimulationsMultiple(toRegress);

        for (int imem = 0; imem < nMember; ++imem)
        {
            double average = toRegress.row(imem).mean();
            for (int is = 0; is < nbSimul; ++is)
                if (fabs(regressedValues(imem, is)) > tiny)
                    BOOST_CHECK_CLOSE(regressedValues(imem, is), average, accuracyEqual);
        }
    }
    // t is not zero
    {
        // constructor not zero
        bool bZeroDate = 0;
        // constructor
        GlobalRegression<Hermite> globalRegressor(degree, nDim);
        globalRegressor.updateSimulations(bZeroDate, x);
        // regression
        ArrayXXd  regressedValues = globalRegressor.getAllSimulationsMultiple(toRegress);
        // regression function coefficient
        ArrayXXd regressedFuntionCoeff = globalRegressor.getCoordBasisFunctionMultiple(toRegress);

        // now regress all second member and test
        for (int imem = 0 ; imem <  nMember; ++imem)
        {
            ArrayXd toRegress1D(nbSimul) ;
            toRegress1D = toRegress.row(imem);
            ArrayXd regressedValues1D = globalRegressor.getAllSimulations(toRegress1D);
            for (int is = 0; is < nbSimul; ++is)
                if (fabs(regressedValues(imem, is)) > tiny)
                    BOOST_CHECK_CLOSE(regressedValues(imem, is), regressedValues1D(is), accuracyEqual);
            ArrayXd regressedFuntionCoeff1D = globalRegressor.getCoordBasisFunction(toRegress1D);
            for (int im = 0; im < regressedFuntionCoeff1D.size(); ++im)
                if (fabs(regressedFuntionCoeff1D(im)) > tiny)
                    BOOST_CHECK_CLOSE(regressedFuntionCoeff(imem, im), regressedFuntionCoeff1D(im), accuracyEqual);
        }
    }
}




// test serialization GlobalRegression
void  TestGlobalRegressionSerialization(const int &p_nDim, const int &p_nbSimul, const int &p_degree)
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
        shared_ptr<GlobalRegression<Hermite> > globalRegressor = make_shared<GlobalRegression<Hermite> >(false, x, p_degree);

        // regress  directly with regressor
        regressedValues = globalRegressor->getAllSimulations(toRegress);

        // basis functions
        basisFunction =  globalRegressor->getCoordBasisFunction(toRegress);

        // test
        for (int is  = 0; is < p_nbSimul; ++is)
        {
            double regVal = globalRegressor->reconstructionASim(is, basisFunction);
            if (fabs(regressedValues(is)) > tiny)
                BOOST_CHECK_CLOSE(regressedValues(is), regVal, accuracyEqual);
        }

        // archive
        BinaryFileArchive ar("archiveGR", "w");
        ar << Record(*globalRegressor, "Regressor", "Top") ;
        ar.flush();
    }
    {
        BinaryFileArchive ar("archiveGR", "r");
        GlobalRegression<Hermite>  reg;
        Reference< GlobalRegression<Hermite> > (ar, "Regressor", "Top").restore(0, &reg);
        for (int is = 0; is < x.size(); ++is)
            if (fabs(regressedValues(is)) > tiny)
                BOOST_CHECK_CLOSE(regressedValues(is), reg.getValue(x.col(is), basisFunction), accuracyEqual);

    }
}

BOOST_AUTO_TEST_CASE(testGlobalRegressionSerialization)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    TestGlobalRegressionSerialization(1, 10, 1);
}

BOOST_AUTO_TEST_CASE(testCloneGlobalRegression)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // create
    int degree = 3 ;
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
        GlobalRegression<Hermite> globalRegressor(degree, 2);
        globalRegressor.updateSimulations(bZeroDate, x);
        ArrayXd  regressedValues = globalRegressor.getAllSimulations(toRegress);

        // now clone
        shared_ptr< BaseRegression> cloneRegressor = globalRegressor.clone();
        ArrayXd  regressedValuesClone = cloneRegressor->getAllSimulations(toRegress);

        for (int is = 0; is < 100; ++is)
            if (fabs(regressedValues(is)) > tiny)
                BOOST_CHECK_CLOSE(regressedValues(is), regressedValuesClone(is), accuracyEqual);
    }

    {
        // constructor not zero
        bool bZeroDate = 0;

        // constructor
        GlobalRegression<Hermite> globalRegressor(degree, 2);
        globalRegressor.updateSimulations(bZeroDate, x);
        // first regression
        ArrayXd  regressedValues = globalRegressor.getAllSimulations(toRegress);

        // now clone
        shared_ptr< BaseRegression> cloneRegressor = globalRegressor.clone();
        ArrayXd  regressedValuesClone = cloneRegressor->getAllSimulations(toRegress);

    }
}
