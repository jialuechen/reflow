
#define BOOST_TEST_MODULE testLocalDiscrLastDimRegression
#define BOOST_TEST_DYN_LINK
#include <memory>
#include <functional>
#include <boost/test/unit_test.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <Eigen/Dense>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/regression/LocalConstDiscrLastDimRegression.h"
#include "libflow/regression/LocalLinearDiscrLastDimRegression.h"
#include "libflow/regression/LocalConstDiscrLastDimRegressionGeners.h"
#include "libflow/regression/LocalLinearDiscrLastDimRegressionGeners.h"
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
double accuracyNearEqual = 0.5;

#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif

// utilities developed because VS2010 doesn't support auto
class Func1
{
private:
    int m_iDisc;
public :
    Func1(const int    &p_iDisc): m_iDisc(p_iDisc) {}

    double operator()(const double &p_x) const
    {
        return (1 + m_iDisc + 0.1 * m_iDisc * p_x) * (1 + m_iDisc + 0.1 * p_x) + p_x + 2. + m_iDisc;
    }
};

// Test both continuous regression for first dimension, depending on a discrete random variable on last dimension
// Different  dimensions p_nDim for continuous uncertainties
// Different  values of the discrete state  with cardinal  p_nbDiscr
template< class Regressor>
void testDimension(const int &p_nDim, const int &p_nbDiscr, const int &p_nbSimul, const int &p_nMesh, const double &p_accuracyRegression)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // create the mesh
    ArrayXi  nbMesh = ArrayXi::Constant(p_nDim + 1, p_nMesh);
    nbMesh(p_nDim) = p_nbDiscr;
    // generator Mersene Twister
    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    boost::random::uniform_int_distribution<> alea_uI(0, p_nbDiscr - 1);
    boost::variate_generator<boost::mt19937 &, boost::random::uniform_int_distribution<> > uniformInt_random(generator, alea_uI);
    boost::random::uniform_real_distribution<double> alea_uD(0., 1.);
    boost::variate_generator<boost::mt19937 &, boost::random::uniform_real_distribution<double> > uniform_random(generator, alea_uD);

    std::vector< std::function< double(double) > > fList(p_nbDiscr);
    for (int id = 0; id < p_nbDiscr; ++id)
        fList[id] = Func1(id);
    ArrayXXd x = ArrayXXd::Zero(p_nDim + 1, p_nbSimul);
    // second member to regress
    ArrayXd toRegress(p_nbSimul);
    ArrayXd toReal(p_nbSimul);
    for (int is  = 0;  is < p_nbSimul; ++is)
    {
        for (int id = 0; id < p_nDim; ++id)
        {
            x(id, is) = uniform_random();
        }
        int iDis = uniformInt_random();
        x(p_nDim, is) =  static_cast<double>(iDis);
        // function to regress depends on the discrete value on last dimension
        double prod = fList[iDis](x(0, is));
        for (int id = 1 ; id < p_nDim ; ++id)
            prod *=  fList[iDis](x(id, is));
        toRegress(is) = prod +  2 * normal_random();
        toReal(is) = prod;
    }

    // constructor not zero date
    bool bZeroDate = 0;

    // constructor
    Regressor localRegressor(nbMesh);
    localRegressor.updateSimulations(bZeroDate, x);
    // first regression
    ArrayXd  regressedValues = localRegressor.getAllSimulations(toRegress);
    // then just calculate function basis coefficient
    ArrayXd regressedFuntionCoeff = localRegressor.getCoordBasisFunction(toRegress);
    for (int is = 0; is < p_nbSimul; ++is)
    {
        Map<ArrayXd> xloc(x.col(is).data(), p_nDim + 1);
        BOOST_CHECK_CLOSE(regressedValues(is), localRegressor.getValue(xloc, regressedFuntionCoeff), accuracyNearEqual);
    }
    // Check that regression is correct
    double erMax = 0. ;
    for (int is = 0; is < p_nbSimul; ++is)
        erMax = std::max(erMax, std::fabs((regressedValues(is) - toReal(is)) / toReal(is)));
    // now get back all
    BOOST_CHECK_CLOSE(1. + erMax, 1., p_accuracyRegression);
    // get back all values once for all
    ArrayXd regressedAllValues  = localRegressor.getValues(x, regressedFuntionCoeff) ;
    for (int is = 0; is < p_nbSimul; ++is)
    {
        BOOST_CHECK_CLOSE(regressedValues(is), regressedAllValues(is), accuracyNearEqual);
    }
}


// 1D check single second member regression
BOOST_AUTO_TEST_CASE(testLocalDiscrRegression1D)
{

#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 1 ;
    // number of discrete values
    int nbDiscr = 2;
    // Number of simulations and parameters for regression.
    int nMeshConst = 16 ; // ;
    int nMeshLinear = 8 ;
    int nbSimul = 1000000; //0900;
    // accuracy
    double accuracy = 2; // percentage error
    testDimension<LocalConstDiscrLastDimRegression>(nDim, nbDiscr, nbSimul, nMeshConst, accuracy);
    testDimension<LocalLinearDiscrLastDimRegression>(nDim, nbDiscr, nbSimul, nMeshLinear, accuracy);
    nbDiscr = 4;
    testDimension<LocalConstDiscrLastDimRegression>(nDim, nbDiscr, nbSimul, nMeshConst, accuracy);
    testDimension<LocalLinearDiscrLastDimRegression>(nDim, nbDiscr, nbSimul, nMeshLinear, accuracy);
}

// 2D check single second member regression
BOOST_AUTO_TEST_CASE(testLocalDiscrRegression2D)
{

#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 1 ;
    // number of discrete values
    int nbDiscr = 2;
    // Number of simulations and parameters for regression.
    int nMeshConst = 16 ; // ;
    int nMeshLinear = 8 ;
    int nbSimul = 1000000; //0900;
    // accuracy
    double accuracy = 5; // percentage error
    testDimension<LocalConstDiscrLastDimRegression>(nDim, nbDiscr, nbSimul, nMeshConst, accuracy);
    testDimension<LocalLinearDiscrLastDimRegression>(nDim, nbDiscr, nbSimul, nMeshLinear, accuracy);
    nbDiscr = 4;
    testDimension<LocalConstDiscrLastDimRegression>(nDim, nbDiscr, nbSimul, nMeshConst, accuracy);
    testDimension<LocalLinearDiscrLastDimRegression>(nDim, nbDiscr, nbSimul, nMeshLinear, accuracy);
}


template< class Regressor>
void testMulipleRegression(const int &p_nDim, const int &p_nbDiscr, const int &p_nMember, const int &p_nbSimul, const int &p_nMesh)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // create the mesh
    ArrayXi  nbMesh = ArrayXi::Constant(p_nDim + 1, p_nMesh);
    nbMesh(p_nDim) = p_nbDiscr;
    // generator Mersene Twister
    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    boost::random::uniform_int_distribution<> alea_uI(0, p_nbDiscr - 1);
    boost::variate_generator<boost::mt19937 &, boost::random::uniform_int_distribution<> > uniformInt_random(generator, alea_uI);
    boost::random::uniform_real_distribution<double> alea_uD(0., 1.);
    boost::variate_generator<boost::mt19937 &, boost::random::uniform_real_distribution<double> > uniform_random(generator, alea_uD);


    std::vector< std::function< double(double) > > fList(p_nbDiscr);
    for (int id = 0; id < p_nbDiscr; ++id)
        fList[id] = Func1(id);
    ArrayXXd x = ArrayXXd::Zero(p_nDim + 1, p_nbSimul);
    // second member to regress
    ArrayXXd toRegress(p_nMember, p_nbSimul);

    for (int imem = 0; imem < p_nMember; ++imem)
        for (int is  = 0;  is < p_nbSimul; ++is)
        {
            for (int id = 0; id < p_nDim; ++id)
            {
                x(id, is) = uniform_random();
            }
            int iDis = uniformInt_random();
            x(p_nDim, is) =  static_cast<double>(iDis);
            // function to regress depends on the discrete value on last dimension
            double prod = fList[iDis](x(0, is));
            for (int id = 1 ; id < p_nDim ; ++id)
                prod *=  fList[iDis](x(id, is));
            toRegress(imem, is) = prod +  2 * normal_random();
        }

    // constructor not zero
    bool bZeroDate = 0;
    // constructor
    Regressor localRegressor(bZeroDate, x, nbMesh);
    // regression
    ArrayXXd  regressedValues = localRegressor.getAllSimulationsMultiple(toRegress);
    // regression function coefficient
    ArrayXXd regressedFuntionCoeff = localRegressor.getCoordBasisFunctionMultiple(toRegress);

    // now regress all second member and test
    for (int imem = 0 ; imem <  p_nMember; ++imem)
    {
        ArrayXd toRegress1D(p_nbSimul) ;
        toRegress1D = toRegress.row(imem);
        ArrayXd regressedValues1D = localRegressor.getAllSimulations(toRegress1D);
        for (int is = 0; is < p_nbSimul; ++is)
            BOOST_CHECK_CLOSE(regressedValues(imem, is), regressedValues1D(is), accuracyEqual);
        ArrayXd regressedFuntionCoeff1D = localRegressor.getCoordBasisFunction(toRegress1D);
        for (int im = 0; im < regressedFuntionCoeff1D.size(); ++im)
            BOOST_CHECK_CLOSE(regressedFuntionCoeff(imem, im), regressedFuntionCoeff1D(im), accuracyEqual);
    }
}

BOOST_AUTO_TEST_CASE(testLocalRegression1DMultipleSecondMember)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 1 ;
    // number of second member
    int nMember = 5 ;
    int nbDiscr = 2;
    int nbSimul = 100000;
    int nbMesh = 4 ;
    // test
    testMulipleRegression<LocalConstDiscrLastDimRegression>(nDim, nbDiscr, nMember, nbSimul, nbMesh);
    testMulipleRegression<LocalLinearDiscrLastDimRegression>(nDim, nbDiscr, nMember, nbSimul, nbMesh);
}

// test serialization
template< class Regressor>
void  TestLocalDiscrLastDimRegressionSerialization(const int &p_nDim,  const int &p_nbDiscr, const int &p_nbSimul, const int &p_nbMesh)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // create the mesh
    ArrayXi  nbMesh = ArrayXi::Constant(p_nDim + 1, p_nbMesh);
    nbMesh(p_nDim) = p_nbDiscr;
    // generator Mersene Twister
    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    boost::random::uniform_int_distribution<> alea_uI(0, p_nbDiscr - 1);
    boost::variate_generator<boost::mt19937 &, boost::random::uniform_int_distribution<> > uniformInt_random(generator, alea_uI);
    boost::random::uniform_real_distribution<double> alea_uD(0., 1.);
    boost::variate_generator<boost::mt19937 &, boost::random::uniform_real_distribution<double> > uniform_random(generator, alea_uD);
    std::vector< std::function< double(double) > > fList(p_nbDiscr);
    for (int id = 0; id < p_nbDiscr; ++id)
        fList[id] = Func1(id);
    ArrayXXd x = ArrayXXd::Zero(p_nDim + 1, p_nbSimul);
    // second member to regress
    ArrayXd toRegress(p_nbSimul);
    // to store basis function
    ArrayXd basisFunction, regressedValues	;
    // test archive
    for (int is  = 0;  is < p_nbSimul; ++is)
    {
        for (int id = 0; id < p_nDim; ++id)
        {
            x(id, is) = uniform_random();
        }
        int iDis = uniformInt_random();
        x(p_nDim, is) =  static_cast<double>(iDis);
        // function to regress depends on the discrete value on last dimension
        double prod = fList[iDis](x(0, is));
        for (int id = 1 ; id < p_nDim ; ++id)
            prod *=  fList[iDis](x(id, is));
        toRegress(is) = prod +  2 * normal_random();
    }

    // conditional expectation
    shared_ptr<Regressor> localRegressor = make_shared<Regressor>(false, x, nbMesh);

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
    {
        BinaryFileArchive ar("archiveLDL", "w");
        ar << Record(*localRegressor, "Regressor", "Top") ;
    }
    {
        BinaryFileArchive ar("archiveLDL", "r");
        Regressor reg;
        Reference< Regressor> (ar, "Regressor", "Top").restore(0, &reg);
        for (int is = 0; is < x.cols(); ++is)
            BOOST_CHECK_CLOSE(regressedValues(is), reg.getValue(x.col(is), basisFunction), accuracyEqual);

    }
}

BOOST_AUTO_TEST_CASE(testLocalDicrLastDimRegressionSerialization)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    TestLocalDiscrLastDimRegressionSerialization<LocalConstDiscrLastDimRegression>(1, 2, 10, 1);
    TestLocalDiscrLastDimRegressionSerialization<LocalLinearDiscrLastDimRegression>(1, 2, 10, 1);
}

/// test serialization for continuation values
template< class Regressor>
void TestContinuationValueDiscr(const int &p_nDim,  const int &p_nbDiscr, const int &p_nbSimul, const int &p_nbMesh)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // nb stock points
    int sizeForStock  = 4;
    // create the mesh
    ArrayXi  nbMesh = ArrayXi::Constant(p_nDim + 1, p_nbMesh);
    nbMesh(p_nDim) = p_nbDiscr;
    // generator Mersene Twister
    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    boost::random::uniform_int_distribution<> alea_uI(0, p_nbDiscr - 1);
    boost::variate_generator<boost::mt19937 &, boost::random::uniform_int_distribution<> > uniformInt_random(generator, alea_uI);
    boost::random::uniform_real_distribution<double> alea_uD(0., 1.);
    boost::variate_generator<boost::mt19937 &, boost::random::uniform_real_distribution<double> > uniform_random(generator, alea_uD);

    std::vector< std::function< double(double) > > fList(p_nbDiscr);
    for (int id = 0; id < p_nbDiscr; ++id)
        fList[id] = Func1(id);
    ArrayXXd x = ArrayXXd::Zero(p_nDim + 1, p_nbSimul);  // nb stock points
    ArrayXXd  regressedValues;
    {
        // second member to regress with one stock
        ArrayXXd toRegress(sizeForStock, p_nbSimul);

        for (int is  = 0;  is < p_nbSimul; ++is)
        {
            for (int id = 0; id < p_nDim; ++id)
            {
                x(id, is) = uniform_random();
            }
            int iDis = uniformInt_random();
            x(p_nDim, is) =  static_cast<double>(iDis);
            // function to regress depends on the discrete value on last dimension
            double prod = fList[iDis](x(0, is));
            for (int id = 1 ; id < p_nDim ; ++id)
                prod *=  fList[iDis](x(id, is));
            double uncertainty = 2 * normal_random();
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
        shared_ptr<Regressor> localRegressor = make_shared<Regressor>(false, x, nbMesh);

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
        BinaryFileArchive ar("archiveLDL1", "w");
        ar << Record(continuation, "FirstContinuation", "Top") ;

    }
    {
        // read archive
        BinaryFileArchive ar("archiveLDL1", "r");
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
    TestContinuationValueDiscr<LocalConstDiscrLastDimRegression>(1, 2, 10000, 1);
    TestContinuationValueDiscr<LocalLinearDiscrLastDimRegression>(1, 2, 10000, 1);
}


