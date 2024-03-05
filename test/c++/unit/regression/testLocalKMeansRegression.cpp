// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#define BOOST_TEST_MODULE testLocalLinearRegression
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
#include "libflow/regression/LocalKMeansRegressionGeners.h"
#include "libflow/regression/ContinuationValueGeners.h"
#include "libflow/regression/ContinuationCutsGeners.h"

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
double accuracyNearEqual = 0.05;

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
void testDimension(const bool &p_bRotate, const int &p_nDim, const int &p_nbSimul, const int &p_nMesh, const double &p_accuracyRegression)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // create the mesh
    ArrayXi  nbMesh = ArrayXi::Constant(p_nDim, p_nMesh);
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
        LocalKMeansRegression localRegressor(nbMesh, p_bRotate);
        localRegressor.updateSimulations(bZeroDate, x);
        ArrayXd  regressedValues = localRegressor.getAllSimulations(toRegress);

        double average = toRegress.mean();
        for (int is = 0; is < p_nbSimul; ++is)
            if ((fabs(regressedValues(is)) > tiny) && (fabs(average) > tiny))
                BOOST_CHECK(fabs(regressedValues(is) - average) <  accuracyEqual * max(fabs(average), fabs(regressedValues(is))));
    }

    {
        // constructor not zero
        bool bZeroDate = false;

        // constructor
        LocalKMeansRegression localRegressor(nbMesh, p_bRotate);
        localRegressor.updateSimulations(bZeroDate, x);
        // first regression
        ArrayXd  regressedValues = localRegressor.getAllSimulations(toRegress);
        // then just calculate function basis coefficient
        ArrayXd regressedFuntionCoeff = localRegressor.getCoordBasisFunction(toRegress);
        for (int is = 0; is < p_nbSimul; ++is)
        {
            Map<ArrayXd> xloc(x.col(is).data(), p_nDim);
            if (fabs(regressedValues(is)) > tiny)
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
            if (fabs(regressedValues(is)) > tiny)
                BOOST_CHECK_CLOSE(regressedValues(is), regressedAllValues(is), accuracyNearEqual);
        }
    }
}


// /// 1D check single second member regression
BOOST_AUTO_TEST_CASE(testLocalKMeansRegression1D)
{

#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 1 ;
    // Number of simulations and parameters for regression.
    int nMesh = 20 ;
    ArrayXi  nbMesh = ArrayXi::Constant(nDim, nMesh);
    int nbSimul = 500000;
    // accuracy
    double accuracy = 7; // percentage error
    testDimension(false, nDim, nbSimul, nMesh, accuracy);
}

// 2D check single second member
BOOST_AUTO_TEST_CASE(testLocalKMeansRegression2D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 2 ;
    // Number of simulations and parameters for regression.
    int nMesh = 20 ;
    ArrayXi  nbMesh = ArrayXi::Constant(nDim, nMesh);
    int nbSimul = 5000000;
    // accuracy
    double accuracy = 15; // percentage error
    testDimension(false, nDim, nbSimul, nMesh, accuracy);
#if !defined(mips) && !defined(__mips__) && !defined(__mips)
    testDimension(true, nDim, nbSimul, nMesh, accuracy);
#endif
}

// // 3D check single second member
BOOST_AUTO_TEST_CASE(testLocalKMeansRegression3D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 3 ;
    // Number of simulations and parameters for regression.
    int nMesh = 20 ;
    ArrayXi  nbMesh = ArrayXi::Constant(nDim, nMesh);
    int nbSimul = 2000000;
    // accuracy
    double accuracy = 20; // percentage error
    testDimension(false, nDim, nbSimul, nMesh, accuracy);
}

// 1D check multiple second members
BOOST_AUTO_TEST_CASE(testLocalKMeansRegression1DMultipleSecondMember)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // dimension
    int nDim = 1 ;
    // number of second member
    int nMember = 5 ;
    // Number of simulations and parameters for regression.
    int nMesh = 10 ;
    ArrayXi  nbMesh = ArrayXi::Constant(nDim, nMesh);
    int nbSimul = 20000;
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
        LocalKMeansRegression localRegressor(nbMesh);
        localRegressor.updateSimulations(bZeroDate, x);
        // regression
        ArrayXXd  regressedValues = localRegressor.getAllSimulationsMultiple(toRegress);

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
        LocalKMeansRegression localRegressor(bZeroDate, x, nbMesh);
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
                if (fabs(regressedValues(imem, is)) > tiny)
                    BOOST_CHECK_CLOSE(regressedValues(imem, is), regressedValues1D(is), accuracyEqual);
            ArrayXd regressedFuntionCoeff1D = localRegressor.getCoordBasisFunction(toRegress1D);
            for (int im = 0; im < regressedFuntionCoeff1D.size(); ++im)
                if (fabs(regressedFuntionCoeff1D(im)) > tiny)
                    BOOST_CHECK_CLOSE(regressedFuntionCoeff(imem, im), regressedFuntionCoeff1D(im), accuracyEqual);
        }
    }
}




// test serialization LocalKMeansRegression
void  TestLocalRegressionSerialization(const int &p_nDim, const int &p_nbSimul, const int &p_nbMesh)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    ArrayXi  nbMesh = ArrayXi::Constant(p_nDim, p_nbMesh);
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
        shared_ptr<LocalKMeansRegression> localRegressor = make_shared<LocalKMeansRegression>(false, x, nbMesh);

        // regress  directly with regressor
        regressedValues = localRegressor->getAllSimulations(toRegress);

        // basis functions
        basisFunction =  localRegressor->getCoordBasisFunction(toRegress);

        // test
        for (int is  = 0; is < p_nbSimul; ++is)
        {
            double regVal = localRegressor->reconstructionASim(is, basisFunction);
            if (fabs(regressedValues(is)) > tiny)
                BOOST_CHECK_CLOSE(regressedValues(is), regVal, accuracyEqual);
        }

        // archive
        BinaryFileArchive ar("archiveLKM", "w");
        ar << Record(*localRegressor, "Regressor", "Top") ;
    }
    {
        BinaryFileArchive ar("archiveLKM", "r");
        LocalKMeansRegression reg;
        Reference< LocalKMeansRegression> (ar, "Regressor", "Top").restore(0, &reg);
        for (int is = 0; is < x.size(); ++is)
            if (fabs(regressedValues(is)) > tiny)
                BOOST_CHECK_CLOSE(regressedValues(is), reg.getValue(x.col(is), basisFunction), accuracyEqual);

    }
}

BOOST_AUTO_TEST_CASE(testLocalKMeansRegressionSerialization)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    TestLocalRegressionSerialization(1, 10, 1);
}

/// test serialization for continuation values
void TestContinuationValue(const int &p_nDim, const int &p_nbSimul, const int &p_nbMesh)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // nb stock points
    int sizeForStock  = 4;
    ArrayXi  nbMesh = ArrayXi::Constant(p_nDim, p_nbMesh);
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
        shared_ptr<LocalKMeansRegression> localRegressor = make_shared<LocalKMeansRegression>(false, x, nbMesh);

        // regress  directly with regressor
        regressedValues = localRegressor->getAllSimulationsMultiple(toRegress);
        // creation continuation value object
        ContinuationValue  continuation(regular, localRegressor,  toRegress.transpose());

        // regress with continuation value object
        ArrayXd ptStock(1) ;
        ptStock(0) = sizeForStock / 2;
        ArrayXd regressedByContinuation = continuation.getAllSimulations(ptStock);
        for (int is  = 0;  is < p_nbSimul; ++is)
            if (fabs(regressedValues(sizeForStock / 2, is)) > tiny)
                BOOST_CHECK_CLOSE(regressedByContinuation(is), regressedValues(sizeForStock / 2, is), accuracyEqual);

        // default non compression
        BinaryFileArchive ar("archiveLKM1", "w");
        ar << Record(continuation, "FirstContinuation", "Top") ;

    }
    {
        // read archive
        BinaryFileArchive ar("archiveLKM1", "r");
        ContinuationValue contRead;
        Reference< ContinuationValue >(ar, "FirstContinuation", "Top").restore(0, &contRead);
        ArrayXd ptStock(1) ;
        ptStock(0) = sizeForStock / 2;
        for (int is  = 0;  is < p_nbSimul; ++is)
            if (fabs(regressedValues(sizeForStock / 2, is)) > tiny)
                BOOST_CHECK_CLOSE(contRead.getValue(ptStock, x.col(is)), regressedValues(sizeForStock / 2, is), accuracyEqual);
    }
}



BOOST_AUTO_TEST_CASE(testKMeansContinuation)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    TestContinuationValue(1, 40, 1);
}

// test serialization for continuation cuts values
void TestContinuationCuts(const int &p_nDim, const int &p_nbSimul, const int &p_nbMesh)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // nb stock points
    int sizeForStock  = 4;
    ArrayXi  nbMesh = ArrayXi::Constant(p_nDim, p_nbMesh);
    // generator Mersene Twister
    boost::mt19937 generator;
    boost::normal_distribution<double> alea_n;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n) ;
    std::function< double(double) > f = Func1();
    ArrayXXd x = ArrayXXd::Random(p_nDim, p_nbSimul);     // test archive
    ArrayXXd  regressedValues;
    ArrayXXd  hyperStock(1, 2) ;

    // grid for storage
    Eigen::ArrayXd lowValues(1), step(1);
    lowValues(0) = 0. ;
    step(0) = 1; // step for storage equal to 1
    Eigen::ArrayXi  nbStep(1);
    nbStep(0) = sizeForStock - 1;
    // grid
    shared_ptr< RegularSpaceGrid > regular = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);


    {
        // second member to regress with one stock  (one storage, so one sensibility)
        ArrayXXd toRegress(sizeForStock, p_nbSimul);
        ArrayXXd toRegressCut(sizeForStock, p_nbSimul * 2);
        for (int is  = 0;  is < p_nbSimul; ++is)
        {
            double prod = f(x(0, is));
            for (int id = 1 ; id < p_nDim ; ++id)
                prod *=  f(x(id, is));
            double uncertainty = 4 * normal_random();
            for (int j = 0; j < sizeForStock; ++j)
            {
                toRegress(j, is) = prod * (j + 1) +  uncertainty;
                toRegressCut(j, is) = prod * (j + 1) +  uncertainty;
                toRegressCut(j, is + p_nbSimul) = prod ; // sensibility equal to prod
            }
        }
        // conditional expectation
        shared_ptr<LocalKMeansRegression> localRegressor = make_shared<LocalKMeansRegression>(false, x, nbMesh);

        // regress  directly with regressor
        regressedValues = localRegressor->getAllSimulationsMultiple(toRegress);

        // creation continuation value object
        ContinuationCuts  contCut(regular, localRegressor,  toRegressCut.transpose());

        // calculate all cuts
        hyperStock(0, 0) = lowValues(0);
        hyperStock(0, 1) = lowValues(0) + nbStep(0) * step(0);

        ArrayXXd regressedCuts = contCut.getCutsAllSimulations(hyperStock);

        // test only values
        shared_ptr<GridIterator> iterGrid = regular->getGridIterator();
        while (iterGrid->isValid())
        {
            for (int is  = 0;  is < p_nbSimul; ++is)
            {
                // coordinates
                ArrayXd pointCoord = iterGrid->getCoordinate();
                // reconstruct value from cuts coefficients
                double valReconst = regressedCuts(is, iterGrid->getCount()) + regressedCuts(is + p_nbSimul, iterGrid->getCount()) * pointCoord(0);
                if (fabs(valReconst) > tiny)
                    BOOST_CHECK_CLOSE(valReconst, regressedValues(iterGrid->getCount(), is), accuracyEqual);
            }
            iterGrid->next();
        }

        // default non compression
        BinaryFileArchive ar("archiveLKM2", "w");
        ar << Record(contCut, "FirstContinuation", "Top") ;

    }
    {
        // read archive
        BinaryFileArchive ar("archiveLKM2", "r");
        ContinuationCuts contCutRead;
        Reference< ContinuationCuts >(ar, "FirstContinuation", "Top").restore(0, &contCutRead);

        // test only values
        shared_ptr<GridIterator> iterGrid = regular->getGridIterator();
        while (iterGrid->isValid())
        {
            for (int is  = 0;  is < p_nbSimul; ++is)
            {
                // coordinates
                ArrayXd pointCoord = iterGrid->getCoordinate();

                // cut from a sim
                ArrayXXd regCuts = contCutRead.getCutsASim(hyperStock, x.col(is));

                // reconstruct value from cuts coefficients
                double valReconst = regCuts(0, iterGrid->getCount()) + regCuts(1, iterGrid->getCount()) * pointCoord(0);
                if (fabs(valReconst) > tiny)
                    BOOST_CHECK_CLOSE(valReconst, regressedValues(iterGrid->getCount(), is), accuracyEqual);
            }
            iterGrid->next();
        }
    }
}



BOOST_AUTO_TEST_CASE(testContKMeansCut)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    TestContinuationCuts(1, 10, 1);
}


BOOST_AUTO_TEST_CASE(testKMeansCloneLocal)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    // create the mesh
    ArrayXi  nbMesh = ArrayXi::Constant(2, 2);
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
        LocalKMeansRegression localRegressor(nbMesh);
        localRegressor.updateSimulations(bZeroDate, x);
        ArrayXd  regressedValues = localRegressor.getAllSimulations(toRegress);

        // now clone
        shared_ptr< BaseRegression> cloneRegressor = localRegressor.clone();
        ArrayXd  regressedValuesClone = cloneRegressor->getAllSimulations(toRegress);

        for (int is = 0; is < 100; ++is)
            if (fabs(regressedValues(is)) > tiny)
                BOOST_CHECK_CLOSE(regressedValues(is), regressedValuesClone(is), accuracyEqual);
    }

    {
        // constructor not zero
        bool bZeroDate = 0;

        // constructor
        LocalKMeansRegression localRegressor(nbMesh);
        localRegressor.updateSimulations(bZeroDate, x);
        // first regression
        ArrayXd  regressedValues = localRegressor.getAllSimulations(toRegress);
        // now clone
        shared_ptr< BaseRegression> cloneRegressor = localRegressor.clone();
        ArrayXd  regressedValuesClone = cloneRegressor->getAllSimulations(toRegress);

        for (int is = 0; is < 100; ++is)
            if (fabs(regressedValues(is)) > tiny)
                BOOST_CHECK_CLOSE(regressedValues(is), regressedValuesClone(is), accuracyEqual);


    }
}
