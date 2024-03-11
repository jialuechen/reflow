// Copyright (C) 2020 EDF

#define BOOST_TEST_MODULE testLaplacianGridKernelRegression
#define BOOST_TEST_DYN_LINK
#include <math.h>
#include <boost/test/unit_test.hpp>
#include <boost/timer/timer.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "libflow/regression/LaplacianGridKernelRegressionGeners.h"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/regression/ContinuationValueGeners.h"

using namespace std;
using namespace Eigen;
using namespace libflow;
using namespace gs;


/// \brief Direct calculation for kernel on  rectilinear grid
/// \param p_x  samples   (dimension, nbSim)
/// \param p_z  rectilinear points
/// \param p_h  window size per direction
/// \param p_y  array of weight values
ArrayXXd KDEDirect(const ArrayXXd &p_x, const std::vector< std::shared_ptr<ArrayXd> >    &p_z, const ArrayXd &p_h,
                   const ArrayXXd &p_y)
{
    // store nbpt per dimension before
    ArrayXi nbPtZ(p_z.size());
    nbPtZ(0) = 1;
    for (size_t id = 0; id < p_z.size() - 1; ++id)
        nbPtZ(id + 1) = nbPtZ(id) * p_z[id]->size();
    // nb point
    int nbPtZT = nbPtZ(p_z.size() - 1) * p_z[p_z.size() - 1]->size();
    // store index
    ArrayXi index(p_z.size());
    // for return
    ArrayXXd retKDE = ArrayXXd::Zero(p_y.rows(), nbPtZT);
    for (int ip  = 0; ip < nbPtZT  ; ++ip)
    {
        // index
        int ipt = ip;
        for (int id = p_z.size() - 1; id  >= 0; --id)
        {
            index(id) = static_cast<int>(ipt / nbPtZ(id));
            ipt -= index(id) * nbPtZ(id);
        }
        // nest on points
        for (int ipp = 0; ipp < p_x.cols(); ++ipp)
        {
            double ssum = 0;
            for (size_t id = 0; id < p_z.size(); ++id)
                ssum +=  fabs((*p_z[id])(index(id)) - p_x(id, ipp)) / p_h(id);
            for (int iy = 0 ; iy < p_y.rows(); ++iy)
                retKDE(iy, ip) += exp(-ssum) * p_y(iy, ipp);
        }
    }
    retKDE /= (p_x.cols() * pow(2, p_z.size()) * p_h.prod());
    return retKDE ;
}

/// calculate constant regression on  rectilinear grid
/// \param p_x  samples   (dimension, nbSim)
/// \param p_z  rectilinear points
/// \param p_h  window size per direction
/// \param p_y  array of weight values
ArrayXd constRegressionDirectOnGrid(const ArrayXXd &p_x, const std::vector< std::shared_ptr<ArrayXd> >    &p_z, const ArrayXd &p_h,
                                    const ArrayXd &p_y)
{
    ArrayXXd y = ArrayXXd::Constant(2, p_y.size(), 1.);
    y.row(1) = p_y.transpose();
    // all terms
    ArrayXXd KDE = KDEDirect(p_x, p_z, p_h, y);
    // KDE
    ArrayXd regOnZ = KDE.row(1).transpose() / KDE.row(0).transpose();
    return regOnZ;
}

/// calculate linear regression on  rectilinear grid
/// \param p_x  samples   (dimension, nbSim)
/// \param p_z  rectilinear points
/// \param p_h  window size per direction
/// \param p_y  array of weight values
ArrayXd linearRegressionDirectOnGrid(const ArrayXXd &p_x, const std::vector< std::shared_ptr<ArrayXd> >    &p_z, const ArrayXd &p_h,
                                     const ArrayXd &p_y)
{
    // dimension
    int nD = p_x.rows();
    // number of functions to calculate for regressions
    int nbFuncReg = (nD + 1) * (nD + 2) / 2;
    int nbFuncSecMem = (nD + 1)    ;

    // store all weights for kernel
    ArrayXXd y(nbFuncReg + nbFuncSecMem, p_x.cols());
    for (int is = 0; is <  p_x.cols(); ++is)
    {
        int iloc = 0;
        // lower triangular matrix
        y(iloc++, is) = 1.;
        for (int id = 0; id < nD; ++id)
        {
            y(iloc++, is) = p_x(id, is);
            for (int idd = 0; idd <= id; ++idd)
                y(iloc++, is) = p_x(id, is) * p_x(idd, is);
        }
        y(iloc++, is) = p_y(is);
        for (int id = 0; id < nD ; ++id)
            y(iloc++, is) = p_y(is) * p_x(id, is);
    }
    // all terms
    ArrayXXd KDE = KDEDirect(p_x, p_z, p_h, y);

    // for regressions
    MatrixXd  matA(1 + nD, 1 + nD);
    VectorXd  vecB(1 + nD);

    ArrayXi coord =  ArrayXi::Zero(nD);

    // number of  points on grid
    int nbPtZ = p_z[0]->size();

    for (size_t id = 1; id < p_z.size(); ++id)
        nbPtZ *= p_z[id]->size();

    // now calculate the regressed values on the grid
    ArrayXd regOnZ(nbPtZ);

    for (int ipt = 0 ; ipt < nbPtZ; ++ipt)
    {
        // create regression matrix
        int iloc = 0;
        for (int id = 0; id <= nD; ++id)
            for (int idd = 0; idd <= id; ++idd)
                matA(id, idd) = KDE(iloc++, ipt);
        for (int id = 0; id <= nD; ++id)
            for (int idd = id + 1; idd <= nD; ++idd)
                matA(id, idd) =  matA(idd, id);
        // second member and inverse
        // inverse
        LLT<MatrixXd>  lltA(matA);
        for (int id = 0; id <= nD; ++id)
            vecB(id) = KDE(iloc++, ipt);
        VectorXd coeff = lltA.solve(vecB);
        regOnZ(ipt) = coeff(0);
        for (int id  = 0; id < nD; ++id)
            regOnZ(ipt)  += coeff(id + 1) * (*p_z[id])(coord(id));
        // update coordinates
        for (int id = 0; id < nD; ++id)
        {
            if (coord(id) < p_z[id]->size() - 1)
            {
                coord(id) += 1;
                break;
            }
            else
            {
                coord(id) = 0;
            }
        }
    }
    return regOnZ;
}

/// \brief Direct calculation for kernel at sample points
/// \param p_x  samples   (dimension, nbSim)
/// \param p_z  rectilinear points
/// \param p_h  window size per direction
/// \param p_y  array of weight values
ArrayXXd KDEDirectAtSample(const ArrayXXd &p_x,  const ArrayXd &p_h, const ArrayXXd &p_y)
{
    // for return
    ArrayXXd retKDE = ArrayXXd::Zero(p_y.rows(), p_y.cols());
    for (int ip  = 0; ip < p_y.cols()  ; ++ip)
    {
        // nest on points
        for (int ipp = 0; ipp < p_x.cols(); ++ipp)
        {
            double ssum = 0;
            for (int id = 0; id < p_x.rows(); ++id)
                ssum +=  fabs(p_x(id, ip) - p_x(id, ipp)) / p_h(id);
            for (int iy = 0 ; iy < p_y.rows(); ++iy)
                retKDE(iy, ip) += exp(-ssum) * p_y(iy, ipp);
        }
    }
    retKDE /= (p_x.cols() * pow(2, p_x.rows()) * p_h.prod());
    return retKDE ;
}

/// calculate constant regression at sample points
/// \param p_x  samples   (dimension, nbSim)
/// \param p_h  window size per direction
/// \param p_y  array of weight values
ArrayXd constRegressionDirectAtSample(const ArrayXXd &p_x,  const ArrayXd &p_h,
                                      const ArrayXd &p_y)
{
    ArrayXXd y = ArrayXXd::Constant(2, p_y.size(), 1.);
    y.row(1) = p_y.transpose();
    // all terms
    ArrayXXd KDE = KDEDirectAtSample(p_x, p_h, y);
    // KDE
    ArrayXd regOnZ = KDE.row(1).transpose() / KDE.row(0).transpose();
    return regOnZ;
}

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

// test in dimension 1 at sample points : to check interpolation
void testDimension1DOnGridForInterp(const int &p_nbSimul,  const double &p_h, const double &p_epsilon)
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

    ArrayXd hB = ArrayXd::Constant(1, p_h);

    // test regression object
    double q = 50; // coeff for the number of grid points used
    LaplacianGridKernelRegression kernelReg(false, x, hB, q, false);

    // regress on grid just for test
    ArrayXd regressed = kernelReg.getAllSimulations(y);

    // now regress at z points
    ArrayXd naivReg  =  constRegressionDirectAtSample(x, hB, y);


    BOOST_CHECK_SMALL((regressed - naivReg).abs().maxCoeff(), p_epsilon);

}


// test in dimension 1
void testDimension1DOnGrid(const bool &p_bLin, const int &p_nbSimul,  const double &p_h, const double &p_epsilon)
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

    ArrayXd hB = ArrayXd::Constant(1, p_h);

    // test regression object
    double q = 2; // coeff for the number of grid points used
    LaplacianGridKernelRegression kernelReg(false, x, hB, q, p_bLin);

    // get back grid
    vector< shared_ptr< ArrayXd> >  z =  kernelReg.getZ() ;

    // regress on grid just for test
    ArrayXd regressed = kernelReg.getAllSimulations(y);

    // now regress at z points
    ArrayXd naivReg  = ((p_bLin) ? linearRegressionDirectOnGrid(x, z, hB, y) : constRegressionDirectOnGrid(x, z, hB, y));

    // get basis
    ArrayXd baseCoeff = kernelReg.getCoordBasisFunction(y);

    // coordinate Z
    ArrayXd coordZ(1);

    for (int i = 0; i < z[0]->size(); ++i)
    {
        coordZ(0) = (*z[0])(i);
        double fastReg = kernelReg.getValue(coordZ, baseCoeff);
        BOOST_CHECK_SMALL(fabs(fastReg - naivReg(i)), p_epsilon);
    }
}

// test in dimension 2
void testDimension2DOnGrid(const bool &p_bLin, const int &p_nbSimul,  const double &p_h, const double &p_epsilon)
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

    // bandwidth
    ArrayXd hB = ArrayXd::Constant(2, p_h);

    // test regression object
    double q = 1; // coeff for the number of grid points used
    LaplacianGridKernelRegression kernelReg(false, x, hB, q, p_bLin);

    // get back grid
    vector< shared_ptr< ArrayXd> >  z =  kernelReg.getZ() ;

    // regress on grid just for test
    ArrayXd regressed = kernelReg.getAllSimulations(y);

    // now regress at z points
    ArrayXd naivReg  = ((p_bLin) ? linearRegressionDirectOnGrid(x, z, hB, y) : constRegressionDirectOnGrid(x, z, hB, y));

    // get basis
    ArrayXd baseCoeff = kernelReg.getCoordBasisFunction(y);

    // coordinate Z
    ArrayXd coordZ(2);

    for (int j = 0; j < z[1]->size(); ++j)
        for (int i = 0; i < z[0]->size(); ++i)
        {
            coordZ(0) = (*z[0])(i);
            coordZ(1) = (*z[1])(j);
            double fastReg = kernelReg.getValue(coordZ, baseCoeff);
            BOOST_CHECK_SMALL(fabs(fastReg - naivReg(i + j * z[0]->size())), p_epsilon);
        }
}


// test in dimension 3
void testDimension3DOnGrid(const bool &p_bLin, const int &p_nbSimul,  const double &p_h, const double &p_epsilon)
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

    // bandwidth
    ArrayXd hB = ArrayXd::Constant(3, p_h);

    // test regression object
    double q = 1; // coeff for the number of grid points used
    LaplacianGridKernelRegression kernelReg(false, x, hB, q, p_bLin);

    // get back grid
    vector< shared_ptr< ArrayXd> >  z =  kernelReg.getZ() ;

    // regress on grid just for test
    ArrayXd regressed = kernelReg.getAllSimulations(y);

    // now regress at z points
    ArrayXd naivReg  = ((p_bLin) ? linearRegressionDirectOnGrid(x, z, hB, y) : constRegressionDirectOnGrid(x, z, hB, y));

    // get basis
    ArrayXd baseCoeff = kernelReg.getCoordBasisFunction(y);

    // coordinate Z
    ArrayXd coordZ(3);

    for (int k = 0; k < z[2]->size(); ++k)
        for (int j = 0; j < z[1]->size(); ++j)
            for (int i = 0; i < z[0]->size(); ++i)
            {
                coordZ(0) = (*z[0])(i);
                coordZ(1) = (*z[1])(j);
                coordZ(2) = (*z[2])(k);
                double fastReg = kernelReg.getValue(coordZ, baseCoeff);
                BOOST_CHECK_SMALL(fabs(fastReg - naivReg(i + j * z[0]->size() + k * z[0]->size()*z[1]->size())), p_epsilon);
            }
}




// test fonctionality in 1D
void testDimensionFonctionality1DOnGrid(const bool &p_bLin,  const int &p_nbSimul,  const double &p_h, const double &p_epsilon)
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

    // bandwidth
    ArrayXd hB = ArrayXd::Constant(1, p_h);

    // test regression object
    double q = 100; // coeff for the number of grid points used
    LaplacianGridKernelRegression kernelReg(false, x, hB, q, p_bLin);


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
        BinaryFileArchive ar("archiveLGKR", "w");
        ar << Record(kernelReg, "Regressor", "Top") ;
        ar.flush();
    }
    {
        BinaryFileArchive ar("archiveLGKR", "r");
        LaplacianGridKernelRegression regAr;
        Reference<LaplacianGridKernelRegression> (ar, "Regressor", "Top").restore(0, &regAr);
        for (int is = 0; is < p_nbSimul / 10; ++is)
        {
            double regressAPoint = regAr.getValue(x.col(is), regressedGrid);
            BOOST_CHECK_SMALL(fabs(regressAPoint - regressed(is)), p_epsilon);
        }
    }
}



// test fonctionality in 2D
void testDimensionFonctionality2DOnGrid(const bool &p_bLin,  const int &p_nbSimul,  const double &p_h, const double &p_epsilon)
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

    // bandwidth
    ArrayXd hB = ArrayXd::Constant(2, p_h);

    // test regression object
    double q = 1; // coeff for the number of grid points used
    LaplacianGridKernelRegression kernelReg(false, x, hB, q, p_bLin);


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
        BinaryFileArchive ar("archiveLGKR1", "w");
        ar << Record(kernelReg, "Regressor", "Top") ;
        ar.flush();
    }
    {
        BinaryFileArchive ar("archiveLGKR1", "r");
        LaplacianGridKernelRegression regAr;
        Reference<LaplacianGridKernelRegression> (ar, "Regressor", "Top").restore(0, &regAr);
        for (int is = 0; is < p_nbSimul / 10; ++is)
        {
            double regressAPoint = regAr.getValue(x.col(is), regressedGrid);
            BOOST_CHECK_SMALL(fabs(regressAPoint - regressed(is)), p_epsilon);
        }
    }
}



// test serialization for continuation values
void TestContinuationValue(const bool &p_bLin, const int &p_nDim,  const int &p_nbSimul,  const double &p_h,  const double &p_accuracyEqual, const double &p_accuracyInterp)
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
        ArrayXd lowValues(1), step(1);
        lowValues(0) = 0. ;
        step(0) = 1;
        ArrayXi  nbStep(1);
        nbStep(0) = sizeForStock - 1;
        // grid
        shared_ptr< RegularSpaceGrid > regular = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);

        ArrayXd hB = ArrayXd::Constant(p_nDim, p_h);

        // conditional expectation
        double q = 2; // coeff for the number of grid points used
        shared_ptr<LaplacianGridKernelRegression> localRegressor = make_shared<LaplacianGridKernelRegression>(false, x, hB, q, p_bLin);

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
        BinaryFileArchive ar("archiveLGKR2", "w");
        ar << Record(continuation, "FirstContinuation", "Top") ;

    }
    {
        // read archive
        BinaryFileArchive ar("archiveLGKR2", "r");
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


BOOST_AUTO_TEST_CASE(testLaplacianGridKernel1D1AtSample)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif


    int nbSimul = 100;
    double epsilon1 = 1e-3;
    double h = 0.4;
    testDimension1DOnGridForInterp(nbSimul, h, epsilon1);
}


BOOST_AUTO_TEST_CASE(testLaplacianGridKernel1D1)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif


    int nbSimul = 10;
    double epsilon1 = 1e-6;
    bool bLin = true;
    double h = 0.2;
    testDimension1DOnGrid(bLin, nbSimul, h, epsilon1);
    bLin = false;
    testDimension1DOnGrid(bLin, nbSimul, h, epsilon1);

}

BOOST_AUTO_TEST_CASE(testLaplacianGridKernel2D1)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSimul = 50;
    double epsilon1 = 1e-6;
    bool bLin = true;
    double h = 0.2;
    testDimension2DOnGrid(bLin, nbSimul, h, epsilon1);
    bLin = false;
    testDimension2DOnGrid(bLin, nbSimul, h, epsilon1);

}


BOOST_AUTO_TEST_CASE(testLaplacianGridKernel3D1)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSimul = 100;
    double epsilon1 = 1e-6;
    bool bLin = true;
    double h = 0.2;
    testDimension3DOnGrid(bLin, nbSimul, h, epsilon1);
    bLin = false;
    testDimension3DOnGrid(bLin, nbSimul, h, epsilon1);

}



BOOST_AUTO_TEST_CASE(testGridKernelFunc1DOnGrid)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSimul = 100;
    double epsilon1 = 1e-6;
    bool bLin = true;
    double h = 0.2;
    testDimensionFonctionality1DOnGrid(bLin, nbSimul, h, epsilon1);
    bLin = false;
    testDimensionFonctionality1DOnGrid(bLin, nbSimul, h, epsilon1);

}

BOOST_AUTO_TEST_CASE(testGridKernelFunc2DOnGrid)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSimul = 100;
    double epsilon1 = 1e-6;
    bool bLin = true;
    double h = 0.2;
    testDimensionFonctionality2DOnGrid(bLin, nbSimul, h, epsilon1);
    bLin = false;
    testDimensionFonctionality2DOnGrid(bLin, nbSimul, h, epsilon1);

}


BOOST_AUTO_TEST_CASE(testContinuationKernel1D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    bool bLin = true;
    TestContinuationValue(bLin, 1, 200, 0.2, 1e-9, 0.0001);
    bLin = false;
    TestContinuationValue(bLin, 1, 200, 0.2, 1e-9, 0.0001);

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
    TestContinuationValue(bLin, 3, 200, 0.1, 1e-9, 0.0001);
    bLin = false;
    TestContinuationValue(bLin, 3, 200, 0.1, 1e-9, 0.0001);
}

BOOST_AUTO_TEST_CASE(testContinuationKernel4D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    bool bLin = true;
    TestContinuationValue(bLin, 4, 1000, 0.1, 1e-9, 0.0001);
    bLin = false;
    TestContinuationValue(bLin, 4, 1000, 0.1, 1e-9, 0.0001);
}

