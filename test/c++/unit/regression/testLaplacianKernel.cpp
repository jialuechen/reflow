
#define BOOST_TEST_MODULE testLaplacianKernel
#define BOOST_TEST_DYN_LINK
#include <memory>
#include <functional>
#include <boost/random.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/chrono.hpp>
#include <boost/timer/timer.hpp>
#include "reflow/regression/nDDominanceKernel.h"



/// test Laplacian kernel in  dimension 1 to 5

using namespace Eigen;
using namespace std;
using namespace reflow;
using boost::timer::cpu_timer;
using boost::timer::auto_cpu_timer;


#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif


double  accuracy_zero = 1e-7 ; // accuracy
// kernel estimation
double KVal(const ArrayXd &x1, const ArrayXd &x2,  const ArrayXd   &h)
{
    double sum = 0;
    for (int i = 0; i < x1.size(); ++i)
        sum += fabs(x1(i) - x2(i)) / h(i);
    return exp(- sum);
}


/// test Laplacian kernel  nD
/// p_nD    dimension of the problem
/// p_nbSim number of simulations
void testnD(const int &p_nD, const  int &p_nbSim)
{
    boost::mt19937 generator;
    boost::normal_distribution<double> normalDistrib;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normalRand(generator, normalDistrib);

    // size of kernel
    ArrayXd  h =  ArrayXd::Constant(p_nD, 0.2);

    // particles
    ArrayXXd x(p_nD, p_nbSim);
    for (int i = 0; i < p_nbSim; ++i)
        for (int id = 0 ; id <  p_nD; ++id)
        {
            x(id, i) = normalRand();
        }

    cpu_timer timerNaiveGrid;
    timerNaiveGrid.start();

    ArrayXd  kernVal(p_nbSim);
    for (int i = 0 ; i < p_nbSim ; ++i)
    {
        kernVal(i) = 0;
        for (int j = 0; j < p_nbSim; ++j)
        {
            kernVal(i) += KVal(x.col(i), x.col(j), h);
        }
    }

    kernVal /= pow(2, p_nD) * h.prod() * p_nbSim;

    boost::chrono::duration<double> secondsNaive = boost::chrono::nanoseconds(timerNaiveGrid.elapsed().user);
    double timeNaive  = secondsNaive.count() ;

    cpu_timer  timeSlide;
    timeSlide.start();


    // creation of the 2^d terms
    int nbSum = pow(2, p_nD);
    vector< shared_ptr<ArrayXXd> > vecToAdd(nbSum);
    // calculate exp values
    Eigen::ArrayXi iCoord(p_nD) ;
    for (int i = 0; i < nbSum; ++i)
    {
        int ires = i;
        for (int id = p_nD - 1 ; id >= 0  ; --id)
        {
            unsigned int idec = (ires >> id) ;
            iCoord(id) = -(2 * idec - 1);
            ires -= (idec << id);
        }
        vecToAdd[i] = make_shared<ArrayXXd>(1, p_nbSim);
        for (int is = 0; is < p_nbSim; ++is)
        {
            double ssum = 0;
            for (int id = 0; id < p_nD; ++id)
                ssum += iCoord(id) * x(id, is) / h(id);
            (*vecToAdd[i])(0, is) = exp(ssum);
        }
    }

    vector< shared_ptr<ArrayXXd> > fDomin(nbSum);


    // kernel resolution
    nDDominanceKernel(x, vecToAdd, fDomin);


    // reconstruction of the 2^d terms
    ArrayXd recons(p_nbSim);
    for (int is = 0; is < p_nbSim; ++is)
    {
        recons(is) = 1;
        for (int id = 0; id < nbSum; ++id)
            recons(is) += (*fDomin[id])(0, is) * (*vecToAdd[nbSum - 1 - id])(0, is);
    }
    recons /= pow(2, p_nD) * p_nbSim * h.prod();
    timeSlide.stop();
    boost::chrono::duration<double> secondsSlide = boost::chrono::nanoseconds(timeSlide.elapsed().user);
    double timeSlideR  = secondsSlide.count() ;

    double error = (recons - kernVal).abs().maxCoeff();

    BOOST_CHECK(error < accuracy_zero);

    cout << " Nb sim " << p_nbSim << "  timeNaive " << timeNaive << " Slide " << timeSlideR << " Error " << error << endl ;

}



BOOST_AUTO_TEST_CASE(testLaplacianKernel1D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSim = 457;
    int ndim = 1;
    testnD(ndim, nbSim);
}

BOOST_AUTO_TEST_CASE(testLaplacianKernel2D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSim = 457;
    int ndim = 2;
    testnD(ndim, nbSim);
}


BOOST_AUTO_TEST_CASE(testLaplacianKernel3D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSim = 457;
    int ndim = 3;
    testnD(ndim, nbSim);
}


BOOST_AUTO_TEST_CASE(testLaplacianKernel4D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSim = 457;
    int ndim = 4;
    testnD(ndim, nbSim);
}


BOOST_AUTO_TEST_CASE(testLaplacianKernel5D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSim = 457;
    int ndim = 5;
    testnD(ndim, nbSim);
}

BOOST_AUTO_TEST_CASE(testLaplacianKernel6D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSim = 2999;
    int ndim = 6;
    testnD(ndim, nbSim);
}
