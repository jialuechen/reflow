
#define BOOST_TEST_MODULE testDominanceKernel
#define BOOST_TEST_DYN_LINK
#include <memory>
#include <functional>
#include <boost/random.hpp>
#include <boost/test/unit_test.hpp>
#include "reflow/regression/nDDominanceKernel.h"




using namespace Eigen;
using namespace std;
using namespace reflow;


#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif

/// test dominance nD
/// p_nD    dimension of the problem
/// p_nbSim number of simulations
void testnD(const int &p_nD, const  int &p_nbSim)
{
    boost::mt19937 generator;
    boost::normal_distribution<double> normalDistrib;
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<double> > normalRand(generator, normalDistrib);

    // particles
    ArrayXXd x(p_nD, p_nbSim);
    for (int i = 0; i < p_nbSim; ++i)
        for (int id = 0 ; id <  p_nD; ++id)
        {
            x(id, i) = normalRand();
        }

    // number of partial sums
    int nbSum = pow(2, p_nD);
    ArrayXXd  sumExp = ArrayXXd::Zero(p_nbSim, nbSum);

    Eigen::ArrayXi iCoord(p_nD) ;
    for (int i = 0; i < nbSum; ++i)
    {
        int ires = i;
        for (int id = p_nD - 1 ; id >= 0  ; --id)
        {
            unsigned int idec = (ires >> id) ;
            iCoord(id) = 2 * idec - 1;
            ires -= (idec << id);
        }
        for (int is = 0; is < p_nbSim; ++is)
        {
            for (int js = 0 ; js < p_nbSim; ++js)
            {
                bool bAdd = true ;
                for (int id = 0; id < p_nD  ; ++id)
                {
                    if (x(id, js)*iCoord(id) <= x(id, is)*iCoord(id))
                    {
                        bAdd = false;
                        break;
                    }
                }
                if (bAdd)
                {
                    sumExp(is, i) += 1;
                }
            }
        }
    }
    vector< shared_ptr<ArrayXXd> > vecToAdd(nbSum);
    // calculate exp values
    for (int id = 0; id < nbSum; ++id)
        vecToAdd[id] = make_shared<ArrayXXd>(ArrayXXd::Constant(1, p_nbSim, 1));

    vector< shared_ptr<ArrayXXd> > fDomin(nbSum);

    nDDominanceKernel(x, vecToAdd, fDomin);

    for (int id  = 0 ; id < nbSum; ++id)
    {
        BOOST_CHECK_EQUAL(((*fDomin[id]).row(0).transpose() - sumExp.col(id)).abs().maxCoeff(), 0);
    }
}


BOOST_AUTO_TEST_CASE(testDominance1D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSim = 457;
    int ndim = 1;
    testnD(ndim, nbSim);
}

BOOST_AUTO_TEST_CASE(testDominance2D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSim = 457;
    int ndim = 2;
    testnD(ndim, nbSim);
}


BOOST_AUTO_TEST_CASE(testDominance3D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSim = 457;
    int ndim = 3;
    testnD(ndim, nbSim);
}


BOOST_AUTO_TEST_CASE(testDominance4D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSim = 457;
    int ndim = 4;
    testnD(ndim, nbSim);
}


BOOST_AUTO_TEST_CASE(testDominance5D)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif

    int nbSim = 457;
    int ndim = 5;
    testnD(ndim, nbSim);
}
