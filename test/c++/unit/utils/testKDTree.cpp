
#define BOOST_TEST_MODULE testKDTree
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/random.hpp>
#include <Eigen/Dense>
#include "reflow/core/utils/KDTree.h"

using namespace std;
using namespace Eigen;
using namespace reflow;

#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif

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


    // new particle
    ArrayXd pt(p_nD);
    for (int id = 0 ; id <  p_nD; ++id)
    {
        pt(id) = normalRand();
    }


    int ipMin = 100000;
    double dist = 100000000000;
    for (int is = 0; is < p_nbSim; ++is)
    {
        double ddis = 0;
        for (int id = 0; id < p_nD; ++id)
        {
            ddis += pow(pt(id) - x(id, is), 2.);
        }
        if (ddis < dist)
        {
            dist = ddis;
            ipMin = is;
        }
    }
    cout << " DIST " << sqrt(dist) << " ip" << ipMin << endl ;

    KDTree tree(x);
    int idx = tree.nearestIndex(pt);

    BOOST_CHECK_EQUAL(ipMin, idx);

}



BOOST_AUTO_TEST_CASE(testKDTree1D)
{
    int ndim = 1;
    int nbSim = 1100;
    testnD(ndim, nbSim);

}
BOOST_AUTO_TEST_CASE(testKDTree2D)
{
    int ndim = 2;
    int nbSim = 2100;
    testnD(ndim, nbSim);

}
BOOST_AUTO_TEST_CASE(testKDTree3D)
{
    int ndim = 3;
    int nbSim = 1100;
    testnD(ndim, nbSim);

}
BOOST_AUTO_TEST_CASE(testKDTree4D)
{
    int ndim = 4;
    int nbSim = 1100;
    testnD(ndim, nbSim);

}
BOOST_AUTO_TEST_CASE(testKDTree5D)
{
    int ndim = 5;
    int nbSim = 1100;
    testnD(ndim, nbSim);

}
