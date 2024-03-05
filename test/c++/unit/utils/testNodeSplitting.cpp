
#define BOOST_TEST_MODULE testNodeSplitting
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "libflow/core/utils/comparisonUtils.h"
#include "libflow/core/utils/NodeParticleSplitting.h"

using namespace std;
using namespace Eigen;
using namespace libflow;

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

void testNodeSplittingDim(const int &p_nDim, const int &p_nbSimul, ArrayXi &p_nbMeshPerDim)
{
    // generate random points between -1 and 1
    unique_ptr<ArrayXXd> pointsSimul(new ArrayXXd(ArrayXXd::Random(p_nbSimul, p_nDim))) ;

    // create node splitting object
    NodeParticleSplitting nodeSplit(pointsSimul, p_nbMeshPerDim);

    // now
    ArrayXi nCell(p_nbSimul);
    int nbMesh = p_nbMeshPerDim.prod();
    Array<  array<double, 2 >, Dynamic, Dynamic >  meshCoord(p_nDim, nbMesh);
    nodeSplit.simToCell(nCell, meshCoord);

    for (int is = 0 ; is < p_nbSimul; ++is)
    {
        for (int id  = 0; id < p_nDim; ++id)
        {
            BOOST_CHECK(isLesserOrEqual((*pointsSimul)(is, id), meshCoord(id, nCell(is))[1]));
            BOOST_CHECK(isLesserOrEqual(meshCoord(id, nCell(is))[0], (*pointsSimul)(is, id)));
        }
    }
}

BOOST_AUTO_TEST_CASE(testNode1D)
{
    ArrayXi nbMeshPerDim = ArrayXi::Constant(1, 5);
    testNodeSplittingDim(1, 100, nbMeshPerDim)	;
}
BOOST_AUTO_TEST_CASE(testNode2D1)
{
    ArrayXi nbMeshPerDim = ArrayXi::Constant(2, 5);
    testNodeSplittingDim(2, 10000, nbMeshPerDim)	;
}
BOOST_AUTO_TEST_CASE(testNode2D2)
{
    ArrayXi nbMeshPerDim(2);
    nbMeshPerDim << 2, 4 ;
    testNodeSplittingDim(2, 10000, nbMeshPerDim);
}
BOOST_AUTO_TEST_CASE(testNode3D1)
{
    ArrayXi nbMeshPerDim(3);
    nbMeshPerDim << 2, 4, 5 ;
    testNodeSplittingDim(3, 100000, nbMeshPerDim)	;
}
BOOST_AUTO_TEST_CASE(testNode3D2)
{
    ArrayXi nbMeshPerDim(3);
    nbMeshPerDim << 4, 2, 5 ;
    testNodeSplittingDim(3, 100000, nbMeshPerDim)	;
}
BOOST_AUTO_TEST_CASE(testNode4D)
{
    ArrayXi nbMeshPerDim(4);
    nbMeshPerDim << 1, 4, 2, 5 ;
    testNodeSplittingDim(4, 100000, nbMeshPerDim)	;
}
