
#define BOOST_TEST_MODULE testGrid
#define BOOST_TEST_DYN_LINK
#include <fstream>
#include <memory>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include <cmath>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "reflow/core/grids/RegularLegendreGrid.h"
#include "reflow/core/grids/RegularLegendreGridGeners.h"

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


BOOST_AUTO_TEST_CASE(testGridLegendreGeneration)
{

#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    int nDim = 4 ;
    ArrayXd lowValues(nDim);
    for (int i = 0; i < nDim; ++i)
        lowValues(i) = static_cast<double>(i);
    ArrayXd step(nDim);
    for (int i = 0; i < nDim; ++i)
        step(i) = 0.6 * (i + 1);
    ArrayXi  nbStep(nDim);
    for (int i = 0; i < nDim; ++i)
        nbStep(i) = 3 * nDim * (i + 1);
    ArrayXi poly = ArrayXi::Constant(nDim, 5);
    // regular
    RegularLegendreGrid regGrid(lowValues, step, nbStep, poly);

    vector<shared_ptr<ArrayXd> > meshPerDimension(nDim);
    for (int i = 0; i < nDim; ++i)
    {
        meshPerDimension[i] = make_shared< ArrayXd >((nbStep(i) + 1));
        (*meshPerDimension[i]) = ArrayXd::LinSpaced(nbStep(i) + 1, lowValues(i), lowValues(i) + nbStep(i) * step(i));
    }

    int nbSimul = 10000;
    // generate random points between 0 and 1
    ArrayXXd pointsSimul = 0.5 * (ArrayXXd::Constant(nDim, nbSimul, 1) + ArrayXXd::Random(nDim, nbSimul));
    for (int i = 0 ; i < nDim; ++i)
        pointsSimul.row(i) = ArrayXd::Constant(pointsSimul.cols(), lowValues(i)).transpose() +  nbStep(i) * step(i) * pointsSimul.row(i);

    for (int is  = 0; is < nbSimul ; ++is)
    {
        ArrayXi iposReg = regGrid.lowerPositionCoord(pointsSimul.col(is));
        ArrayXd meshReg = regGrid.getMeshSize(iposReg);
    }
    for (int is  = 0; is < nbSimul ; ++is)
    {
        ArrayXi iposReg = regGrid.upperPositionCoord(pointsSimul.col(is));
    }

    // test sub grids
    Array< array<int, 2>, Dynamic, 1>   mesh(nDim);
    for (int i = 0 ; i < nDim; ++i)
    {
        mesh(i)[0] = nDim * poly(i);
        mesh(i)[1] = 2 * nDim * poly(i) + 1;
    }

    shared_ptr<FullGrid> subRegGrid = regGrid.getSubGrid(mesh);

    for (int is  = 0; is < nbSimul ; ++is)
    {
        if (subRegGrid->isInside(pointsSimul.col(is)))
        {
            ArrayXi iposReg = subRegGrid->lowerPositionCoord(pointsSimul.col(is));
            ArrayXd meshReg = subRegGrid->getMeshSize(iposReg);
        }
    }
    // test grid iterators
    shared_ptr<GridIterator> itersubRegGrid = subRegGrid->getGridIterator();

    while (itersubRegGrid->isValid())
    {
        Eigen::ArrayXd pointCoordReg = itersubRegGrid->getCoordinate();
        itersubRegGrid->next();
    }
}



BOOST_AUTO_TEST_CASE(testDynamicSerializationForLegendre)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    int nDim = 4 ;
    ArrayXd lowValues(nDim);
    for (int i = 0; i < nDim; ++i)
        lowValues(i) = static_cast<double>(i);
    ArrayXd step(nDim);
    for (int i = 0; i < nDim; ++i)
        step(i) = 0.6 * (i + 1);
    ArrayXi  nbStep(nDim);
    for (int i = 0; i < nDim; ++i)
        nbStep(i) = 3 * nDim * (i + 1);
    ArrayXi poly = ArrayXi::Constant(nDim, 3);
    // regular
    RegularLegendreGrid regGrid(lowValues, step, nbStep, poly);

    vector<shared_ptr<ArrayXd> > meshPerDimension(nDim);
    for (int i = 0; i < nDim; ++i)
    {
        meshPerDimension[i] = make_shared< ArrayXd >((nbStep(i) + 1));
        (*meshPerDimension[i]) = ArrayXd::LinSpaced(nbStep(i) + 1, lowValues(i), lowValues(i) + nbStep(i) * step(i));
    }

    // The archive to use
    {
        // default non compression
        BinaryFileArchive ar("archiveLegendre", "w");

        ar << Record(regGrid, "First grid", "Top") ;

        ar.flush();
    }
    {
        BinaryFileArchive ar("archiveLegendre", "r");
        RegularLegendreGrid regGridNew;
        Reference<RegularLegendreGrid>(ar, "First grid", "Top").restore(0, &regGridNew);

        for (int i = 0; i < nDim; ++i)
        {
            BOOST_CHECK_EQUAL(regGridNew.getLowValues()(i), lowValues(i));
            BOOST_CHECK_EQUAL(regGridNew.getStep()(i), step(i));
            BOOST_CHECK_EQUAL(regGridNew.getNbStep()(i), nbStep(i));
        }
    }
}


BOOST_AUTO_TEST_CASE(printLegendregrid)
{
    int nDim = 2 ;
    ArrayXd lowValues = ArrayXd::Constant(nDim, 1.);
    ArrayXd step = ArrayXd::Constant(nDim, 1.);
    ArrayXi  nbStep =  ArrayXi::Constant(nDim, 2);

    ArrayXi poly = ArrayXi::Constant(nDim, 8);

    // regular
    RegularLegendreGrid regGrid(lowValues, step, nbStep, poly);

    // iterator
    shared_ptr<GridIterator> iterRegGrid = regGrid.getGridIterator();

    // file to print
    std::fstream File("PrintGridLegendre", std::fstream::out);
    while (iterRegGrid->isValid())
    {
        Eigen::ArrayXd pointCoordReg = iterRegGrid->getCoordinate();
        File << pointCoordReg(0) << "  " << pointCoordReg(1) << std::endl ;
        iterRegGrid->next();
    }
    File.close();
}
