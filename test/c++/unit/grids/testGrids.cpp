
#define BOOST_TEST_MODULE testGrid
#define BOOST_TEST_DYN_LINK
#include <fstream>
#include <functional>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include <cmath>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "libflow/core/utils/eigenGeners.h"
#include "libflow/core/grids/GeneralSpaceGrid.h"
#include "libflow/core/grids/RegularSpaceGrid.h"
#include "libflow/core/grids/FullRegularGridIterator.h"
#include "libflow/core/grids/FullGeneralGridIterator.h"
#include "libflow/core/grids/SpaceGridGeners.h"
#include "libflow/core/grids/FullGridGeners.h"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/core/grids/GeneralSpaceGridGeners.h"

using namespace std;
using namespace Eigen;
using namespace gs;
using namespace libflow;

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

BOOST_AUTO_TEST_CASE(testGridGeneration)
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
    // regular
    RegularSpaceGrid regGrid(lowValues, step, nbStep);

    vector<shared_ptr<ArrayXd> > meshPerDimension(nDim);
    for (int i = 0; i < nDim; ++i)
    {
        meshPerDimension[i] = make_shared< ArrayXd >((nbStep(i) + 1));
        (*meshPerDimension[i]) = ArrayXd::LinSpaced(nbStep(i) + 1, lowValues(i), lowValues(i) + nbStep(i) * step(i));
    }

    // general grid
    GeneralSpaceGrid genGrid(meshPerDimension);


    int nbSimul = 10000;
    // generate random points between 0 and 1
    ArrayXXd pointsSimul = 0.5 * (ArrayXXd::Constant(nDim, nbSimul, 1) + ArrayXXd::Random(nDim, nbSimul));
    for (int i = 0 ; i < nDim; ++i)
        pointsSimul.row(i) = ArrayXd::Constant(pointsSimul.cols(), lowValues(i)).transpose() +  nbStep(i) * step(i) * pointsSimul.row(i);

    for (int is  = 0; is < nbSimul ; ++is)
    {
        ArrayXi iposReg = regGrid.lowerPositionCoord(pointsSimul.col(is));
        ArrayXi iposGen = genGrid.lowerPositionCoord(pointsSimul.col(is));
        for (int i = 0; i < nDim; ++i)
            BOOST_CHECK_EQUAL(iposReg(i), iposGen(i));
        ArrayXd meshReg = regGrid.getMeshSize(iposReg);
        ArrayXd meshGen = genGrid.getMeshSize(iposGen);
        for (int i = 0; i < nDim; ++i)
            BOOST_CHECK_CLOSE(meshReg(i), meshGen(i), accuracyEqual);
    }
    for (int is  = 0; is < nbSimul ; ++is)
    {
        ArrayXi iposReg = regGrid.upperPositionCoord(pointsSimul.col(is));
        ArrayXi iposGen = genGrid.upperPositionCoord(pointsSimul.col(is));
        for (int i = 0; i < nDim; ++i)
            BOOST_CHECK_EQUAL(iposReg(i), iposGen(i));
    }

    // test sub grids
    Array< array<int, 2>, Dynamic, 1>   mesh(nDim);
    for (int i = 0 ; i < nDim; ++i)
    {
        mesh(i)[0] = nDim;
        mesh(i)[1] = 2 * nDim;
    }

    shared_ptr<FullGrid> subRegGrid = regGrid.getSubGrid(mesh);
    shared_ptr<FullGrid> subGenGrid = genGrid.getSubGrid(mesh);

    for (int is  = 0; is < nbSimul ; ++is)
    {
        if (subRegGrid->isInside(pointsSimul.col(is)))
        {
            ArrayXi iposReg = subRegGrid->lowerPositionCoord(pointsSimul.col(is));
            ArrayXi iposGen = subGenGrid->lowerPositionCoord(pointsSimul.col(is));
            for (int i = 0; i < nDim; ++i)
                BOOST_CHECK_EQUAL(iposReg(i), iposGen(i));
            ArrayXd meshReg = subRegGrid->getMeshSize(iposReg);
            ArrayXd meshGen = subGenGrid->getMeshSize(iposGen);
            for (int i = 0; i < nDim; ++i)
                BOOST_CHECK_CLOSE(meshReg(i), meshGen(i), accuracyEqual);
        }
    }

    // test grid iterators
    shared_ptr<GridIterator> itersubRegGrid = subRegGrid->getGridIterator();
    shared_ptr<GridIterator> itersubGenGrid = subGenGrid->getGridIterator();

    while (itersubRegGrid->isValid())
    {
        Eigen::ArrayXd pointCoordReg = itersubRegGrid->getCoordinate();
        Eigen::ArrayXd pointCoordGen = itersubGenGrid->getCoordinate();
        for (int i = 0 ; i < pointCoordReg.size(); ++i)
            BOOST_CHECK_CLOSE(pointCoordReg(i), pointCoordGen(i), accuracyEqual);
        itersubRegGrid->next();
        itersubGenGrid->next();
    }

}



BOOST_AUTO_TEST_CASE(testDynamicSerialization)
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
    // regular
    RegularSpaceGrid regGrid(lowValues, step, nbStep);

    vector<shared_ptr<ArrayXd> > meshPerDimension(nDim);
    for (int i = 0; i < nDim; ++i)
    {
        meshPerDimension[i] = make_shared< ArrayXd >((nbStep(i) + 1));
        (*meshPerDimension[i]) = ArrayXd::LinSpaced(nbStep(i) + 1, lowValues(i), lowValues(i) + nbStep(i) * step(i));
    }

    // general grid
    GeneralSpaceGrid genGrid(meshPerDimension);

    // The archive to use
    {
        // default non compression
        BinaryFileArchive ar("archiveGrid", "w");

        ar << Record(regGrid, "First grid", "Top") << Record(genGrid, "Second grid", "Top");

        ar.flush();
    }

    {
        BinaryFileArchive ar("archiveGrid", "r");
        RegularSpaceGrid regGridNew;
        Reference<RegularSpaceGrid>(ar, "First grid", "Top").restore(0, &regGridNew);
        GeneralSpaceGrid genGridNew;
        Reference<GeneralSpaceGrid>(ar, "Second grid", "Top").restore(0, &genGridNew);

        const std::vector<shared_ptr<Eigen::ArrayXd> >   &vMesh = genGridNew.getMeshPerDimension();
        for (int i = 0; i < nDim; ++i)
        {
            BOOST_CHECK_EQUAL(regGridNew.getLowValues()(i), lowValues(i));
            BOOST_CHECK_EQUAL(regGridNew.getStep()(i), step(i));
            BOOST_CHECK_EQUAL(regGridNew.getNbStep()(i), nbStep(i));
            for (int j = 0; j < meshPerDimension[i]->size(); ++j)
                BOOST_CHECK_EQUAL((*vMesh[i])(j), (*meshPerDimension[i])(j));
        }
    }
}

BOOST_AUTO_TEST_CASE(MoreSerializationGrid)
{
    {
        gs::BinaryFileArchive ar("toto", "w");
        std::vector< shared_ptr< ArrayXd > > phiIn(2);
        phiIn[0] = make_shared< ArrayXd>(5);
        phiIn[0]->setConstant(5.);
        phiIn[1] = make_shared< ArrayXd>(10);
        phiIn[1]->setConstant(10.);
        ar << gs::Record(phiIn, "phi", "");
        ArrayXd lowValues = ArrayXd::Constant(1, 0.);
        ArrayXd step  = ArrayXd::Constant(1, 1.);
        ArrayXi nstep = ArrayXi::Constant(1, 1);
        shared_ptr<SpaceGrid>  grid = make_shared<RegularSpaceGrid>(lowValues, step, nstep);
        ar << gs::Record(*grid, "grid", "");
        ar.flush();
    }
    {
        gs::BinaryFileArchive ar("toto", "r");
        std::vector< shared_ptr< ArrayXd > > phiIn;
        gs::Reference<decltype(phiIn)>(ar, "phi", "").restore(0, &phiIn);
        shared_ptr<SpaceGrid>  grid = gs::Reference<SpaceGrid>(ar, "grid", "").get(0);
        shared_ptr<RegularSpaceGrid> gridCast = static_pointer_cast<RegularSpaceGrid> (grid);
        BOOST_CHECK_EQUAL(gridCast->getLowValues()(0), 0.);
    }
}
