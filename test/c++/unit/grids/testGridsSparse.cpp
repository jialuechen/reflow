
#define BOOST_TEST_MODULE testSparseGrid
#define BOOST_TEST_DYN_LINK
#include <memory>
#include <fstream>
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>
#include <Eigen/Dense>
#include <cmath>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "libflow/core/grids/SparseSpaceGridBound.h"
#include "libflow/core/grids/SparseSpaceGridNoBound.h"
#include "libflow/core/grids/SparseGridBoundIterator.h"
#include "libflow/core/grids/SparseSpaceGridBoundGeners.h"
#include "libflow/core/grids/SparseSpaceGridNoBoundGeners.h"

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

/// \brief Sparse grid generation test
/// \param p_level  maximal sparse grid level
/// \param p_weight weights associated to sparse grids
void testSparseGridGenerationBound(int p_level, const Eigen::ArrayXd &p_weight)
{

#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    Eigen::ArrayXd  lowValues(p_weight.size());
    for (int i = 0; i < p_weight.size(); ++i)
        lowValues(i) = static_cast<double>(i);
    Eigen::ArrayXd  sizeDomain(p_weight.size());
    for (int i = 0; i < p_weight.size(); ++i)
        sizeDomain(i) = (i + 1) * 10.;

    // linear
    size_t degree = 1;

    // sparse grid generation
    SparseSpaceGridBound sparseGrid(lowValues, sizeDomain, p_level, p_weight, degree);

    // test grid iterators
    shared_ptr<GridIterator > iterGrid = sparseGrid.getGridIterator();

    ArrayXXd values(p_weight.size(), sparseGrid.getNbPoints());
    while (iterGrid->isValid())
    {
        Eigen::ArrayXd pointCoordReg = iterGrid->getCoordinate();
        values.col(iterGrid->getCount()) = pointCoordReg ;
        iterGrid->next();
    }

    // The archive to use
    {
        // default non compression
        BinaryFileArchive ar("archiveSparse", "w");

        ar << Record(sparseGrid, "First grid", "Top") ;

    }
    {
        BinaryFileArchive ar("archiveSparse", "r");
        SparseSpaceGridBound gridNew;
        Reference<SparseSpaceGridBound>(ar, "First grid", "Top").restore(0, &gridNew);

        shared_ptr<GridIterator > iterNewGrid = gridNew.getGridIterator();
        while (iterNewGrid->isValid())
        {
            Eigen::ArrayXd pointCoordReg = iterNewGrid->getCoordinate();
            ArrayXd oldCoord = values.col(iterNewGrid->getCount());
            for (int i = 0; i < p_weight.size(); ++i)
                BOOST_CHECK_EQUAL(oldCoord(i), pointCoordReg(i));
            iterNewGrid->next();
        }
    }
}

void testSparseGridBoundPlot(int p_level, const Eigen::ArrayXd &p_weight)
{

#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    Eigen::ArrayXd  lowValues = Eigen::ArrayXd::Constant(p_weight.size(), 0.);
    Eigen::ArrayXd  sizeDomain = Eigen::ArrayXd::Constant(p_weight.size(), 1.);

    // linear
    size_t degree = 1;

    // sparse grid generation
    SparseSpaceGridBound sparseGrid(lowValues, sizeDomain, p_level, p_weight, degree);

    // test grid iterators
    shared_ptr<GridIterator > iterGrid = sparseGrid.getGridIterator();

    std::string fileName = "SparseGridLevelBound" + boost::lexical_cast<std::string>(p_level) + "Dim" + boost::lexical_cast<std::string>(p_weight.size());
    std::fstream File(fileName.c_str(), std::fstream::out);
    while (iterGrid->isValid())
    {
        Eigen::ArrayXd pointCoordReg = iterGrid->getCoordinate();
        for (int i = 0; i < pointCoordReg.size() ; ++i)
            File << pointCoordReg(i) << " " ;
        File << std::endl ;
        iterGrid->next();
    }
    File.close();
}


BOOST_AUTO_TEST_CASE(testSparseGridBound)
{
    // 1D, level 1
    ArrayXd  weight1D = ArrayXd::Constant(1, 1.);
    testSparseGridGenerationBound(1, weight1D);
    // 1D, level 3
    testSparseGridGenerationBound(2, weight1D);

    // 2D , level 2
    ArrayXd  weight2D = ArrayXd::Constant(2, 1.);
    testSparseGridGenerationBound(2, weight2D);
    // 2D , level 4
    testSparseGridGenerationBound(4, weight2D);
    // 3D , level 4
    ArrayXd  weight3D = ArrayXd::Constant(3, 1.);
    testSparseGridGenerationBound(4, weight3D);
    // 5D , level 5
    ArrayXd  weight5D = ArrayXd::Constant(5, 1.);
    testSparseGridGenerationBound(5, weight5D);

}

BOOST_AUTO_TEST_CASE(testSparseBoundPlot)
{
    ArrayXd  weight2D = ArrayXd::Constant(2, 1.);
    testSparseGridBoundPlot(5, weight2D);

    ArrayXd  weight3D = ArrayXd::Constant(3, 1.);
    testSparseGridBoundPlot(5, weight3D);

}
/// \brief Sparse grid generation test
/// \param p_level  maximal sparse grid level
/// \param p_weight weights associated to sparse grids
void testSparseGridGenerationNoBound(int p_level, const Eigen::ArrayXd &p_weight)
{

#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    Eigen::ArrayXd  lowValues(p_weight.size());
    for (int i = 0; i < p_weight.size(); ++i)
        lowValues(i) = static_cast<double>(i);
    Eigen::ArrayXd  sizeDomain(p_weight.size());
    for (int i = 0; i < p_weight.size(); ++i)
        sizeDomain(i) = (i + 1) * 10.;

    // linear
    size_t degree = 1;

    // sparse grid generation
    SparseSpaceGridNoBound sparseGrid(lowValues, sizeDomain, p_level, p_weight, degree);

    // test grid iterators
    shared_ptr<GridIterator > iterGrid = sparseGrid.getGridIterator();

    ArrayXXd values(p_weight.size(), sparseGrid.getNbPoints());
    while (iterGrid->isValid())
    {
        Eigen::ArrayXd pointCoordReg = iterGrid->getCoordinate();
        values.col(iterGrid->getCount()) = pointCoordReg ;
        iterGrid->next();
    }

    // The archive to use
    {
        // default non compression
        BinaryFileArchive ar("archiveSparse", "w");

        ar << Record(sparseGrid, "First grid", "Top") ;

        ar.flush();
    }
    {
        BinaryFileArchive ar("archiveSparse", "r");
        SparseSpaceGridNoBound gridNew;
        Reference<SparseSpaceGridNoBound>(ar, "First grid", "Top").restore(0, &gridNew);

        shared_ptr<GridIterator > iterNewGrid = gridNew.getGridIterator();
        while (iterNewGrid->isValid())
        {
            Eigen::ArrayXd pointCoordReg = iterNewGrid->getCoordinate();
            ArrayXd oldCoord = values.col(iterNewGrid->getCount());
            for (int i = 0; i < p_weight.size(); ++i)
                BOOST_CHECK_EQUAL(oldCoord(i), pointCoordReg(i));
            iterNewGrid->next();
        }
    }
}

void testSparseGridNoBoundPlot(int p_level, const Eigen::ArrayXd &p_weight)
{

#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    Eigen::ArrayXd  lowValues = Eigen::ArrayXd::Constant(p_weight.size(), 0.);
    Eigen::ArrayXd  sizeDomain = Eigen::ArrayXd::Constant(p_weight.size(), 1.);

    // linear
    size_t degree = 1;

    // sparse grid generation
    SparseSpaceGridNoBound sparseGrid(lowValues, sizeDomain, p_level, p_weight, degree);

    // test grid iterators
    shared_ptr<GridIterator > iterGrid = sparseGrid.getGridIterator();

    std::string fileName = "SparseGridLevelNoBound" + boost::lexical_cast<std::string>(p_level) + "Dim" + boost::lexical_cast<std::string>(p_weight.size());
    std::fstream File(fileName.c_str(), std::fstream::out);
    while (iterGrid->isValid())
    {
        Eigen::ArrayXd pointCoordReg = iterGrid->getCoordinate();
        for (int i = 0; i < pointCoordReg.size() ; ++i)
            File << pointCoordReg(i) << " " ;
        File << std::endl ;
        iterGrid->next();
    }
    File.close();
}


BOOST_AUTO_TEST_CASE(testSparseGridNoBound)
{
    // 1D, level 1
    ArrayXd  weight1D = ArrayXd::Constant(1, 1.);
    testSparseGridGenerationNoBound(1, weight1D);
    // 1D, level 3
    testSparseGridGenerationNoBound(2, weight1D);

    // 2D , level 2
    ArrayXd  weight2D = ArrayXd::Constant(2, 1.);
    testSparseGridGenerationNoBound(2, weight2D);
    // 2D , level 4
    testSparseGridGenerationNoBound(4, weight2D);
    // 3D , level 4
    ArrayXd  weight3D = ArrayXd::Constant(3, 1.);
    testSparseGridGenerationNoBound(4, weight3D);
    // 5D , level 5
    ArrayXd  weight5D = ArrayXd::Constant(5, 1.);
    testSparseGridGenerationNoBound(5, weight5D);

}


BOOST_AUTO_TEST_CASE(testSparseNoBoundPlot)
{
    ArrayXd  weight2D = ArrayXd::Constant(2, 1.);
    testSparseGridNoBoundPlot(5, weight2D);

    ArrayXd  weight3D = ArrayXd::Constant(3, 1.);
    testSparseGridNoBoundPlot(5, weight3D);

}
