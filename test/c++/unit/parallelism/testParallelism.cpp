
//##define BOOST_TEST_MODULE testParallelism
#define BOOST_TEST_DYN_LINK
#include <fstream>
#include <functional>
#include <boost/mpi.hpp>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include "libflow/core/parallelism/ParallelComputeGridSplitting.h"
#include "libflow/core/utils/primeNumber.h"

using namespace std;
using namespace Eigen;
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

/// \class MeshExtension testParallelism.cpp
/// Defines extension of the region for parallelism
class MeshExtension
{
private :

    ArrayXi m_maxMesh;

public:

    /// \brief Constructor
    /// \param p_maxMesh   maximal number of mesh in each dimension
    explicit MeshExtension(const ArrayXi &p_maxMesh): m_maxMesh(p_maxMesh) {}

/// \brief extend the meshes
/// \param p_meshToExtend the initial meshing owned by a processor that will be extended
/// \return extended mesh
    Array<  array<int, 2 >, Dynamic, 1 > operator()(const Array<  array<int, 2 >, Dynamic, 1 > &p_meshToExtend)
    {
        Array<  array<int, 2 >, Dynamic, 1 >  ret(p_meshToExtend.size());
        for (int id = 0; id < p_meshToExtend.size(); ++id)
        {
            ret(id)[0] = std::max(p_meshToExtend(id)[0] - 10, 0) ;
            ret(id)[1] = std::min(2 * p_meshToExtend(id)[1], m_maxMesh(id));
        }
        return ret ;
    }
};

BOOST_AUTO_TEST_CASE(testParallelism1D)
{
    int iSize = 100;
    boost::mpi::communicator world;
    // initial dimension
    ArrayXi initialDimension(1);
    initialDimension(0) = iSize;
    // splitting for process
    ArrayXi splittingRatio(1);
    splittingRatio(0) = world.size();
    ParallelComputeGridSplitting paral(initialDimension, splittingRatio, world);
    // local grid
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridLocal = paral.getCurrentCalculationGrid();
    int iSizeLoc = gridLocal(0)[1] - gridLocal(0)[0];
    // data
    ArrayXXd data(2, iSizeLoc);
    for (int i = 0; i < iSizeLoc; ++i)
    {
        data(0, i) = exp(static_cast<double>(gridLocal(0)[0] + i) / iSize);
        data(1, i) = log1p(static_cast<double>(gridLocal(0)[0] + i) / iSize);
    }
    // now reconstruct on global mesh
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > globalGrid(1);
    globalGrid(0)[0] = 0;
    globalGrid(0)[1] = iSize;

    ArrayXXd dataRecons = paral.reconstruct(data, globalGrid, 0);

    if (world.rank() == 0)
    {
        for (int i = 0; i <  iSize; ++i)
        {
            BOOST_CHECK_CLOSE(dataRecons(0, i), exp(static_cast<double>(i) / iSize), accuracyEqual);
            BOOST_CHECK_CLOSE(dataRecons(1, i), log1p(static_cast<double>(i) / iSize), accuracyEqual);
        }
    }

    // create  special reconstruction
    globalGrid(0)[0] = iSize / 3;
    globalGrid(0)[1] = 2 * iSize / 3;
    int idim1 = 2 * iSize / 3 - iSize / 3;

    dataRecons.resize(2, idim1);
    dataRecons = paral.reconstruct(data, globalGrid, 0);

    if (world.rank() == 0)
    {
        for (int i = 0; i <  idim1; ++i)
        {
            BOOST_CHECK_CLOSE(dataRecons(0, i), exp(static_cast<double>(globalGrid(0)[0] + i) / iSize), accuracyEqual);
            BOOST_CHECK_CLOSE(dataRecons(1, i), log1p(static_cast<double>(globalGrid(0)[0] + i) / iSize), accuracyEqual);
        }
    }


    ArrayXXd dataReconsAll = paral.reconstructAll(data, globalGrid);

    for (int i = 0; i <  idim1; ++i)
    {
        BOOST_CHECK_CLOSE(dataReconsAll(0, i), exp(static_cast<double>(globalGrid(0)[0] + i) / iSize), accuracyEqual);
        BOOST_CHECK_CLOSE(dataReconsAll(1, i), log1p(static_cast<double>(globalGrid(0)[0] + i) / iSize), accuracyEqual);
    }
}

BOOST_AUTO_TEST_CASE(testParallelism1DOneD)
{
    int iSize = 100;
    boost::mpi::communicator world;
    // initial dimension
    ArrayXi initialDimension(1);
    initialDimension(0) = iSize;
    // splitting for process
    ArrayXi splittingRatio(1);
    splittingRatio(0) = world.size();
    ParallelComputeGridSplitting paral(initialDimension, splittingRatio, world);
    // local grid
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridLocal = paral.getCurrentCalculationGrid();
    int iSizeLoc = gridLocal(0)[1] - gridLocal(0)[0];
    // data
    ArrayXd data(iSizeLoc);
    for (int i = 0; i < iSizeLoc; ++i)
    {
        data(i) = exp(static_cast<double>(gridLocal(0)[0] + i) / iSize);
    }
    // now reconstruct on global mesh
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > globalGrid(1);
    globalGrid(0)[0] = 0;
    globalGrid(0)[1] = iSize;

    ArrayXd dataRecons = paral.reconstruct(data, globalGrid, 0);

    if (world.rank() == 0)
    {
        for (int i = 0; i <  iSize; ++i)
        {
            BOOST_CHECK_CLOSE(dataRecons(i), exp(static_cast<double>(i) / iSize), accuracyEqual);
        }
    }

    // create  special reconstruction
    globalGrid(0)[0] = iSize / 3;
    globalGrid(0)[1] = 2 * iSize / 3;
    int idim1 = 2 * iSize / 3 - iSize / 3;

    dataRecons.resize(idim1);
    dataRecons = paral.reconstruct(data, globalGrid, 0);

    if (world.rank() == 0)
    {
        for (int i = 0; i <  idim1; ++i)
        {
            BOOST_CHECK_CLOSE(dataRecons(i), exp(static_cast<double>(globalGrid(0)[0] + i) / iSize), accuracyEqual);
        }
    }
}


BOOST_AUTO_TEST_CASE(testParallelism2D)
{
    int iSize1 = 100;
    int iSize2 = 50;
    boost::mpi::communicator world;
    // initial dimension
    ArrayXi initialDimension(2);
    initialDimension(0) = iSize1;
    initialDimension(1) = iSize2;
    // splitting for process
    std::vector<int> prime = primeNumber(world.size());
    ArrayXi splittingRatio = ArrayXi::Constant(2, 1);
    for (size_t i = 0; i < prime.size(); ++i)
    {
        splittingRatio(i % 2) *= prime[i] ;
    }
    ParallelComputeGridSplitting paral(initialDimension, splittingRatio, world);
    // local grid
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridLocal = paral.getCurrentCalculationGrid();
    int iSizeLoc1 = (gridLocal(0)[1] - gridLocal(0)[0]);
    int iSizeLoc2 = (gridLocal(1)[1] - gridLocal(1)[0]);
    // data
    ArrayXXd data(2, iSizeLoc1 * iSizeLoc2);
    for (int j = 0; j < iSizeLoc2; ++j)
        for (int i = 0; i < iSizeLoc1; ++i)
        {
            data(0, i + j * iSizeLoc1) = exp(static_cast<double>(gridLocal(0)[0] + i) / iSize1) * exp(static_cast<double>(gridLocal(1)[0] + j) / iSize2);
            data(1, i + j * iSizeLoc1) = log1p(static_cast<double>(gridLocal(0)[0] + i) / iSize1) * log1p(static_cast<double>(gridLocal(1)[0] + j) / iSize2);
        }
    // now reconstruct on global mesh
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > globalGrid(2);
    globalGrid(0)[0] = 0;
    globalGrid(0)[1] = iSize1;
    globalGrid(1)[0] = 0;
    globalGrid(1)[1] = iSize2;

    ArrayXXd dataRecons = paral.reconstruct(data, globalGrid, 0);

    if (world.rank() == 0)
    {
        for (int j = 0; j < iSize2; ++j)
            for (int i = 0; i <  iSize1; ++i)
            {
                BOOST_CHECK_CLOSE(dataRecons(0, i + j * iSize1), exp(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*exp(static_cast<double>(globalGrid(1)[0] + j) / iSize2), accuracyEqual);
                BOOST_CHECK_CLOSE(dataRecons(1, i + j * iSize1), log1p(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*log1p(static_cast<double>(globalGrid(1)[0] + j) / iSize2), accuracyEqual);
            }
    }

    // create  special reconstruction
    globalGrid(0)[0] = iSize1 / 3;
    globalGrid(0)[1] = 2 * iSize1 / 3;
    int idim1 = 2 * iSize1 / 3 - iSize1 / 3;
    globalGrid(1)[0] = iSize2 / 3;
    globalGrid(1)[1] = 2 * iSize2 / 3;
    int idim2 = 2 * iSize2 / 3 - iSize2 / 3;

    dataRecons.resize(2, idim1 * idim2);
    dataRecons = paral.reconstruct(data, globalGrid, 0);

    if (world.rank() == 0)
    {
        for (int j = 0; j < idim2; ++j)
            for (int i = 0; i <  idim1; ++i)
            {
                BOOST_CHECK_CLOSE(dataRecons(0, i + j * idim1), exp(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*exp(static_cast<double>(globalGrid(1)[0] + j) / iSize2), accuracyEqual);
                BOOST_CHECK_CLOSE(dataRecons(1, i + j * idim1), log1p(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*log1p(static_cast<double>(globalGrid(1)[0] + j) / iSize2), accuracyEqual);
            }
    }


    ArrayXXd dataReconsAll = paral.reconstructAll(data, globalGrid);

    for (int j = 0; j < idim2; ++j)
        for (int i = 0; i <  idim1; ++i)
        {
            BOOST_CHECK_CLOSE(dataReconsAll(0, i + j * idim1), exp(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*exp(static_cast<double>(globalGrid(1)[0] + j) / iSize2), accuracyEqual);
            BOOST_CHECK_CLOSE(dataReconsAll(1, i + j * idim1), log1p(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*log1p(static_cast<double>(globalGrid(1)[0] + j) / iSize2), accuracyEqual);
        }
}

BOOST_AUTO_TEST_CASE(testParallelism2DOneDArray)
{
    int iSize1 = 100;
    int iSize2 = 50;
    boost::mpi::communicator world;
    // initial dimension
    ArrayXi initialDimension(2);
    initialDimension(0) = iSize1;
    initialDimension(1) = iSize2;
    // splitting for process
    std::vector<int> prime = primeNumber(world.size());
    ArrayXi splittingRatio = ArrayXi::Constant(2, 1);
    for (size_t i = 0; i < prime.size(); ++i)
    {
        splittingRatio(i % 2) *= prime[i] ;
    }
    ParallelComputeGridSplitting paral(initialDimension, splittingRatio, world);
    // local grid
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridLocal = paral.getCurrentCalculationGrid();
    int iSizeLoc1 = (gridLocal(0)[1] - gridLocal(0)[0]);
    int iSizeLoc2 = (gridLocal(1)[1] - gridLocal(1)[0]);
    // data
    ArrayXd data(iSizeLoc1 * iSizeLoc2);
    for (int j = 0; j < iSizeLoc2; ++j)
        for (int i = 0; i < iSizeLoc1; ++i)
        {
            data(i + j * iSizeLoc1) = exp(static_cast<double>(gridLocal(0)[0] + i) / iSize1) * exp(static_cast<double>(gridLocal(1)[0] + j) / iSize2);
        }
    // now reconstruct on global mesh
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > globalGrid(2);
    globalGrid(0)[0] = 0;
    globalGrid(0)[1] = iSize1;
    globalGrid(1)[0] = 0;
    globalGrid(1)[1] = iSize2;

    ArrayXd dataRecons = paral.reconstruct(data, globalGrid, 0);

    if (world.rank() == 0)
    {
        for (int j = 0; j < iSize2; ++j)
            for (int i = 0; i <  iSize1; ++i)
            {
                BOOST_CHECK_CLOSE(dataRecons(i + j * iSize1), exp(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*exp(static_cast<double>(globalGrid(1)[0] + j) / iSize2), accuracyEqual);
            }
    }

    // create  special reconstruction
    globalGrid(0)[0] = iSize1 / 3;
    globalGrid(0)[1] = 2 * iSize1 / 3;
    int idim1 = 2 * iSize1 / 3 - iSize1 / 3;
    globalGrid(1)[0] = iSize2 / 3;
    globalGrid(1)[1] = 2 * iSize2 / 3;
    int idim2 = 2 * iSize2 / 3 - iSize2 / 3;

    dataRecons.resize(idim1 * idim2);
    dataRecons = paral.reconstruct(data, globalGrid, 0);

    if (world.rank() == 0)
    {
        for (int j = 0; j < idim2; ++j)
            for (int i = 0; i <  idim1; ++i)
            {
                BOOST_CHECK_CLOSE(dataRecons(i + j * idim1), exp(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*exp(static_cast<double>(globalGrid(1)[0] + j) / iSize2), accuracyEqual);
            }
    }

}


BOOST_AUTO_TEST_CASE(testParallelism3D)
{
    int iSize1 = 70;
    int iSize2 = 50;
    int iSize3 = 60;
    boost::mpi::communicator world;
    // initial dimension
    ArrayXi initialDimension(3);
    initialDimension(0) = iSize1;
    initialDimension(1) = iSize2;
    initialDimension(2) = iSize3;
    // splitting for process
    std::vector<int> prime = primeNumber(world.size());
    ArrayXi splittingRatio = ArrayXi::Constant(3, 1);
    for (size_t i = 0; i < prime.size(); ++i)
    {
        splittingRatio(i % 3) *= prime[i] ;
    }
    ParallelComputeGridSplitting paral(initialDimension, splittingRatio, world);
    // local grid
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridLocal = paral.getCurrentCalculationGrid();
    int iSizeLoc1 = (gridLocal(0)[1] - gridLocal(0)[0]);
    int iSizeLoc2 = (gridLocal(1)[1] - gridLocal(1)[0]);
    int iSizeLoc3 = (gridLocal(2)[1] - gridLocal(2)[0]);
    // data
    ArrayXXd data(2, iSizeLoc1 * iSizeLoc2 * iSizeLoc3);
    for (int k = 0; k < iSizeLoc3; ++k)
        for (int j = 0; j < iSizeLoc2; ++j)
            for (int i = 0; i < iSizeLoc1; ++i)
            {
                data(0, i + j * iSizeLoc1 + k * iSizeLoc1 * iSizeLoc2) = exp(static_cast<double>(gridLocal(0)[0] + i) / iSize1) * exp(static_cast<double>(gridLocal(1)[0] + j) / iSize2) *
                        exp(static_cast<double>(gridLocal(2)[0] + k) / iSize3);
                data(1, i + j * iSizeLoc1 + k * iSizeLoc1 * iSizeLoc2) = log1p(static_cast<double>(gridLocal(0)[0] + i) / iSize1) * log1p(static_cast<double>(gridLocal(1)[0] + j) / iSize2) *
                        log1p(static_cast<double>(gridLocal(2)[0] + k) / iSize3);
            }
    // now reconstruct on global mesh
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > globalGrid(3);
    globalGrid(0)[0] = 0;
    globalGrid(0)[1] = iSize1;
    globalGrid(1)[0] = 0;
    globalGrid(1)[1] = iSize2;
    globalGrid(2)[0] = 0;
    globalGrid(2)[1] = iSize3;

    ArrayXXd dataRecons = paral.reconstruct(data, globalGrid, 0);

    if (world.rank() == 0)
    {
        for (int k = 0; k < iSize3; ++k)
            for (int j = 0; j < iSize2; ++j)
                for (int i = 0; i <  iSize1; ++i)
                {
                    BOOST_CHECK_CLOSE(dataRecons(0, i + j * iSize1 + k * iSize1 * iSize2), exp(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*exp(static_cast<double>(globalGrid(1)[0] + j) / iSize2)
                                      *exp(static_cast<double>(globalGrid(2)[0] + k) / iSize3), accuracyEqual);
                    BOOST_CHECK_CLOSE(dataRecons(1, i + j * iSize1 + k * iSize1 * iSize2), log1p(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*log1p(static_cast<double>(globalGrid(1)[0] + j) / iSize2)
                                      *log1p(static_cast<double>(globalGrid(2)[0] + k) / iSize3), accuracyEqual);
                }
    }

    // create  special reconstruction
    globalGrid(0)[0] = iSize1 / 3;
    globalGrid(0)[1] = 2 * iSize1 / 3;
    int idim1 = 2 * iSize1 / 3 - iSize1 / 3;
    globalGrid(1)[0] = iSize2 / 3;
    globalGrid(1)[1] = 2 * iSize2 / 3;
    int idim2 = 2 * iSize2 / 3 - iSize2 / 3;
    globalGrid(2)[0] = iSize3 / 3;
    globalGrid(2)[1] = 2 * iSize3 / 3;
    int idim3 = 2 * iSize3 / 3 - iSize3 / 3;

    dataRecons.resize(2, idim1 * idim2 * idim3);
    dataRecons = paral.reconstruct(data, globalGrid, 0);

    if (world.rank() == 0)
    {
        for (int k = 0; k < idim3; ++k)
            for (int j = 0; j < idim2; ++j)
                for (int i = 0; i <  idim1; ++i)
                {
                    BOOST_CHECK_CLOSE(dataRecons(0, i + j * idim1 + k * idim1 * idim2), exp(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*exp(static_cast<double>(globalGrid(1)[0] + j) / iSize2)
                                      *exp(static_cast<double>(globalGrid(2)[0] + k) / iSize3), accuracyEqual);
                    BOOST_CHECK_CLOSE(dataRecons(1, i + j * idim1 + k * idim1 * idim2), log1p(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*log1p(static_cast<double>(globalGrid(1)[0] + j) / iSize2)
                                      *log1p(static_cast<double>(globalGrid(2)[0] + k) / iSize3), accuracyEqual);
                }
    }

    ArrayXXd dataReconsAll = paral.reconstructAll(data, globalGrid);

    for (int k = 0; k < idim3; ++k)
        for (int j = 0; j < idim2; ++j)
            for (int i = 0; i <  idim1; ++i)
            {
                BOOST_CHECK_CLOSE(dataReconsAll(0, i + j * idim1 + k * idim1 * idim2), exp(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*exp(static_cast<double>(globalGrid(1)[0] + j) / iSize2)
                                  *exp(static_cast<double>(globalGrid(2)[0] + k) / iSize3), accuracyEqual);
                BOOST_CHECK_CLOSE(dataReconsAll(1, i + j * idim1 + k * idim1 * idim2), log1p(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*log1p(static_cast<double>(globalGrid(1)[0] + j) / iSize2)
                                  *log1p(static_cast<double>(globalGrid(2)[0] + k) / iSize3), accuracyEqual);
            }

}

BOOST_AUTO_TEST_CASE(testParallelism4D)
{
    int iSize1 = 30;
    int iSize2 = 20;
    int iSize3 = 10;
    int iSize4 = 15 ;
    boost::mpi::communicator world;
    // initial dimension
    ArrayXi initialDimension(4);
    initialDimension(0) = iSize1;
    initialDimension(1) = iSize2;
    initialDimension(2) = iSize3;
    initialDimension(3) = iSize4;
    // splitting for process
    std::vector<int> prime = primeNumber(world.size());
    ArrayXi splittingRatio = ArrayXi::Constant(4, 1);
    for (size_t i = 0; i < prime.size(); ++i)
    {
        splittingRatio(i % 4) *= prime[i] ;
    }
    ParallelComputeGridSplitting paral(initialDimension, splittingRatio, world);
    // local grid
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridLocal = paral.getCurrentCalculationGrid();
    int iSizeLoc1 = (gridLocal(0)[1] - gridLocal(0)[0]);
    int iSizeLoc2 = (gridLocal(1)[1] - gridLocal(1)[0]);
    int iSizeLoc3 = (gridLocal(2)[1] - gridLocal(2)[0]);
    int iSizeLoc4 = (gridLocal(3)[1] - gridLocal(3)[0]);
    // data
    ArrayXXd data(2, iSizeLoc1 * iSizeLoc2 * iSizeLoc3 * iSizeLoc4);
    for (int l = 0; l < iSizeLoc4; ++l)
        for (int k = 0; k < iSizeLoc3; ++k)
            for (int j = 0; j < iSizeLoc2; ++j)
                for (int i = 0; i < iSizeLoc1; ++i)
                {
                    int ipos = i + j * iSizeLoc1 + k * iSizeLoc1 * iSizeLoc2 + l * iSizeLoc1 * iSizeLoc2 * iSizeLoc3;
                    data(0, ipos) = exp(static_cast<double>(gridLocal(0)[0] + i) / iSize1) * exp(static_cast<double>(gridLocal(1)[0] + j) / iSize2) *
                                    exp(static_cast<double>(gridLocal(2)[0] + k) / iSize3) * exp(static_cast<double>(gridLocal(3)[0] + l) / iSize4);
                    data(1, ipos) = log1p(static_cast<double>(gridLocal(0)[0] + i) / iSize1) * log1p(static_cast<double>(gridLocal(1)[0] + j) / iSize2) *
                                    log1p(static_cast<double>(gridLocal(2)[0] + k) / iSize3) * log1p(static_cast<double>(gridLocal(3)[0] + l) / iSize4);
                }
    // now reconstruct on global mesh
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > globalGrid(4);
    globalGrid(0)[0] = 0;
    globalGrid(0)[1] = iSize1;
    globalGrid(1)[0] = 0;
    globalGrid(1)[1] = iSize2;
    globalGrid(2)[0] = 0;
    globalGrid(2)[1] = iSize3;
    globalGrid(3)[0] = 0;
    globalGrid(3)[1] = iSize4;

    ArrayXXd dataRecons = paral.reconstruct(data, globalGrid, 0);

    if (world.rank() == 0)
    {
        for (int l = 0; l < iSize4; ++l)
            for (int k = 0; k < iSize3; ++k)
                for (int j = 0; j < iSize2; ++j)
                    for (int i = 0; i <  iSize1; ++i)
                    {
                        int ipos = i + j * iSize1 + k * iSize1 * iSize2 + l * iSize1 * iSize2 * iSize3;
                        BOOST_CHECK_CLOSE(dataRecons(0, ipos), exp(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*exp(static_cast<double>(globalGrid(1)[0] + j) / iSize2)
                                          *exp(static_cast<double>(globalGrid(2)[0] + k) / iSize3)*exp(static_cast<double>(globalGrid(3)[0] + l) / iSize4), accuracyEqual);
                        BOOST_CHECK_CLOSE(dataRecons(1, ipos), log1p(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*log1p(static_cast<double>(globalGrid(1)[0] + j) / iSize2)
                                          *log1p(static_cast<double>(globalGrid(2)[0] + k) / iSize3)*log1p(static_cast<double>(globalGrid(3)[0] + l) / iSize4), accuracyEqual);
                    }
    }

    // create  special reconstruction
    globalGrid(0)[0] = iSize1 / 3;
    globalGrid(0)[1] = 2 * iSize1 / 3;
    int idim1 = 2 * iSize1 / 3 - iSize1 / 3;
    globalGrid(1)[0] = iSize2 / 3;
    globalGrid(1)[1] = 2 * iSize2 / 3;
    int idim2 = 2 * iSize2 / 3 - iSize2 / 3;
    globalGrid(2)[0] = iSize3 / 3;
    globalGrid(2)[1] = 2 * iSize3 / 3;
    int idim3 = 2 * iSize3 / 3 - iSize3 / 3;
    globalGrid(3)[0] = iSize4 / 3;
    globalGrid(3)[1] = 2 * iSize4 / 3;
    int idim4 = 2 * iSize4 / 3 - iSize4 / 3;

    dataRecons.resize(2, idim1 * idim2 * idim3 * idim4);
    dataRecons = paral.reconstruct(data, globalGrid, 0);

    if (world.rank() == 0)
    {
        for (int l = 0; l < idim4; ++l)
            for (int k = 0; k < idim3; ++k)
                for (int j = 0; j < idim2; ++j)
                    for (int i = 0; i <  idim1; ++i)
                    {
                        int ipos = i + j * idim1 + k * idim1 * idim2 + l * idim1 * idim2 * idim3;
                        BOOST_CHECK_CLOSE(dataRecons(0, ipos), exp(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*exp(static_cast<double>(globalGrid(1)[0] + j) / iSize2)
                                          *exp(static_cast<double>(globalGrid(2)[0] + k) / iSize3)*exp(static_cast<double>(globalGrid(3)[0] + l) / iSize4), accuracyEqual);
                        BOOST_CHECK_CLOSE(dataRecons(1, ipos), log1p(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*log1p(static_cast<double>(globalGrid(1)[0] + j) / iSize2)
                                          *log1p(static_cast<double>(globalGrid(2)[0] + k) / iSize3)*log1p(static_cast<double>(globalGrid(3)[0] + l) / iSize4), accuracyEqual);
                    }
    }

    ArrayXXd dataReconsAll = paral.reconstructAll(data, globalGrid);

    for (int l = 0; l < idim4; ++l)
        for (int k = 0; k < idim3; ++k)
            for (int j = 0; j < idim2; ++j)
                for (int i = 0; i <  idim1; ++i)
                {
                    int ipos = i + j * idim1 + k * idim1 * idim2 + l * idim1 * idim2 * idim3;
                    BOOST_CHECK_CLOSE(dataReconsAll(0, ipos), exp(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*exp(static_cast<double>(globalGrid(1)[0] + j) / iSize2)
                                      *exp(static_cast<double>(globalGrid(2)[0] + k) / iSize3)*exp(static_cast<double>(globalGrid(3)[0] + l) / iSize4), accuracyEqual);
                    BOOST_CHECK_CLOSE(dataReconsAll(1, ipos), log1p(static_cast<double>(globalGrid(0)[0] + i) / iSize1)*log1p(static_cast<double>(globalGrid(1)[0] + j) / iSize2)
                                      *log1p(static_cast<double>(globalGrid(2)[0] + k) / iSize3)*log1p(static_cast<double>(globalGrid(3)[0] + l) / iSize4), accuracyEqual);
                }

}

BOOST_AUTO_TEST_CASE(testParallelismWithCone1D)
{

    int iSize = 100;
    boost::mpi::communicator world;
    // initial dimension
    ArrayXi initialDimension(1);
    initialDimension(0) = iSize;
    function<  Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 >(const Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > &) > fMesh = MeshExtension(initialDimension);
    // splitting for process
    ArrayXi splittingRatio(1);
    splittingRatio(0) = world.size();
    ParallelComputeGridSplitting paral(initialDimension, fMesh, splittingRatio, world);
    // local grid
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridLocal = paral.getCurrentCalculationGrid();
    int iSizeLoc = gridLocal(0)[1] - gridLocal(0)[0];

    // data on grid
    ArrayXXd data(2, iSizeLoc);
    for (int i = 0; i < iSizeLoc; ++i)
    {
        data(0, i) = exp(static_cast<double>(gridLocal(0)[0] + i) / iSize);
        data(1, i) = log1p(static_cast<double>(gridLocal(0)[0] + i) / iSize);
    }

    // reconstruct data on cone defined by fMesh
    ArrayXXd dataRecons = paral.runOneStep(data);

    // get extend grid
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridExtendec = paral.getExtendedGridProcOldGrid();
    BOOST_CHECK_EQUAL(dataRecons.size(), 2 * (gridExtendec(0)[1] - gridExtendec(0)[0]));

    for (int i = 0; i <  dataRecons.cols(); ++i)
    {
        BOOST_CHECK_CLOSE(dataRecons(0, i), exp(static_cast<double>(gridExtendec(0)[0] + i) / iSize), accuracyEqual);
        BOOST_CHECK_CLOSE(dataRecons(1, i), log1p(static_cast<double>(gridExtendec(0)[0] + i) / iSize), accuracyEqual);
    }
}


BOOST_AUTO_TEST_CASE(testParallelismWithCone2D)
{
    int iSize1 = 100;
    int iSize2 = 50;
    boost::mpi::communicator world;
    // initial dimension
    ArrayXi initialDimension(2);
    initialDimension(0) = iSize1;
    initialDimension(1) = iSize2;
    function<  Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 >(const Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > &) > fMesh = MeshExtension(initialDimension);
    // splitting for process
    std::vector<int> prime = primeNumber(world.size());
    ArrayXi splittingRatio = ArrayXi::Constant(2, 1);
    for (size_t i = 0; i < prime.size(); ++i)
    {
        splittingRatio(i % 2) *= prime[i] ;
    }
    ParallelComputeGridSplitting paral(initialDimension, fMesh, splittingRatio, world);
    // local grid
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridLocal = paral.getCurrentCalculationGrid();
    int iSizeLoc1 = (gridLocal(0)[1] - gridLocal(0)[0]);
    int iSizeLoc2 = (gridLocal(1)[1] - gridLocal(1)[0]);
    // data
    ArrayXXd data(2, iSizeLoc1 * iSizeLoc2);
    for (int j = 0; j < iSizeLoc2; ++j)
        for (int i = 0; i < iSizeLoc1; ++i)
        {
            data(0, i + j * iSizeLoc1) = exp(static_cast<double>(gridLocal(0)[0] + i) / iSize1) * exp(static_cast<double>(gridLocal(1)[0] + j) / iSize2);
            data(1, i + j * iSizeLoc1) = log1p(static_cast<double>(gridLocal(0)[0] + i) / iSize1) * log1p(static_cast<double>(gridLocal(1)[0] + j) / iSize2);
        }

    ArrayXXd dataRecons =  paral.runOneStep(data);

    // get extend grid
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridExtendec = paral.getExtendedGridProcOldGrid();
    BOOST_CHECK_EQUAL(dataRecons.size(), 2 * (gridExtendec(0)[1] - gridExtendec(0)[0]) * (gridExtendec(1)[1] - gridExtendec(1)[0]));

    int iLoc1 = gridExtendec(0)[1] - gridExtendec(0)[0];
    int iLoc2 = gridExtendec(1)[1] - gridExtendec(1)[0];

    for (int j = 0; j < iLoc2; ++j)
        for (int i = 0; i <  iLoc1; ++i)
        {
            BOOST_CHECK_CLOSE(dataRecons(0, i + j * iLoc1), exp(static_cast<double>(gridExtendec(0)[0] + i) / iSize1)*exp(static_cast<double>(gridExtendec(1)[0] + j) / iSize2), accuracyEqual);
            BOOST_CHECK_CLOSE(dataRecons(1, i + j * iLoc1), log1p(static_cast<double>(gridExtendec(0)[0] + i) / iSize1)*log1p(static_cast<double>(gridExtendec(1)[0] + j) / iSize2), accuracyEqual);
        }
}

BOOST_AUTO_TEST_CASE(testParallelismWithCone3D)
{
    int iSize1 = 70;
    int iSize2 = 50;
    int iSize3 = 60;
    boost::mpi::communicator world;
    // initial dimension
    ArrayXi initialDimension(3);
    initialDimension(0) = iSize1;
    initialDimension(1) = iSize2;
    initialDimension(2) = iSize3;
    function<  Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 >(const Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > &) > fMesh = MeshExtension(initialDimension);
    // splitting for process
    std::vector<int> prime = primeNumber(world.size());
    ArrayXi splittingRatio = ArrayXi::Constant(3, 1);
    for (size_t i = 0; i < prime.size(); ++i)
    {
        splittingRatio(i % 3) *= prime[i] ;
    }
    ParallelComputeGridSplitting paral(initialDimension, fMesh, splittingRatio, world);
    // local grid
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridLocal = paral.getCurrentCalculationGrid();
    int iSizeLoc1 = (gridLocal(0)[1] - gridLocal(0)[0]);
    int iSizeLoc2 = (gridLocal(1)[1] - gridLocal(1)[0]);
    int iSizeLoc3 = (gridLocal(2)[1] - gridLocal(2)[0]);
    // data
    ArrayXXd data(2, iSizeLoc1 * iSizeLoc2 * iSizeLoc3);
    for (int k = 0; k < iSizeLoc3; ++k)
        for (int j = 0; j < iSizeLoc2; ++j)
            for (int i = 0; i < iSizeLoc1; ++i)
            {
                data(0, i + j * iSizeLoc1 + k * iSizeLoc1 * iSizeLoc2) = exp(static_cast<double>(gridLocal(0)[0] + i) / iSize1) * exp(static_cast<double>(gridLocal(1)[0] + j) / iSize2) *
                        exp(static_cast<double>(gridLocal(2)[0] + k) / iSize3);
                data(1, i + j * iSizeLoc1 + k * iSizeLoc1 * iSizeLoc2) = log1p(static_cast<double>(gridLocal(0)[0] + i) / iSize1) * log1p(static_cast<double>(gridLocal(1)[0] + j) / iSize2) *
                        log1p(static_cast<double>(gridLocal(2)[0] + k) / iSize3);
            }
    ArrayXXd dataRecons =  paral.runOneStep(data);

    // get extend grid
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridExtendec = paral.getExtendedGridProcOldGrid();
    BOOST_CHECK_EQUAL(dataRecons.size(), 2 * (gridExtendec(0)[1] - gridExtendec(0)[0]) * (gridExtendec(1)[1] - gridExtendec(1)[0]) * (gridExtendec(2)[1] - gridExtendec(2)[0]));

    int iLoc1 = gridExtendec(0)[1] - gridExtendec(0)[0];
    int iLoc2 = gridExtendec(1)[1] - gridExtendec(1)[0];
    int iLoc3 = gridExtendec(2)[1] - gridExtendec(2)[0];

    for (int k = 0; k < iLoc3; ++k)
        for (int j = 0; j < iLoc2; ++j)
            for (int i = 0; i <  iLoc1; ++i)
            {
                BOOST_CHECK_CLOSE(dataRecons(0, i + j * iLoc1 + k * iLoc1 * iLoc2), exp(static_cast<double>(gridExtendec(0)[0] + i) / iSize1)*exp(static_cast<double>(gridExtendec(1)[0] + j) / iSize2)
                                  *exp(static_cast<double>(gridExtendec(2)[0] + k) / iSize3), accuracyEqual);
                BOOST_CHECK_CLOSE(dataRecons(1, i + j * iLoc1 + k * iLoc1 * iLoc2), log1p(static_cast<double>(gridExtendec(0)[0] + i) / iSize1)*log1p(static_cast<double>(gridExtendec(1)[0] + j) / iSize2)
                                  *log1p(static_cast<double>(gridExtendec(2)[0] + k) / iSize3), accuracyEqual);
            }

}


BOOST_AUTO_TEST_CASE(testParallelismWithConeAndVaryingMesh1D)
{

    int iSize = 100;
    boost::mpi::communicator world;
    // initial dimension
    ArrayXi initialDimension(1);
    initialDimension(0) = iSize;
    // previous dimension
    ArrayXi initialDimensionPrev(1);
    initialDimensionPrev(0) = 2 * iSize;
    function<  Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 >(const Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > &) > fMesh = MeshExtension(initialDimension);
    // splitting for process
    ArrayXi splittingRatio(1);
    splittingRatio(0) = world.size();
    ParallelComputeGridSplitting paral(initialDimension, initialDimensionPrev, fMesh, splittingRatio, splittingRatio, world);
    // grid at previous step
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridLocalPrev = paral.getPreviousCalculationGrid();

    int iSizeLoc = gridLocalPrev(0)[1] - gridLocalPrev(0)[0];

    // data on grid
    ArrayXXd data(2, iSizeLoc);
    for (int i = 0; i < iSizeLoc; ++i)
    {
        data(0, i) = exp(static_cast<double>(gridLocalPrev(0)[0] + i) / iSize);
        data(1, i) = log1p(static_cast<double>(gridLocalPrev(0)[0] + i) / iSize);
    }

    // reconstruct data on cone defined by fMesh
    ArrayXXd dataRecons = paral.runOneStep(data);

    // get extend grid
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridExtendec = paral.getExtendedGridProcOldGrid();
    BOOST_CHECK_EQUAL(dataRecons.size(), 2 * (gridExtendec(0)[1] - gridExtendec(0)[0]));

    for (int i = 0; i <  dataRecons.cols(); ++i)
    {
        BOOST_CHECK_CLOSE(dataRecons(0, i), exp(static_cast<double>(gridExtendec(0)[0] + i) / iSize), accuracyEqual);
        BOOST_CHECK_CLOSE(dataRecons(1, i), log1p(static_cast<double>(gridExtendec(0)[0] + i) / iSize), accuracyEqual);
    }
}

BOOST_AUTO_TEST_CASE(testParallelismWithConeAndVaryingMesh2D)
{
    int iSize1 = 100;
    int iSize2 = 50;
    boost::mpi::communicator world;
    // initial dimension
    ArrayXi initialDimension(2);
    initialDimension(0) = iSize1;
    initialDimension(1) = iSize2;
    // previous dimension
    ArrayXi initialDimensionPrev(2);
    initialDimensionPrev = 2 * initialDimension;
    function<  Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 >(const Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > &) > fMesh = MeshExtension(initialDimension);
    // splitting for process
    std::vector<int> prime = primeNumber(world.size());
    ArrayXi splittingRatio = ArrayXi::Constant(2, 1);
    for (size_t i = 0; i < prime.size(); ++i)
    {
        splittingRatio(i % 2) *= prime[i] ;
    }
    ParallelComputeGridSplitting paral(initialDimension, initialDimensionPrev, fMesh, splittingRatio, splittingRatio, world);
    // grid at previous step
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridLocalPrev = paral.getPreviousCalculationGrid();
    int iSizeLoc1 = (gridLocalPrev(0)[1] - gridLocalPrev(0)[0]);
    int iSizeLoc2 = (gridLocalPrev(1)[1] - gridLocalPrev(1)[0]);
    // data
    ArrayXXd data(2, iSizeLoc1 * iSizeLoc2);
    for (int j = 0; j < iSizeLoc2; ++j)
        for (int i = 0; i < iSizeLoc1; ++i)
        {
            data(0, i + j * iSizeLoc1) = exp(static_cast<double>(gridLocalPrev(0)[0] + i) / iSize1) * exp(static_cast<double>(gridLocalPrev(1)[0] + j) / iSize2);
            data(1, i + j * iSizeLoc1) = log1p(static_cast<double>(gridLocalPrev(0)[0] + i) / iSize1) * log1p(static_cast<double>(gridLocalPrev(1)[0] + j) / iSize2);
        }

    ArrayXXd dataRecons =  paral.runOneStep(data);

    // get extend grid
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridExtendec = paral.getExtendedGridProcOldGrid();
    BOOST_CHECK_EQUAL(dataRecons.size(), 2 * (gridExtendec(0)[1] - gridExtendec(0)[0]) * (gridExtendec(1)[1] - gridExtendec(1)[0]));

    int iLoc1 = gridExtendec(0)[1] - gridExtendec(0)[0];
    int iLoc2 = gridExtendec(1)[1] - gridExtendec(1)[0];

    for (int j = 0; j < iLoc2; ++j)
        for (int i = 0; i <  iLoc1; ++i)
        {
            BOOST_CHECK_CLOSE(dataRecons(0, i + j * iLoc1), exp(static_cast<double>(gridExtendec(0)[0] + i) / iSize1)*exp(static_cast<double>(gridExtendec(1)[0] + j) / iSize2), accuracyEqual);
            BOOST_CHECK_CLOSE(dataRecons(1, i + j * iLoc1), log1p(static_cast<double>(gridExtendec(0)[0] + i) / iSize1)*log1p(static_cast<double>(gridExtendec(1)[0] + j) / iSize2), accuracyEqual);
        }
}


BOOST_AUTO_TEST_CASE(testParallelismWithConeAndVaryingMesh3D)
{
    int iSize1 = 70;
    int iSize2 = 50;
    int iSize3 = 60;
    boost::mpi::communicator world;
    // initial dimension
    ArrayXi initialDimension(3);
    initialDimension(0) = iSize1;
    initialDimension(1) = iSize2;
    initialDimension(2) = iSize3;
    // previous dimension
    ArrayXi initialDimensionPrev(3);
    initialDimensionPrev = 3 * initialDimension;
    function<  Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 >(const Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > &) > fMesh = MeshExtension(initialDimension);
    // splitting for process
    std::vector<int> prime = primeNumber(world.size());
    ArrayXi splittingRatio = ArrayXi::Constant(3, 1);
    for (size_t i = 0; i < prime.size(); ++i)
    {
        splittingRatio(i % 3) *= prime[i] ;
    }
    ParallelComputeGridSplitting paral(initialDimension, initialDimensionPrev, fMesh, splittingRatio, splittingRatio, world);
    // grid at previous step
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridLocalPrev = paral.getPreviousCalculationGrid();
    int iSizeLoc1 = (gridLocalPrev(0)[1] - gridLocalPrev(0)[0]);
    int iSizeLoc2 = (gridLocalPrev(1)[1] - gridLocalPrev(1)[0]);
    int iSizeLoc3 = (gridLocalPrev(2)[1] - gridLocalPrev(2)[0]);
    // data
    ArrayXXd data(2, iSizeLoc1 * iSizeLoc2 * iSizeLoc3);
    for (int k = 0; k < iSizeLoc3; ++k)
        for (int j = 0; j < iSizeLoc2; ++j)
            for (int i = 0; i < iSizeLoc1; ++i)
            {
                data(0, i + j * iSizeLoc1 + k * iSizeLoc1 * iSizeLoc2) = exp(static_cast<double>(gridLocalPrev(0)[0] + i) / iSize1) * exp(static_cast<double>(gridLocalPrev(1)[0] + j) / iSize2) *
                        exp(static_cast<double>(gridLocalPrev(2)[0] + k) / iSize3);
                data(1, i + j * iSizeLoc1 + k * iSizeLoc1 * iSizeLoc2) = log1p(static_cast<double>(gridLocalPrev(0)[0] + i) / iSize1) * log1p(static_cast<double>(gridLocalPrev(1)[0] + j) / iSize2) *
                        log1p(static_cast<double>(gridLocalPrev(2)[0] + k) / iSize3);
            }
    ArrayXXd dataRecons =  paral.runOneStep(data);

    // get extend grid
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridExtendec = paral.getExtendedGridProcOldGrid();
    BOOST_CHECK_EQUAL(dataRecons.size(), 2 * (gridExtendec(0)[1] - gridExtendec(0)[0]) * (gridExtendec(1)[1] - gridExtendec(1)[0]) * (gridExtendec(2)[1] - gridExtendec(2)[0]));

    int iLoc1 = gridExtendec(0)[1] - gridExtendec(0)[0];
    int iLoc2 = gridExtendec(1)[1] - gridExtendec(1)[0];
    int iLoc3 = gridExtendec(2)[1] - gridExtendec(2)[0];

    for (int k = 0; k < iLoc3; ++k)
        for (int j = 0; j < iLoc2; ++j)
            for (int i = 0; i <  iLoc1; ++i)
            {
                BOOST_CHECK_CLOSE(dataRecons(0, i + j * iLoc1 + k * iLoc1 * iLoc2), exp(static_cast<double>(gridExtendec(0)[0] + i) / iSize1)*exp(static_cast<double>(gridExtendec(1)[0] + j) / iSize2)
                                  *exp(static_cast<double>(gridExtendec(2)[0] + k) / iSize3), accuracyEqual);
                BOOST_CHECK_CLOSE(dataRecons(1, i + j * iLoc1 + k * iLoc1 * iLoc2), log1p(static_cast<double>(gridExtendec(0)[0] + i) / iSize1)*log1p(static_cast<double>(gridExtendec(1)[0] + j) / iSize2)
                                  *log1p(static_cast<double>(gridExtendec(2)[0] + k) / iSize3), accuracyEqual);
            }

}


BOOST_AUTO_TEST_CASE(testParallelismWithConeAndVaryingMesh4D)
{
    int iSize1 = 30;
    int iSize2 = 40;
    int iSize3 = 15;
    int iSize4 = 20;
    boost::mpi::communicator world;
    // initial dimension
    ArrayXi initialDimension(4);
    initialDimension(0) = iSize1;
    initialDimension(1) = iSize2;
    initialDimension(2) = iSize3;
    initialDimension(3) = iSize4;
    // previous dimension
    ArrayXi initialDimensionPrev(4);
    initialDimensionPrev = 3 * initialDimension;
    function<  Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 >(const Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > &) > fMesh = MeshExtension(initialDimension);
    // splitting for process
    std::vector<int> prime = primeNumber(world.size());
    ArrayXi splittingRatio = ArrayXi::Constant(4, 1);
    for (size_t i = 0; i < prime.size(); ++i)
    {
        splittingRatio(i % 4) *= prime[i] ;
    }
    ParallelComputeGridSplitting paral(initialDimension, initialDimensionPrev, fMesh, splittingRatio, splittingRatio, world);
    // grid at previous step
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridLocalPrev = paral.getPreviousCalculationGrid();
    int iSizeLoc1 = (gridLocalPrev(0)[1] - gridLocalPrev(0)[0]);
    int iSizeLoc2 = (gridLocalPrev(1)[1] - gridLocalPrev(1)[0]);
    int iSizeLoc3 = (gridLocalPrev(2)[1] - gridLocalPrev(2)[0]);
    int iSizeLoc4 = (gridLocalPrev(3)[1] - gridLocalPrev(3)[0]);
    // data
    ArrayXXd data(2, iSizeLoc1 * iSizeLoc2 * iSizeLoc3 * iSizeLoc4);
    for (int l = 0; l < iSizeLoc4; ++l)
        for (int k = 0; k < iSizeLoc3; ++k)
            for (int j = 0; j < iSizeLoc2; ++j)
                for (int i = 0; i < iSizeLoc1; ++i)
                {
                    int ipos = i + j * iSizeLoc1 + k * iSizeLoc1 * iSizeLoc2 + l * iSizeLoc1 * iSizeLoc2 * iSizeLoc3;
                    data(0, ipos) = exp(static_cast<double>(gridLocalPrev(0)[0] + i) / iSize1) * exp(static_cast<double>(gridLocalPrev(1)[0] + j) / iSize2) *
                                    exp(static_cast<double>(gridLocalPrev(2)[0] + k) / iSize3) * exp(static_cast<double>(gridLocalPrev(3)[0] + l) / iSize4);
                    data(1, ipos) = log1p(static_cast<double>(gridLocalPrev(0)[0] + i) / iSize1) * log1p(static_cast<double>(gridLocalPrev(1)[0] + j) / iSize2) *
                                    log1p(static_cast<double>(gridLocalPrev(2)[0] + k) / iSize3) * log1p(static_cast<double>(gridLocalPrev(3)[0] + l) / iSize4);
                }
    ArrayXXd dataRecons =  paral.runOneStep(data);

    // get extend grid
    Eigen::Array<  array<int, 2 >, Eigen::Dynamic, 1 > gridExtendec = paral.getExtendedGridProcOldGrid();
    BOOST_CHECK_EQUAL(dataRecons.size(), 2 * (gridExtendec(0)[1] - gridExtendec(0)[0]) * (gridExtendec(1)[1] - gridExtendec(1)[0]) * (gridExtendec(2)[1] - gridExtendec(2)[0]) *
                      (gridExtendec(3)[1] - gridExtendec(3)[0]));

    int iLoc1 = gridExtendec(0)[1] - gridExtendec(0)[0];
    int iLoc2 = gridExtendec(1)[1] - gridExtendec(1)[0];
    int iLoc3 = gridExtendec(2)[1] - gridExtendec(2)[0];
    int iLoc4 = gridExtendec(3)[1] - gridExtendec(3)[0];

    for (int l = 0; l < iLoc4; ++l)
        for (int k = 0; k < iLoc3; ++k)
            for (int j = 0; j < iLoc2; ++j)
                for (int i = 0; i <  iLoc1; ++i)
                {
                    int ipos = i + j * iLoc1 + k * iLoc1 * iLoc2 + l * iLoc1 * iLoc2 * iLoc3;
                    BOOST_CHECK_CLOSE(dataRecons(0, ipos), exp(static_cast<double>(gridExtendec(0)[0] + i) / iSize1)*exp(static_cast<double>(gridExtendec(1)[0] + j) / iSize2)
                                      *exp(static_cast<double>(gridExtendec(2)[0] + k) / iSize3)*exp(static_cast<double>(gridExtendec(3)[0] + l) / iSize4), accuracyEqual);
                    BOOST_CHECK_CLOSE(dataRecons(1, ipos), log1p(static_cast<double>(gridExtendec(0)[0] + i) / iSize1)*log1p(static_cast<double>(gridExtendec(1)[0] + j) / iSize2)
                                      *log1p(static_cast<double>(gridExtendec(2)[0] + k) / iSize3)*log1p(static_cast<double>(gridExtendec(3)[0] + l) / iSize4), accuracyEqual);
                }

}

// (empty) Initialization function. Can't use testing tools here.
bool init_function()
{
    return true;
}

int main(int argc, char *argv[])
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    boost::mpi::environment env(argc, argv);
    return ::boost::unit_test::unit_test_main(&init_function, argc, argv);
}
