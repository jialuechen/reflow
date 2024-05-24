
#define BOOST_TEST_MODULE testSerialization
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <Eigen/Sparse>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "reflow/core/utils/eigenGeners.h"
#include "reflow/regression/MultiVariateBasisGeners.h"
#include "reflow/regression/GlobalRegressionGeners.h"

using namespace Eigen;
using namespace gs;
using namespace reflow;

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

typedef Eigen::Triplet<double> T;


BOOST_AUTO_TEST_CASE(testMatrix)
{
    {
        SparseMatrix<double> A1(3, 3);

        std::vector<T> triplets;
        triplets.push_back(T(0, 0, 1.));
        triplets.push_back(T(0, 1, 2.));
        triplets.push_back(T(0, 2, 3.));
        triplets.push_back(T(1, 1, 4.));
        triplets.push_back(T(2, 1, 5.));
        triplets.push_back(T(2, 2, 6.));
        A1.setFromTriplets(triplets.begin(), triplets.end());
        std::cout << "Mat " << A1 << std::endl ;
        {
            BinaryFileArchive ar("archiveSer", "w");
            ar <<  Record(A1, "Sparse", "Top") ;
        }
        {
            BinaryFileArchive ar("archiveSer", "r");
            SparseMatrix< double > matRead;
            Reference<SparseMatrix<double> >(ar, "Sparse", "Top").restore(0, &matRead);
            std::cout << "Mat read " << matRead << std::endl ;
        }
    }
    {
        SparseMatrix<double, Eigen::RowMajor> A1(3, 3);

        std::vector<T> triplets;
        triplets.push_back(T(0, 0, 1.));
        triplets.push_back(T(0, 1, 2.));
        triplets.push_back(T(0, 2, 3.));
        triplets.push_back(T(1, 1, 4.));
        triplets.push_back(T(2, 1, 5.));
        triplets.push_back(T(2, 2, 6.));
        A1.setFromTriplets(triplets.begin(), triplets.end());
        {
            BinaryFileArchive ar("archiveSER", "w");
            ar <<  Record(A1, "Sparse", "Top") ;
        }
        {
            BinaryFileArchive ar("archiveSER", "r");
            SparseMatrix< double, Eigen::RowMajor > matRead;
            Reference<SparseMatrix<double, Eigen::RowMajor> >(ar, "Sparse", "Top").restore(0, &matRead);
        }
    }
}


BOOST_AUTO_TEST_CASE(MultiVariateHermite)
{
    int deg = 4;
    int dim = 5;
    ComputeDegreeSum cDegree;
    MultiVariateBasis<Hermite> basis(cDegree, dim, deg);
    {
        BinaryFileArchive ar("archiveSER1", "w");
        ar <<  Record(basis, "MultiVariate", "Top") ;
    }
    {
        BinaryFileArchive ar("archiveSER1", "r");
        MultiVariateBasis<Hermite>  basisRead;
        Reference< MultiVariateBasis<Hermite>  >(ar, "MultiVariate", "Top").restore(0, &basisRead);
    }
}



BOOST_AUTO_TEST_CASE(MultiVariateCanonical)
{
    int deg = 4;
    int dim = 5;
    ComputeDegreeSum cDegree;
    MultiVariateBasis<Canonical> basis(cDegree, dim, deg);
    {
        BinaryFileArchive ar("archiveSER2", "w");
        ar <<  Record(basis, "MultiVariate", "Top") ;
    }
    {
        BinaryFileArchive ar("archiveSER2", "r");
        MultiVariateBasis<Canonical>  basisRead;
        Reference< MultiVariateBasis<Canonical>  >(ar, "MultiVariate", "Top").restore(0, &basisRead);
    }

}

BOOST_AUTO_TEST_CASE(MultiVariateTchebychev)
{
    int deg = 4;
    int dim = 5;
    ComputeDegreeSum cDegree;
    MultiVariateBasis<Tchebychev> basis(cDegree, dim, deg);
    {
        BinaryFileArchive ar("archiveSER3", "w");
        ar <<  Record(basis, "MultiVariate", "Top") ;
    }
    {
        BinaryFileArchive ar("archiveSER3", "r");
        MultiVariateBasis<Tchebychev>  basisRead;
        Reference< MultiVariateBasis<Tchebychev>  >(ar, "MultiVariate", "Top").restore(0, &basisRead);
    }

}

BOOST_AUTO_TEST_CASE(GlobalRegressionSerialization)
{
    int ndegree = 3;
    int ndim = 4;
    GlobalRegression<Hermite> regressor(ndegree, ndim);

    {
        BinaryFileArchive ar("archiveSER4", "w");
        ar << Record(regressor, "GlobalRegressor", "Top");
    }
    std::cout << " PASS " << std::endl ;
    {
        BinaryFileArchive ar("archiveSER4", "r");
        GlobalRegression<Hermite> regressorRead;
        Reference< GlobalRegression<Hermite> >(ar, "GlobalRegressor", "Top").restore(0, &regressorRead);
    }

}
