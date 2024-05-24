// Copyright (C) 2016 Jerome Lelong <jerome.lelong@imag.fr>

#define BOOST_TEST_MODULE testMultiVariateBasis
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/function.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <ctime>
#include "reflow/regression/MultiVariateBasis.h"
#include "reflow/core/utils/Polynomials1D.h"

using namespace std;
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


/**
 * \class CanonicalCheck
 * \brief Check the 1D Canonical polynomials
 */
struct CanonicalCheck : public Canonical
{
    /**
     * \brief Check the recurrence relation between the derivatives
     *
     * \param p_x Evaluation point
     * \param p_d degree
     */
    void operator()(double p_x, int p_d) const
    {
        double accuracyEqual = 1E-6;
        if (p_d == 1)
        {
            double D2val = D2F(p_x, p_d);
            BOOST_CHECK_EQUAL(D2val, 0.);
        }
        if (p_d == 0)
        {
            double Dval = DF(p_x, p_d);
            BOOST_CHECK_EQUAL(Dval, 0.);
        }
        if (p_d >= 2)
        {
            double val = F(p_x, p_d);
            double Dval = DF(p_x, p_d);
            double D2val = D2F(p_x, p_d);
            BOOST_CHECK_CLOSE(p_d * val, p_x * Dval, accuracyEqual);
            BOOST_CHECK_CLOSE(p_x * D2val, (p_d - 1) * Dval, accuracyEqual);
        }
    }
};

/**
 * \class TchebychevCheck
 * \brief Check the 1D Tchebychev polynomials
 */
struct TchebychevCheck : public Tchebychev
{
    /**
     * \brief Check the recurrence relation between the derivatives
     *
     * \param p_x Evaluation point
     * \param p_d degree
     */
    void operator()(double p_x, int p_d) const
    {
        double accuracyEqual = 1E-6;
        if (p_d == 1)
        {
            double D2val = D2F(p_x, p_d);
            BOOST_CHECK_EQUAL(D2val, 0.);
        }
        if (p_d == 0)
        {
            double Dval = DF(p_x, p_d);
            BOOST_CHECK_EQUAL(Dval, 0.);
        }
        if (p_d >= 2)
        {
            double val = F(p_x, p_d);
            double val_1 = F(p_x, p_d + 1);
            double val_2 = F(p_x, p_d + 2);
            double Dval = DF(p_x, p_d);
            double D2val = D2F(p_x, p_d);
            // Check the ODE
            BOOST_CHECK_CLOSE((1 - p_x * p_x) * D2val, p_x * Dval - p_d * p_d * val, accuracyEqual);
            // Check recurrence
            BOOST_CHECK_CLOSE(val_2, 2 * p_x * val_1 - val, accuracyEqual);
        }
    }
};

/**
 * \class HermiteCheck
 * \brief Check the 1D Tchebychev polynomials
 */
struct HermiteCheck : public Hermite
{
    /**
     * \brief Check the recurrence relation between the derivatives
     *
     * \param p_x Evaluation point
     * \param p_d degree
     */
    void operator()(double p_x, int p_d)
    {
        double accuracyEqual = 1E-6;
        if (p_d == 1)
        {
            double D2val = D2F(p_x, p_d);
            BOOST_CHECK_EQUAL(D2val, 0.);
        }
        if (p_d == 0)
        {
            double Dval = DF(p_x, p_d);
            BOOST_CHECK_EQUAL(Dval, 0.);
        }
        if (p_d >= 2)
        {
            double val = F(p_x, p_d);
            double val_1 = F(p_x, p_d + 1);
            double val_minus_1 = F(p_x, p_d - 1);
            double Dval = DF(p_x, p_d);
            double D2val = D2F(p_x, p_d);
            // Check the recurrence
            BOOST_CHECK_CLOSE(p_x * val - Dval, val_1, accuracyEqual);
            BOOST_CHECK_CLOSE(p_x * val - p_d * val_minus_1, val_1, accuracyEqual);
            // Check the ODE
            BOOST_CHECK_CLOSE(-p_x * Dval + D2val, -p_d * val, accuracyEqual);
        }
    }
};

void testEvaluationPolynomials1D(std::function<void(double, int)> p_check)
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    int nTtests = 100;
    int maxDegree = 10;
    // generateur Mersene Twister
    boost::mt19937 generator(time(0));
    boost::normal_distribution<double> alea_n;
    boost::variate_generator< boost::mt19937 &, boost::normal_distribution<double> > normal_random(generator, alea_n);
    for (int i = 0; i < nTtests; i++)
    {
        double x = normal_random();
        for (int d = 0; d < maxDegree; d++)
        {
            p_check(x, d);
        }
    }
}


// Test recurrence formulae for Polynomials1D
BOOST_AUTO_TEST_CASE(testCanonical1D)
{
    testEvaluationPolynomials1D(CanonicalCheck());
}

BOOST_AUTO_TEST_CASE(testTchebychev1D)
{
    testEvaluationPolynomials1D(TchebychevCheck());
}

BOOST_AUTO_TEST_CASE(testHermite1D)
{
    testEvaluationPolynomials1D(HermiteCheck());
}

// Test tensor constructors
BOOST_AUTO_TEST_CASE(testMultiVariateSumConstructor)
{
    Eigen::MatrixXi TensorSumExpected(126, 5);
    TensorSumExpected << 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 1,
                      0, 0, 0, 1, 0,
                      0, 0, 1, 0, 0,
                      0, 1, 0, 0, 0,
                      1, 0, 0, 0, 0,
                      0, 0, 0, 0, 2,
                      0, 0, 0, 1, 1,
                      0, 0, 1, 0, 1,
                      0, 1, 0, 0, 1,
                      1, 0, 0, 0, 1,
                      0, 0, 0, 2, 0,
                      0, 0, 1, 1, 0,
                      0, 1, 0, 1, 0,
                      1, 0, 0, 1, 0,
                      0, 0, 2, 0, 0,
                      0, 1, 1, 0, 0,
                      1, 0, 1, 0, 0,
                      0, 2, 0, 0, 0,
                      1, 1, 0, 0, 0,
                      2, 0, 0, 0, 0,
                      0, 0, 0, 0, 3,
                      0, 0, 0, 1, 2,
                      0, 0, 1, 0, 2,
                      0, 1, 0, 0, 2,
                      1, 0, 0, 0, 2,
                      0, 0, 0, 2, 1,
                      0, 0, 1, 1, 1,
                      0, 1, 0, 1, 1,
                      1, 0, 0, 1, 1,
                      0, 0, 2, 0, 1,
                      0, 1, 1, 0, 1,
                      1, 0, 1, 0, 1,
                      0, 2, 0, 0, 1,
                      1, 1, 0, 0, 1,
                      2, 0, 0, 0, 1,
                      0, 0, 0, 3, 0,
                      0, 0, 1, 2, 0,
                      0, 1, 0, 2, 0,
                      1, 0, 0, 2, 0,
                      0, 0, 2, 1, 0,
                      0, 1, 1, 1, 0,
                      1, 0, 1, 1, 0,
                      0, 2, 0, 1, 0,
                      1, 1, 0, 1, 0,
                      2, 0, 0, 1, 0,
                      0, 0, 3, 0, 0,
                      0, 1, 2, 0, 0,
                      1, 0, 2, 0, 0,
                      0, 2, 1, 0, 0,
                      1, 1, 1, 0, 0,
                      2, 0, 1, 0, 0,
                      0, 3, 0, 0, 0,
                      1, 2, 0, 0, 0,
                      2, 1, 0, 0, 0,
                      3, 0, 0, 0, 0,
                      0, 0, 0, 0, 4,
                      0, 0, 0, 1, 3,
                      0, 0, 1, 0, 3,
                      0, 1, 0, 0, 3,
                      1, 0, 0, 0, 3,
                      0, 0, 0, 2, 2,
                      0, 0, 1, 1, 2,
                      0, 1, 0, 1, 2,
                      1, 0, 0, 1, 2,
                      0, 0, 2, 0, 2,
                      0, 1, 1, 0, 2,
                      1, 0, 1, 0, 2,
                      0, 2, 0, 0, 2,
                      1, 1, 0, 0, 2,
                      2, 0, 0, 0, 2,
                      0, 0, 0, 3, 1,
                      0, 0, 1, 2, 1,
                      0, 1, 0, 2, 1,
                      1, 0, 0, 2, 1,
                      0, 0, 2, 1, 1,
                      0, 1, 1, 1, 1,
                      1, 0, 1, 1, 1,
                      0, 2, 0, 1, 1,
                      1, 1, 0, 1, 1,
                      2, 0, 0, 1, 1,
                      0, 0, 3, 0, 1,
                      0, 1, 2, 0, 1,
                      1, 0, 2, 0, 1,
                      0, 2, 1, 0, 1,
                      1, 1, 1, 0, 1,
                      2, 0, 1, 0, 1,
                      0, 3, 0, 0, 1,
                      1, 2, 0, 0, 1,
                      2, 1, 0, 0, 1,
                      3, 0, 0, 0, 1,
                      0, 0, 0, 4, 0,
                      0, 0, 1, 3, 0,
                      0, 1, 0, 3, 0,
                      1, 0, 0, 3, 0,
                      0, 0, 2, 2, 0,
                      0, 1, 1, 2, 0,
                      1, 0, 1, 2, 0,
                      0, 2, 0, 2, 0,
                      1, 1, 0, 2, 0,
                      2, 0, 0, 2, 0,
                      0, 0, 3, 1, 0,
                      0, 1, 2, 1, 0,
                      1, 0, 2, 1, 0,
                      0, 2, 1, 1, 0,
                      1, 1, 1, 1, 0,
                      2, 0, 1, 1, 0,
                      0, 3, 0, 1, 0,
                      1, 2, 0, 1, 0,
                      2, 1, 0, 1, 0,
                      3, 0, 0, 1, 0,
                      0, 0, 4, 0, 0,
                      0, 1, 3, 0, 0,
                      1, 0, 3, 0, 0,
                      0, 2, 2, 0, 0,
                      1, 1, 2, 0, 0,
                      2, 0, 2, 0, 0,
                      0, 3, 1, 0, 0,
                      1, 2, 1, 0, 0,
                      2, 1, 1, 0, 0,
                      3, 0, 1, 0, 0,
                      0, 4, 0, 0, 0,
                      1, 3, 0, 0, 0,
                      2, 2, 0, 0, 0,
                      3, 1, 0, 0, 0,
                      4, 0, 0, 0, 0;
    ComputeDegreeSum cDegree;
    MultiVariateBasis<Canonical> Basis(cDegree, 5, 4);
    BOOST_CHECK_EQUAL(Basis.getTensorFull(), TensorSumExpected);
}

BOOST_AUTO_TEST_CASE(testMultiVariateProdBasisConstructor)
{
    ComputeDegreeProd cDegree;
    Eigen::MatrixXi TensorProdExpected(352, 5);
    TensorProdExpected << 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 1,
                       0, 0, 0, 1, 0,
                       0, 0, 1, 0, 0,
                       0, 1, 0, 0, 0,
                       1, 0, 0, 0, 0,
                       1, 1, 0, 0, 0,
                       0, 1, 1, 0, 0,
                       1, 0, 1, 0, 0,
                       1, 1, 1, 0, 0,
                       0, 0, 1, 1, 0,
                       0, 1, 0, 1, 0,
                       1, 0, 0, 1, 0,
                       1, 1, 0, 1, 0,
                       0, 1, 1, 1, 0,
                       1, 0, 1, 1, 0,
                       1, 1, 1, 1, 0,
                       0, 0, 0, 1, 1,
                       0, 0, 1, 0, 1,
                       0, 1, 0, 0, 1,
                       1, 0, 0, 0, 1,
                       1, 1, 0, 0, 1,
                       0, 1, 1, 0, 1,
                       1, 0, 1, 0, 1,
                       1, 1, 1, 0, 1,
                       0, 0, 1, 1, 1,
                       0, 1, 0, 1, 1,
                       1, 0, 0, 1, 1,
                       1, 1, 0, 1, 1,
                       0, 1, 1, 1, 1,
                       1, 0, 1, 1, 1,
                       1, 1, 1, 1, 1,
                       0, 0, 0, 0, 2,
                       0, 0, 0, 1, 2,
                       0, 0, 1, 0, 2,
                       0, 1, 0, 0, 2,
                       1, 0, 0, 0, 2,
                       1, 1, 0, 0, 2,
                       0, 1, 1, 0, 2,
                       1, 0, 1, 0, 2,
                       1, 1, 1, 0, 2,
                       0, 0, 1, 1, 2,
                       0, 1, 0, 1, 2,
                       1, 0, 0, 1, 2,
                       1, 1, 0, 1, 2,
                       0, 1, 1, 1, 2,
                       1, 0, 1, 1, 2,
                       1, 1, 1, 1, 2,
                       0, 0, 0, 2, 0,
                       0, 0, 1, 2, 0,
                       0, 1, 0, 2, 0,
                       1, 0, 0, 2, 0,
                       1, 1, 0, 2, 0,
                       0, 1, 1, 2, 0,
                       1, 0, 1, 2, 0,
                       1, 1, 1, 2, 0,
                       0, 0, 2, 0, 0,
                       0, 1, 2, 0, 0,
                       1, 0, 2, 0, 0,
                       1, 1, 2, 0, 0,
                       0, 2, 0, 0, 0,
                       1, 2, 0, 0, 0,
                       2, 0, 0, 0, 0,
                       2, 1, 0, 0, 0,
                       0, 2, 1, 0, 0,
                       1, 2, 1, 0, 0,
                       2, 0, 1, 0, 0,
                       2, 1, 1, 0, 0,
                       0, 0, 2, 1, 0,
                       0, 1, 2, 1, 0,
                       1, 0, 2, 1, 0,
                       1, 1, 2, 1, 0,
                       0, 2, 0, 1, 0,
                       1, 2, 0, 1, 0,
                       2, 0, 0, 1, 0,
                       2, 1, 0, 1, 0,
                       0, 2, 1, 1, 0,
                       1, 2, 1, 1, 0,
                       2, 0, 1, 1, 0,
                       2, 1, 1, 1, 0,
                       0, 0, 0, 2, 1,
                       0, 0, 1, 2, 1,
                       0, 1, 0, 2, 1,
                       1, 0, 0, 2, 1,
                       1, 1, 0, 2, 1,
                       0, 1, 1, 2, 1,
                       1, 0, 1, 2, 1,
                       1, 1, 1, 2, 1,
                       0, 0, 2, 0, 1,
                       0, 1, 2, 0, 1,
                       1, 0, 2, 0, 1,
                       1, 1, 2, 0, 1,
                       0, 2, 0, 0, 1,
                       1, 2, 0, 0, 1,
                       2, 0, 0, 0, 1,
                       2, 1, 0, 0, 1,
                       0, 2, 1, 0, 1,
                       1, 2, 1, 0, 1,
                       2, 0, 1, 0, 1,
                       2, 1, 1, 0, 1,
                       0, 0, 2, 1, 1,
                       0, 1, 2, 1, 1,
                       1, 0, 2, 1, 1,
                       1, 1, 2, 1, 1,
                       0, 2, 0, 1, 1,
                       1, 2, 0, 1, 1,
                       2, 0, 0, 1, 1,
                       2, 1, 0, 1, 1,
                       0, 2, 1, 1, 1,
                       1, 2, 1, 1, 1,
                       2, 0, 1, 1, 1,
                       2, 1, 1, 1, 1,
                       0, 0, 0, 0, 3,
                       0, 0, 0, 1, 3,
                       0, 0, 1, 0, 3,
                       0, 1, 0, 0, 3,
                       1, 0, 0, 0, 3,
                       1, 1, 0, 0, 3,
                       0, 1, 1, 0, 3,
                       1, 0, 1, 0, 3,
                       1, 1, 1, 0, 3,
                       0, 0, 1, 1, 3,
                       0, 1, 0, 1, 3,
                       1, 0, 0, 1, 3,
                       1, 1, 0, 1, 3,
                       0, 1, 1, 1, 3,
                       1, 0, 1, 1, 3,
                       1, 1, 1, 1, 3,
                       0, 0, 0, 3, 0,
                       0, 0, 1, 3, 0,
                       0, 1, 0, 3, 0,
                       1, 0, 0, 3, 0,
                       1, 1, 0, 3, 0,
                       0, 1, 1, 3, 0,
                       1, 0, 1, 3, 0,
                       1, 1, 1, 3, 0,
                       0, 0, 3, 0, 0,
                       0, 1, 3, 0, 0,
                       1, 0, 3, 0, 0,
                       1, 1, 3, 0, 0,
                       0, 3, 0, 0, 0,
                       1, 3, 0, 0, 0,
                       3, 0, 0, 0, 0,
                       3, 1, 0, 0, 0,
                       0, 3, 1, 0, 0,
                       1, 3, 1, 0, 0,
                       3, 0, 1, 0, 0,
                       3, 1, 1, 0, 0,
                       0, 0, 3, 1, 0,
                       0, 1, 3, 1, 0,
                       1, 0, 3, 1, 0,
                       1, 1, 3, 1, 0,
                       0, 3, 0, 1, 0,
                       1, 3, 0, 1, 0,
                       3, 0, 0, 1, 0,
                       3, 1, 0, 1, 0,
                       0, 3, 1, 1, 0,
                       1, 3, 1, 1, 0,
                       3, 0, 1, 1, 0,
                       3, 1, 1, 1, 0,
                       0, 0, 0, 3, 1,
                       0, 0, 1, 3, 1,
                       0, 1, 0, 3, 1,
                       1, 0, 0, 3, 1,
                       1, 1, 0, 3, 1,
                       0, 1, 1, 3, 1,
                       1, 0, 1, 3, 1,
                       1, 1, 1, 3, 1,
                       0, 0, 3, 0, 1,
                       0, 1, 3, 0, 1,
                       1, 0, 3, 0, 1,
                       1, 1, 3, 0, 1,
                       0, 3, 0, 0, 1,
                       1, 3, 0, 0, 1,
                       3, 0, 0, 0, 1,
                       3, 1, 0, 0, 1,
                       0, 3, 1, 0, 1,
                       1, 3, 1, 0, 1,
                       3, 0, 1, 0, 1,
                       3, 1, 1, 0, 1,
                       0, 0, 3, 1, 1,
                       0, 1, 3, 1, 1,
                       1, 0, 3, 1, 1,
                       1, 1, 3, 1, 1,
                       0, 3, 0, 1, 1,
                       1, 3, 0, 1, 1,
                       3, 0, 0, 1, 1,
                       3, 1, 0, 1, 1,
                       0, 3, 1, 1, 1,
                       1, 3, 1, 1, 1,
                       3, 0, 1, 1, 1,
                       3, 1, 1, 1, 1,
                       0, 0, 0, 0, 4,
                       0, 0, 0, 1, 4,
                       0, 0, 1, 0, 4,
                       0, 1, 0, 0, 4,
                       1, 0, 0, 0, 4,
                       1, 1, 0, 0, 4,
                       0, 1, 1, 0, 4,
                       1, 0, 1, 0, 4,
                       1, 1, 1, 0, 4,
                       0, 0, 1, 1, 4,
                       0, 1, 0, 1, 4,
                       1, 0, 0, 1, 4,
                       1, 1, 0, 1, 4,
                       0, 1, 1, 1, 4,
                       1, 0, 1, 1, 4,
                       1, 1, 1, 1, 4,
                       0, 0, 0, 2, 2,
                       0, 0, 1, 2, 2,
                       0, 1, 0, 2, 2,
                       1, 0, 0, 2, 2,
                       1, 1, 0, 2, 2,
                       0, 1, 1, 2, 2,
                       1, 0, 1, 2, 2,
                       1, 1, 1, 2, 2,
                       0, 0, 2, 0, 2,
                       0, 1, 2, 0, 2,
                       1, 0, 2, 0, 2,
                       1, 1, 2, 0, 2,
                       0, 2, 0, 0, 2,
                       1, 2, 0, 0, 2,
                       2, 0, 0, 0, 2,
                       2, 1, 0, 0, 2,
                       0, 2, 1, 0, 2,
                       1, 2, 1, 0, 2,
                       2, 0, 1, 0, 2,
                       2, 1, 1, 0, 2,
                       0, 0, 2, 1, 2,
                       0, 1, 2, 1, 2,
                       1, 0, 2, 1, 2,
                       1, 1, 2, 1, 2,
                       0, 2, 0, 1, 2,
                       1, 2, 0, 1, 2,
                       2, 0, 0, 1, 2,
                       2, 1, 0, 1, 2,
                       0, 2, 1, 1, 2,
                       1, 2, 1, 1, 2,
                       2, 0, 1, 1, 2,
                       2, 1, 1, 1, 2,
                       0, 0, 0, 4, 0,
                       0, 0, 1, 4, 0,
                       0, 1, 0, 4, 0,
                       1, 0, 0, 4, 0,
                       1, 1, 0, 4, 0,
                       0, 1, 1, 4, 0,
                       1, 0, 1, 4, 0,
                       1, 1, 1, 4, 0,
                       0, 0, 2, 2, 0,
                       0, 1, 2, 2, 0,
                       1, 0, 2, 2, 0,
                       1, 1, 2, 2, 0,
                       0, 2, 0, 2, 0,
                       1, 2, 0, 2, 0,
                       2, 0, 0, 2, 0,
                       2, 1, 0, 2, 0,
                       0, 2, 1, 2, 0,
                       1, 2, 1, 2, 0,
                       2, 0, 1, 2, 0,
                       2, 1, 1, 2, 0,
                       0, 0, 4, 0, 0,
                       0, 1, 4, 0, 0,
                       1, 0, 4, 0, 0,
                       1, 1, 4, 0, 0,
                       0, 2, 2, 0, 0,
                       1, 2, 2, 0, 0,
                       2, 0, 2, 0, 0,
                       2, 1, 2, 0, 0,
                       0, 4, 0, 0, 0,
                       1, 4, 0, 0, 0,
                       2, 2, 0, 0, 0,
                       4, 0, 0, 0, 0,
                       4, 1, 0, 0, 0,
                       0, 4, 1, 0, 0,
                       1, 4, 1, 0, 0,
                       2, 2, 1, 0, 0,
                       4, 0, 1, 0, 0,
                       4, 1, 1, 0, 0,
                       0, 0, 4, 1, 0,
                       0, 1, 4, 1, 0,
                       1, 0, 4, 1, 0,
                       1, 1, 4, 1, 0,
                       0, 2, 2, 1, 0,
                       1, 2, 2, 1, 0,
                       2, 0, 2, 1, 0,
                       2, 1, 2, 1, 0,
                       0, 4, 0, 1, 0,
                       1, 4, 0, 1, 0,
                       2, 2, 0, 1, 0,
                       4, 0, 0, 1, 0,
                       4, 1, 0, 1, 0,
                       0, 4, 1, 1, 0,
                       1, 4, 1, 1, 0,
                       2, 2, 1, 1, 0,
                       4, 0, 1, 1, 0,
                       4, 1, 1, 1, 0,
                       0, 0, 0, 4, 1,
                       0, 0, 1, 4, 1,
                       0, 1, 0, 4, 1,
                       1, 0, 0, 4, 1,
                       1, 1, 0, 4, 1,
                       0, 1, 1, 4, 1,
                       1, 0, 1, 4, 1,
                       1, 1, 1, 4, 1,
                       0, 0, 2, 2, 1,
                       0, 1, 2, 2, 1,
                       1, 0, 2, 2, 1,
                       1, 1, 2, 2, 1,
                       0, 2, 0, 2, 1,
                       1, 2, 0, 2, 1,
                       2, 0, 0, 2, 1,
                       2, 1, 0, 2, 1,
                       0, 2, 1, 2, 1,
                       1, 2, 1, 2, 1,
                       2, 0, 1, 2, 1,
                       2, 1, 1, 2, 1,
                       0, 0, 4, 0, 1,
                       0, 1, 4, 0, 1,
                       1, 0, 4, 0, 1,
                       1, 1, 4, 0, 1,
                       0, 2, 2, 0, 1,
                       1, 2, 2, 0, 1,
                       2, 0, 2, 0, 1,
                       2, 1, 2, 0, 1,
                       0, 4, 0, 0, 1,
                       1, 4, 0, 0, 1,
                       2, 2, 0, 0, 1,
                       4, 0, 0, 0, 1,
                       4, 1, 0, 0, 1,
                       0, 4, 1, 0, 1,
                       1, 4, 1, 0, 1,
                       2, 2, 1, 0, 1,
                       4, 0, 1, 0, 1,
                       4, 1, 1, 0, 1,
                       0, 0, 4, 1, 1,
                       0, 1, 4, 1, 1,
                       1, 0, 4, 1, 1,
                       1, 1, 4, 1, 1,
                       0, 2, 2, 1, 1,
                       1, 2, 2, 1, 1,
                       2, 0, 2, 1, 1,
                       2, 1, 2, 1, 1,
                       0, 4, 0, 1, 1,
                       1, 4, 0, 1, 1,
                       2, 2, 0, 1, 1,
                       4, 0, 0, 1, 1,
                       4, 1, 0, 1, 1,
                       0, 4, 1, 1, 1,
                       1, 4, 1, 1, 1,
                       2, 2, 1, 1, 1,
                       4, 0, 1, 1, 1,
                       4, 1, 1, 1, 1;
    MultiVariateBasis<Canonical> Basis(cDegree, 5, 4);
    BOOST_CHECK_EQUAL(Basis.getTensorFull(), TensorProdExpected);
}

BOOST_AUTO_TEST_CASE(testMultiVariateHyperbolicBasisConstructor)
{
    ComputeDegreeHyperbolic cDegree(0.7);
    Eigen::MatrixXi TensorHyperbolicExpected(51, 5);
    TensorHyperbolicExpected << 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 1,
                             0, 0, 0, 1, 0,
                             0, 0, 1, 0, 0,
                             0, 1, 0, 0, 0,
                             1, 0, 0, 0, 0,
                             0, 0, 0, 0, 2,
                             0, 0, 0, 1, 1,
                             0, 0, 1, 0, 1,
                             0, 1, 0, 0, 1,
                             1, 0, 0, 0, 1,
                             0, 0, 0, 2, 0,
                             0, 0, 1, 1, 0,
                             0, 1, 0, 1, 0,
                             1, 0, 0, 1, 0,
                             0, 0, 2, 0, 0,
                             0, 1, 1, 0, 0,
                             1, 0, 1, 0, 0,
                             0, 2, 0, 0, 0,
                             1, 1, 0, 0, 0,
                             2, 0, 0, 0, 0,
                             0, 0, 0, 0, 3,
                             0, 0, 0, 1, 2,
                             0, 0, 1, 0, 2,
                             0, 1, 0, 0, 2,
                             1, 0, 0, 0, 2,
                             0, 0, 0, 2, 1,
                             0, 0, 2, 0, 1,
                             0, 2, 0, 0, 1,
                             2, 0, 0, 0, 1,
                             0, 0, 0, 3, 0,
                             0, 0, 1, 2, 0,
                             0, 1, 0, 2, 0,
                             1, 0, 0, 2, 0,
                             0, 0, 2, 1, 0,
                             0, 2, 0, 1, 0,
                             2, 0, 0, 1, 0,
                             0, 0, 3, 0, 0,
                             0, 1, 2, 0, 0,
                             1, 0, 2, 0, 0,
                             0, 2, 1, 0, 0,
                             2, 0, 1, 0, 0,
                             0, 3, 0, 0, 0,
                             1, 2, 0, 0, 0,
                             2, 1, 0, 0, 0,
                             3, 0, 0, 0, 0,
                             0, 0, 0, 0, 4,
                             0, 0, 0, 4, 0,
                             0, 0, 4, 0, 0,
                             0, 4, 0, 0, 0,
                             4, 0, 0, 0, 0;
    MultiVariateBasis<Canonical> Basis(cDegree, 5, 4);
    BOOST_CHECK_EQUAL(Basis.getTensorFull(), TensorHyperbolicExpected);
}

// Test regression functions
static double fToRegressLog(const Eigen::MatrixXd &p_x)
{
    double normX = p_x.squaredNorm();
    normX *= normX;
    return log1p(normX);
}

/**
 * \brief A polynomial function, which can be exactly recovered using a
 * MultiVariateBasis
 *
 * \param p_x a vector representing the coordinates of a point
 */
static double fToRegressPol(const Eigen::MatrixXd &p_x)
{
    double aux = 0.;
    // Using .size() enables us to deal with row or column vectors.
    for (int i = 0; i < p_x.size(); i++)
    {
        double x_i = p_x(i);
        aux += (i + 1.) * x_i * x_i;
    }
    return aux;
}

/**
 * \brief Compute the infinity norm of the regression error
 *
 * \param p_f the function to regress
 * \param p_Basis the MultiVariateBasis to use
 * \param p_dim the number of variates
 * \param p_nPoints the number of points of the regression grid
 * \param p_a
 * \param p_b the grid is [p_a,p_b]^{p_dim}
 */
template <class T> double
multiVariateRegression(std::function<double(const Eigen::ArrayXXd &)> p_f, MultiVariateBasis<T> &p_basis, int p_dim, int p_nPoints, double p_a, double p_b)
{
    // Draw points in [a, b]^dim and compute f on it
    Eigen::ArrayXXd grid(p_dim, p_nPoints);
    Eigen::ArrayXd fValues(p_nPoints);
    boost::mt19937 generator(time(0));
    boost::random::uniform_real_distribution< > uniformDistribution(p_a, p_b);
    boost::variate_generator<boost::mt19937 &, boost::random::uniform_real_distribution<> > uniform_random(generator, uniformDistribution);
    for (int i = 0; i < grid.size(); i++)
    {
        grid(i) = uniform_random();
    }
    for (int i = 0; i < p_nPoints; i++)
    {
        fValues(i) = p_f(grid.col(i));
    }

    // Solve the least square problem
    Eigen::ArrayXd basisCoordinates;
    p_basis.fitLeastSquare(basisCoordinates, grid, fValues);
    // Compute the error
    double error = 0.;
    for (int i = 0; i < p_nPoints; i++)
    {
        const double y = p_basis.eval(grid.col(i), basisCoordinates);
        error = max(error, abs(y - fValues(i)));
    }
    return error;
}

BOOST_AUTO_TEST_CASE(testMultiVariateBasisRegression1D)
{
    double a = -2., b = 3.;
    int dim = 1, nPoints = 10;
    double error;
    {
        int deg = 4;
        ComputeDegreeSum cDegree;
        MultiVariateBasis<Hermite> basis(cDegree, dim, deg);
        error = multiVariateRegression<Hermite>(fToRegressPol, basis, dim, nPoints, a, b);
        BOOST_CHECK_SMALL(error, 1E-10);
    }
    {
        int deg = 10;
        ComputeDegreeSum cDegree;
        MultiVariateBasis<Hermite> basis(cDegree, dim, deg);
        error = multiVariateRegression<Hermite>(fToRegressLog, basis, dim, nPoints, a, b);
        BOOST_CHECK_SMALL(error, 2E-3);
    }
}

BOOST_AUTO_TEST_CASE(testMultiVariateBasisRegressionExact)
{
    double a = -2., b = 3.;
    int dim = 4, deg = 3, nPoints = 10;
    double error;
    ComputeDegreeSum cDegree;
    MultiVariateBasis<Hermite> basis(cDegree, dim, deg);
    error = multiVariateRegression<Hermite>(fToRegressPol, basis, dim, nPoints, a, b);
    BOOST_CHECK_SMALL(error, 1E-10);
    error = multiVariateRegression<Hermite>(fToRegressLog, basis, dim, nPoints, a, b);
    BOOST_CHECK_SMALL(error, 2E-3);
}

BOOST_AUTO_TEST_CASE(testMultiVariateBasisRegressionReducedExact)
{
    double a = -2., b = 3.;
    int dim = 4, deg = 3, nPoints = 10;
    double error;
    ComputeDegreeSum cDegree;
    MultiVariateBasis<Hermite> basis(cDegree, dim, deg);
    Eigen::ArrayXd center(dim);
    center.setConstant(0.5);
    Eigen::ArrayXd scale(dim);
    scale.setConstant(5);
    basis.setReduced(center, scale);
    error = multiVariateRegression<Hermite>(fToRegressPol, basis, dim, nPoints, a, b);
    BOOST_CHECK_SMALL(error, 1E-10);
    error = multiVariateRegression<Hermite>(fToRegressLog, basis, dim, nPoints, a, b);
    BOOST_CHECK_SMALL(error, 2E-3);
}

BOOST_AUTO_TEST_CASE(testMultiVariateBasisRegressionDomainExact)
{
    double a = -2., b = 3.;
    int dim = 4, deg = 3, nPoints = 10;
    double error;
    ComputeDegreeSum cDegree;
    MultiVariateBasis<Hermite> basis(cDegree, dim, deg);
    Eigen::ArrayXd lower(dim);
    lower.setConstant(a);
    Eigen::ArrayXd upper(dim);
    upper.setConstant(b);
    basis.setDomain(lower, upper);
    error = multiVariateRegression<Hermite>(fToRegressPol, basis, dim, nPoints, a, b);
    BOOST_CHECK_SMALL(error, 1E-10);
    error = multiVariateRegression<Hermite>(fToRegressLog, basis, dim, nPoints, a, b);
    BOOST_CHECK_SMALL(error, 2E-3);
}


BOOST_AUTO_TEST_CASE(testMultiVariateBasisDerivatives)
{
    // double a = -2., b = 3.;
    double a = 0., b = 1.;
    int dim = 3, deg = 3, nPoints = 10;
    ComputeDegreeSum cDegree;
    MultiVariateBasis<Hermite> basis(cDegree, dim, deg);
    Eigen::ArrayXd lower(dim);
    lower.setConstant(a);
    Eigen::ArrayXd upper(dim);
    upper.setConstant(b);
    // basis.setDomain(lower, upper);
    Eigen::ArrayXd coefficients(basis.getNumberOfFunctions());
    coefficients.setRandom();

    Eigen::ArrayXXd grid(dim, nPoints);
    Eigen::ArrayXd fValues(nPoints);
    boost::mt19937 generator(time(0));
    boost::random::uniform_real_distribution< > uniformDistribution(a, b);
    boost::variate_generator<boost::mt19937 &, boost::random::uniform_real_distribution<> > uniform_random(generator, uniformDistribution);
    for (int i = 0; i < grid.size(); i++)
    {
        grid(i) = uniform_random();
    }

    double fx;
    Eigen::ArrayXd grad, gradBis;
    Eigen::ArrayXXd hes, hesBis;
    for (int i = 0; i < nPoints; i++)
    {
        basis.evalGradient(grid.col(i), coefficients, grad);
        basis.evalHessian(grid.col(i), coefficients, hes);
        basis.evalDerivatives(grid.col(i), coefficients, fx, gradBis, hesBis);
        BOOST_CHECK_SMALL((grad - gradBis).matrix().lpNorm<Eigen::Infinity>(), 2e-3);
        BOOST_CHECK_SMALL((hes - hesBis).matrix().lpNorm<Eigen::Infinity>(), 2e-3);
    }
}
