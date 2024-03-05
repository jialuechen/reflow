// Copyright (C) 2016 EDF
// All Rights Reserved
// This code is published under the GNU Lesser General Public License (GNU LGPL)
#define BOOST_TEST_MODULE testGridAndRegressedValue
#define BOOST_TEST_DYN_LINK
#include <functional>
#include <memory>
#include <boost/test/unit_test.hpp>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <boost/random.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include "geners/BinaryFileArchive.hh"
#include "geners/Record.hh"
#include "geners/Reference.hh"
#include "libflow/core/grids/GridIterator.h"
#include "libflow/core/grids/RegularSpaceGrid.h"
#include "libflow/core/grids/RegularSpaceGridGeners.h"
#include "libflow/core/grids/RegularLegendreGrid.h"
#include "libflow/core/grids/RegularLegendreGridGeners.h"
#include "libflow/core/grids/SparseSpaceGridBound.h"
#include "libflow/core/grids/SparseSpaceGridBoundGeners.h"
#include "libflow/regression/GridAndRegressedValue.h"
#include "libflow/regression/GridAndRegressedValueGeners.h"
#include "libflow/regression/LocalLinearRegression.h"
#include "libflow/regression/LocalLinearRegressionGeners.h"
#include "libflow/core/utils/Polynomials1D.h"
#include "libflow/regression/GlobalRegression.h"
#include "libflow/regression/GlobalRegressionGeners.h"

using namespace std;
using namespace Eigen;
using namespace gs;
using namespace libflow;


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

double accuracyNearEqual = 0.5 ;
double accuracyEqual = 1e-10 ;

/// \brief To test grids and regression
/// \param p_grid  grid
/// \param p_reg   regression object
/// \param p_nDimReg dimension for regression
/// \param p_function function to store
/// \param p_rho   correl between  a dimension and the previous one
void test(shared_ptr<SpaceGrid> &p_grid,
          shared_ptr<BaseRegression> &p_reg,
          const int &p_nbSimul,
          const int &p_nDimReg,
          const function< double(const ArrayXd &, const ArrayXd &)>    &p_function,
          const double &p_rho)
{
    // get dimension of the grid
    int nDimGrid = p_grid->getDimension();
    ArrayXd xGrid(nDimGrid);
    // get dimension for regression
    ArrayXXd  xReg(p_nDimReg, p_nbSimul);

    // generator Mersene Twister
    boost::mt19937 generator;
    boost::uniform_real<> uni_dist(0, 1);
    boost::variate_generator<boost::mt19937 &, boost::uniform_real<> > uni_random(generator, uni_dist) ;
    // fill in Xreg
    double rhoB = sqrt(1 - p_rho * p_rho);
    for (int is = 0; is < p_nbSimul; ++is)
        xReg(0, is) = uni_random();
    for (int id = 1 ; id < p_nDimReg; ++id)
        for (int is = 0; is < p_nbSimul; ++is)
            xReg(id, is) = p_rho * xReg(id - 1, is) + rhoB * uni_random();

    bool bZeroDate =  false ; /// not regression at 0
    p_reg->updateSimulations(bZeroDate, xReg);

    // to store values
    ArrayXXd values(p_nbSimul, p_grid->getNbPoints());
    // iterate
    shared_ptr<GridIterator> gridIterator = p_grid->getGridIterator();
    while (gridIterator->isValid())
    {
        ArrayXd  coord = gridIterator->getCoordinate();
        for (int is = 0; is < p_nbSimul; ++is)
            values(is, gridIterator->getCount()) = p_function(coord, xReg.col(is));
        gridIterator->next();
    }

    /// create the state
    GridAndRegressedValue pointRep(p_grid, p_reg, values);

    // test in 0
    ArrayXd  coordGrid = ArrayXd::Zero(nDimGrid);
    ArrayXd  coordReg(p_nDimReg);
    for (int id = 0 ; id < p_nDimReg ; ++id)
        coordReg(id) = uni_random();
    double realValue = p_function(coordGrid, coordReg);
    double interp = pointRep.getValue(coordGrid, coordReg);

    cout << " realValue " << realValue << " interp " << interp << endl ;
    BOOST_CHECK_CLOSE(realValue, interp, accuracyNearEqual);

    // The archive to use
    {
        // default non compression
        BinaryFileArchive ar("archiveGRV", "w");
        ar << Record(pointRep, "First rep", "Top") ;
        ar.flush();
    }

    {
        BinaryFileArchive ar("archiveGRV", "r");
        GridAndRegressedValue pointRepNew ;
        Reference<GridAndRegressedValue>(ar, "First rep", "Top").restore(0, &pointRepNew);

        double interpDeser = pointRepNew.getValue(coordGrid, coordReg);

        BOOST_CHECK_CLOSE(interpDeser, interp, accuracyNearEqual);
    }
}

// simple 1D
BOOST_AUTO_TEST_CASE(GridAndRegressedValueLoc1)
{
    cout << " 1D  LOC" << endl ;
    cout << "********" << endl ;
    // Regressor
    //**********
    int nDimReg = 1 ;
    int nMesh = 10 ;
    // create the mesh
    ArrayXi  nbMesh = ArrayXi::Constant(nDimReg, nMesh);
    // regressor
    shared_ptr< BaseRegression > reg = make_shared<LocalLinearRegression>(nbMesh);
    // Grid
    //*******
    int nDimGrid = 1 ;
    ArrayXd lowValues = ArrayXd::Constant(nDimGrid, -1);
    ArrayXd step = ArrayXd::Constant(nDimGrid, 0.09);
    ArrayXi  nbStep = ArrayXi::Constant(nDimGrid, 20);
    // grid
    shared_ptr<SpaceGrid> grid  = make_shared< RegularSpaceGrid>(lowValues, step, nbStep);
    // Function
    //*********
    auto func([](const ArrayXd & p_x, const ArrayXd & p_y)
    {
        return (1 + p_x.sum()) * pow(p_y.sum() + 1., 2.);
    });
    function< double(const ArrayXd &, const ArrayXd &)> cFunc(std::cref(func));

    int nbSimul = 1000;
    // Test
    test(grid, reg, nbSimul, nDimReg, cFunc, 0);
}

// 2D
BOOST_AUTO_TEST_CASE(GridAndRegressedValueLoc2)
{
    cout << " 2D  LOC" << endl ;
    cout << "********" << endl ;
    // Regressor
    //**********
    int nDimReg = 2 ;
    int nMesh = 10 ;
    // create the mesh
    ArrayXi  nbMesh = ArrayXi::Constant(nDimReg, nMesh);
    // regressor
    shared_ptr< BaseRegression > reg = make_shared<LocalLinearRegression>(nbMesh);
    // Grid
    //*******
    int nDimGrid = 2 ;
    ArrayXd lowValues = ArrayXd::Constant(nDimGrid, -1);
    ArrayXd step = ArrayXd::Constant(nDimGrid, 0.19);
    ArrayXi  nbStep = ArrayXi::Constant(nDimGrid, 10);
    // grid
    shared_ptr<SpaceGrid> grid  = make_shared< RegularSpaceGrid>(lowValues, step, nbStep);
    // Function
    //*********
    auto func([](const ArrayXd & p_x, const ArrayXd & p_y)
    {
        return (1 + p_x.sum()) * pow(p_y.sum() + 1., 2.);
    });
    function< double(const ArrayXd &, const ArrayXd &)> cFunc(std::cref(func));

    int nbSimul = 40000;
    // Test
    test(grid, reg, nbSimul, nDimReg, cFunc, 0);
}

// 2D correl 1
BOOST_AUTO_TEST_CASE(GridAndRegressedValueLoc3)
{
    cout << " 2D  LOC CORREL 0.4" << endl ;
    cout << "********************" << endl ;
    // Regressor
    //**********
    int nDimReg = 2 ;
    int nMesh = 10 ;
    double rho = 0.4;
    // create the mesh
    ArrayXi  nbMesh(2) ;
    nbMesh(0) = nMesh;
    nbMesh(1) = nMesh;

    // regressor
    shared_ptr< BaseRegression > reg = make_shared<LocalLinearRegression>(nbMesh);
    // Grid
    //*******
    int nDimGrid = 2 ;
    ArrayXd lowValues = ArrayXd::Constant(nDimGrid, -1);
    ArrayXd step = ArrayXd::Constant(nDimGrid, 0.19);
    ArrayXi  nbStep = ArrayXi::Constant(nDimGrid, 10);
    // grid
    shared_ptr<SpaceGrid> grid  = make_shared< RegularSpaceGrid>(lowValues, step, nbStep);
    // Function
    //*********
    auto func([](const ArrayXd & p_x, const ArrayXd & p_y)
    {
        return (1 + p_x.sum()) * pow(p_y.sum() + 1., 2.);
    });
    function< double(const ArrayXd &, const ArrayXd &)> cFunc(std::cref(func));

    int nbSimul = 60000;
    // Test
    test(grid, reg, nbSimul, nDimReg, cFunc, rho);
}

// 2D correl 2
BOOST_AUTO_TEST_CASE(GridAndRegressedValueLoc4)
{
    cout << " 2D  LOC CORREL 0.9" << endl ;
    cout << "********************" << endl ;

    // Regressor
    //**********
    int nDimReg = 2 ;
    int nMesh = 10 ;
    double rho = 0.9;
    // create the mesh
    ArrayXi  nbMesh(2) ;
    nbMesh(0) = nMesh;
    nbMesh(1) = nMesh;
    // regressor
    shared_ptr< BaseRegression > reg = make_shared<LocalLinearRegression>(nbMesh);
    // Grid
    //*******
    int nDimGrid = 2 ;
    ArrayXd lowValues = ArrayXd::Constant(nDimGrid, -1);
    ArrayXd step = ArrayXd::Constant(nDimGrid, 0.19);
    ArrayXi  nbStep = ArrayXi::Constant(nDimGrid, 10);
    // grid
    shared_ptr<SpaceGrid> grid  = make_shared< RegularSpaceGrid>(lowValues, step, nbStep);
    // Function
    //*********
    auto func([](const ArrayXd & p_x, const ArrayXd & p_y)
    {
        return (1 + p_x.sum()) * pow(p_y.sum() + 1., 2.);
    });
    function< double(const ArrayXd &, const ArrayXd &)> cFunc(std::cref(func));

    int nbSimul = 10000;
    // Test
    test(grid, reg, nbSimul, nDimReg, cFunc, rho);
}

// 2D correl 2
BOOST_AUTO_TEST_CASE(GridAndRegressedValueLoc5)
{
    cout << " 2D  LOC CORREL 0.9 with rotation" << endl ;
    cout << "*********************************" << endl ;

    // Regressor
    //**********
    int nDimReg = 2 ;
    int nMesh = 10 ;
    double rho = 0.9;
    // create the mesh
    ArrayXi  nbMesh(2) ;
    nbMesh(0) = nMesh;
    nbMesh(1) = nMesh / 3;
    // regressor
    shared_ptr< BaseRegression > reg = make_shared<LocalLinearRegression>(nbMesh, true);
    // Grid
    //*******
    int nDimGrid = 2 ;
    ArrayXd lowValues = ArrayXd::Constant(nDimGrid, -1);
    ArrayXd step = ArrayXd::Constant(nDimGrid, 0.19);
    ArrayXi  nbStep = ArrayXi::Constant(nDimGrid, 10);
    // grid
    shared_ptr<SpaceGrid> grid  = make_shared< RegularSpaceGrid>(lowValues, step, nbStep);
    // Function
    //*********
    auto func([](const ArrayXd & p_x, const ArrayXd & p_y)
    {
        return (1 + p_x.sum()) * pow(p_y.sum() + 1., 2.);
    });
    function< double(const ArrayXd &, const ArrayXd &)> cFunc(std::cref(func));

    int nbSimul = 10000;
    // Test
    test(grid, reg, nbSimul, nDimReg, cFunc, rho);
}


// 1D Global regression
BOOST_AUTO_TEST_CASE(GridAndRegressedValueGlob1)
{
    cout << " 1D  GLOB" << endl ;
    cout << "**********" << endl ;

    // Regressor
    //**********
    int nDimReg = 1 ;
    int degree = 3;
    // regressor
    shared_ptr< BaseRegression > reg = make_shared<GlobalRegression<Hermite> >(degree, nDimReg);
    // Grid
    //*******
    int nDimGrid = 2 ;
    ArrayXd lowValues = ArrayXd::Constant(nDimGrid, -1);
    ArrayXd step = ArrayXd::Constant(nDimGrid, 0.19);
    ArrayXi  nbStep = ArrayXi::Constant(nDimGrid, 10);
    ArrayXi  npol = ArrayXi::Constant(nDimGrid, 2);

    // grid
    shared_ptr<SpaceGrid> grid  = make_shared< RegularLegendreGrid>(lowValues, step, nbStep, npol);
    // Function
    //*********
    auto func([](const ArrayXd & p_x, const ArrayXd & p_y)
    {
        return (1 + p_x.sum()) * pow(p_y.sum() + 1., 2.);
    });
    function< double(const ArrayXd &, const ArrayXd &)> cFunc(std::cref(func));

    int nbSimul = 10000;
    // Test
    test(grid, reg, nbSimul, nDimReg, cFunc, 0.);
}

// 2D Global regression with correl
BOOST_AUTO_TEST_CASE(GridAndRegressedValueGlob2)
{
    cout << " 2D  GLOB CORREL 0.95" << endl ;
    cout << "********************" << endl ;
    // Regressor
    //**********
    int nDimReg = 2 ;
    int degree = 3;
    double rho = 0.7;
    // regressor
    shared_ptr< BaseRegression > reg = make_shared<GlobalRegression<Hermite> >(degree, nDimReg);
    // Grid
    //*******
    int nDimGrid = 2 ;
    ArrayXd lowValues = ArrayXd::Constant(nDimGrid, -1);
    ArrayXd step = ArrayXd::Constant(nDimGrid, 0.19);
    ArrayXi  nbStep = ArrayXi::Constant(nDimGrid, 10);
    ArrayXi  npol = ArrayXi::Constant(nDimGrid, 2);

    // grid
    shared_ptr<SpaceGrid> grid  = make_shared< RegularLegendreGrid>(lowValues, step, nbStep, npol);
    // Function
    //*********
    auto func([](const ArrayXd & p_x, const ArrayXd & p_y)
    {
        return (1 + p_x.sum()) * pow(p_y.sum() + 1., 2.);
    });
    function< double(const ArrayXd &, const ArrayXd &)> cFunc(std::cref(func));

    int nbSimul = 10000;
    // Test
    test(grid, reg, nbSimul, nDimReg, cFunc, rho);
}


// 2D Sparse grid, local regression
BOOST_AUTO_TEST_CASE(GridAndRegressedValueSparse1)
{
    cout << " SPARSE LOC      " << endl ;
    cout << "********************" << endl ;
    // Regressor
    //**********
    int nDimReg = 2 ;
    int nMesh = 10 ;
    // create the mesh
    ArrayXi  nbMesh = ArrayXi::Constant(nDimReg, nMesh);
    // regressor
    shared_ptr< BaseRegression > reg = make_shared<LocalLinearRegression>(nbMesh);
    // Grid
    //*******
    int nDimGrid = 2 ;
    ArrayXd lowValues = ArrayXd::Constant(nDimGrid, -1);
    ArrayXd sizeDomain = ArrayXd::Constant(nDimGrid, 1.9);
    ArrayXi  nbStep = ArrayXi::Constant(nDimGrid, 10);
    ArrayXd  weight = ArrayXd::Constant(nDimGrid, 1.);
    int level = 6;
    int degree = 2 ;
    // grid
    shared_ptr<SpaceGrid> grid  = make_shared< SparseSpaceGridBound>(lowValues, sizeDomain, level, weight, degree);
    // Function
    //*********
    auto func([](const ArrayXd & p_x, const ArrayXd & p_y)
    {
        return (1 + p_x.sum()) * pow(p_y.sum() + 1., 2.);
    });
    function< double(const ArrayXd &, const ArrayXd &)> cFunc(std::cref(func));

    int nbSimul = 1000;
    // Test
    test(grid, reg, nbSimul, nDimReg, cFunc, 0.);
}

// 2D Sparse grid, local regression
BOOST_AUTO_TEST_CASE(GridAndRegressedValueSparse2)
{
    cout << " SPARSE LOC CORREL 0.9      " << endl ;
    cout << "*****************************" << endl ;
    // Regressor
    //**********
    int nDimReg = 2 ;
    int nMesh = 10 ;
    double rho = 0.9;
    // create the mesh
    // create the mesh
    ArrayXi  nbMesh(2) ;
    nbMesh(0) = nMesh;
    nbMesh(1) = nMesh / 3;
    // regressor
    shared_ptr< BaseRegression > reg = make_shared<LocalLinearRegression>(nbMesh);
    // Grid
    //*******
    int nDimGrid = 2 ;
    ArrayXd lowValues = ArrayXd::Constant(nDimGrid, -1);
    ArrayXd sizeDomain = ArrayXd::Constant(nDimGrid, 1.9);
    ArrayXi  nbStep = ArrayXi::Constant(nDimGrid, 10);
    ArrayXd  weight = ArrayXd::Constant(nDimGrid, 1.);
    int level = 6;
    int degree = 2 ;
    // grid
    shared_ptr<SpaceGrid> grid  = make_shared< SparseSpaceGridBound>(lowValues, sizeDomain, level, weight, degree);
    // Function
    //*********
    auto func([](const ArrayXd & p_x, const ArrayXd & p_y)
    {
        return (1 + p_x.sum()) * pow(p_y.sum() + 1., 2.);
    });
    function< double(const ArrayXd &, const ArrayXd &)> cFunc(std::cref(func));


    int nbSimul = 1000;
    // Test
    test(grid, reg, nbSimul, nDimReg, cFunc, rho);
}

