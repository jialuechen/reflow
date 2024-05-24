
#include <math.h>
#include <functional>
#include <memory>
#include <fstream>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "reflow/core/grids/RegularLegendreGrid.h"
#include "reflow/core/grids/SparseSpaceGridBound.h"
#include "test/c++/tools/semilagrangien/OptimizeSLEmissive.h"
#include "test/c++/tools/semilagrangien/semiLagrangTimeNonEmissive.h"
#include "test/c++/tools/semilagrangien/semiLagrangTimeNonEmissiveSparse.h"
#include "test/c++/tools/semilagrangien/simuSLNonEmissive.h"
#include "test/c++/tools/semilagrangien/simuSLNonEmissiveSparse.h"


using namespace std;
using namespace Eigen ;
using namespace reflow;

#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif

// define Pi function
class PiFunc
{
public :

    PiFunc() {}

    double operator()(const double &m_D, const double &m_L) const
    {
        double prix  = pow(1 + m_D, 2.) - m_L;
        double cost  = 1. ;
        return  prix * m_D -  cost * max(m_D - m_L, 0.);
    }
};


class CBar
{
    double m_beta;
    double m_cInf;
    double m_c0;

public :

    CBar(const double p_beta,  const double &p_cInf, const double &p_c0): m_beta(p_beta), m_cInf(p_cInf), m_c0(p_c0)
    {}

    double operator()(const double &p_l, const double   &p_L)
    {
        return m_beta * (m_cInf  + (m_c0 - m_cInf) * exp(-m_beta * p_L)) * (1 + p_l) * p_l;
    }
};

class Boundary
{
    function<double(const double &, const double &) > m_piFunc;
    double  m_s;
    double m_alpha ;
    double m_lambda ;
    function<double(const int &, const ArrayXd &)>  m_final ;

public :

    Boundary(const function<double(const double &, const double &) >   &p_piFunc, double p_s, double p_alpha, double p_lambda,
             const function<double(const int &, const ArrayXd &)>   &p_final) : m_piFunc(p_piFunc), m_s(p_s), m_alpha(p_alpha),
        m_lambda(p_lambda), m_final(p_final) {}

    double operator()(const double &p_timeToMat, const int &p_iReg, const ArrayXd &p_point)
    {
        if (p_iReg == 0)
        {
            return p_timeToMat * (m_piFunc(p_point(0), p_point(2)) + m_s * pow(p_point(2), 1. - m_alpha) - m_lambda * max(p_point(0) - p_point(2), 0.));
        }
        else
            return  m_final(1, p_point)  ;
    }
}
;

// Implement  the HJB resolution for the  problem  with Semi Lagrangian
int main(int argc, char *argv[])
{
#if defined   __linux
    enable_abort_on_floating_point_exception();
#endif
    boost::mpi::environment env(argc, argv);

    boost::mpi::communicator world;

    // gain function
    function< double(const double &, const double &)> funcPi = PiFunc();
    // cost function for investment
    double beta = 0.5 ;
    double cInf = 1.;
    double c0 = 2. ;
    function< double(const double &, const double &)> funcCost = CBar(beta, cInf, c0);

    // demand
    double m = 1.4;
    double alpha = 1. ;
    double sig = 0.1;

    // s parameter for subvention
    double s = 0.2 ;

    // CO2 price parameters
    double lambda = 1 ;
    double H = 1 ;

    // final value function
    auto fSimu([H, lambda](const int &p_iReg, const ArrayXd & x)
    {
        if (p_iReg == 0)
            return  0. ;
        else
            return ((x[1] >= H) ? lambda : 0.);
    });

    function<double(const int &, const ArrayXd &)>  finalFunctionValue(cref(fSimu));

    // boundary
    function<double(const double &, const int &, const ArrayXd &)>  boundary = Boundary(funcPi, s, alpha, lambda, finalFunctionValue);
    // grid
    ArrayXd lowValues(3);
    lowValues << 0.4, 0.0, 0.;
    ArrayXd step(3);
    step <<  0.1, 0.1, 0.1 ;
    ArrayXi nstep(3);
    nstep << 40, 20, 20;
    ArrayXi npoly(3);
    npoly(0) = 2;
    npoly(1) = 1;
    npoly(2) = 1;
    shared_ptr<reflow::FullGrid>  grid = make_shared<RegularLegendreGrid>(lowValues, step, nstep, npoly);
    // time step
    int ndt = 200 ;
    double dt = 1. / ndt ;
    // max investment intensity
    double lMax = 2.;
    double lStep = 0.05 ;
    // Optimizer with full grids
    //**************************
    shared_ptr<OptimizeSLEmissive> optimizer = make_shared<OptimizeSLEmissive>(alpha, m, sig, funcPi, funcCost, s, dt, lMax, lStep, grid->getExtremeValues());
    // optimization
    string fileToDump = "NonEmissivSLDump";
    semiLagrangTimeNonEmissive(grid, optimizer,  finalFunctionValue, boundary, dt, ndt,  fileToDump, world);

    // simulation with full grids
    //***************************
    int nbSimul = 10000;
    ArrayXd stateInit(3);
    stateInit(0) = m;
    stateInit(1) = 0.; //Q
    stateInit(2) = 0.2; //L
    // nb simulations to print
    int nbPrint = 10 ;
    double averageValue = simuSLNonEmissive(grid, optimizer,  finalFunctionValue, ndt, stateInit, nbSimul, fileToDump, nbPrint, world);
    if (world.rank() == 0)
        cout << "Average SL " << averageValue << endl ;


    // Optimizer with sparse grids
    //****************************
    ArrayXd sizeDomain(3);
    sizeDomain << 4., 2., 2. ;
    ArrayXd  weight(3);
    weight << 1.2, 1, 1;
    int level = 7;
    size_t degree = 1;
    shared_ptr<SparseSpaceGrid> gridSparse = make_shared<SparseSpaceGridBound>(lowValues, sizeDomain, level, weight, degree);
    string fileToDumpSparse = "NonEmissivSLDumpSparse";
    semiLagrangTimeNonEmissiveSparse(gridSparse, optimizer,  finalFunctionValue, boundary, dt, ndt,  fileToDumpSparse, world);


    // Simulate with sparse grids
    //***************************
    double averageValueSparse = simuSLNonEmissiveSparse(gridSparse, optimizer,  finalFunctionValue, ndt, stateInit, nbSimul, fileToDumpSparse, nbPrint, world);
    if (world.rank() == 0)
        cout << "Average SL Sparse" << averageValueSparse << endl ;
    return 0. ;
}
