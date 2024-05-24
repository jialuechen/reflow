
#include <math.h>
#include <functional>
#include <memory>
#include <fstream>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include "reflow/core/grids/RegularLegendreGrid.h"
#include "reflow/core/grids/SparseSpaceGridBound.h"
#include "reflow/regression/LocalLinearRegressionGeners.h"
#include "test/c++/tools/simulators/AR1Simulator.h"
#include "test/c++/tools/dp/DpTimeNonEmissive.h"
#include "test/c++/tools/dp/DpTimeNonEmissiveSparse.h"
#include "test/c++/tools/dp/OptimizeDPEmissive.h"
#include "test/c++/tools/dp/simuDPNonEmissive.h"


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


// Implement  a  HJB resolution fo problem  with Semi Lagrangian for  Aid, Ren, Touzi :
//  "Transition to non-emissive electricity production under optimal subsidy and endogenous carbon price"
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

    // maturity
    double maturity = 1 ;

    // CO2 price parameters
    double lambda = 1 ;
    double H = 1 ;

    // final value function  (x is the stock = (Q,L))
    auto fSimu([H, lambda](const int &p_iReg, const ArrayXd & x, const ArrayXd &)
    {
        if (p_iReg == 0)
            return  0. ;
        else
            return ((x(0) >= H) ? lambda : 0.);
    });

    function<double(const int &, const ArrayXd &, const ArrayXd &)>  finalFunctionValue(cref(fSimu));

    // resolution
    int nstep = 50;
    int nbsimulOpt = 1000;

    // simulator backward
    //*******************
    bool bForward = false;
    shared_ptr< AR1Simulator> backSimulator = make_shared<AR1Simulator> (m, m, sig, alpha, maturity, nstep, nbsimulOpt, bForward);

    // regressor
    //**********
    ArrayXi nbMesh = ArrayXi::Constant(1, 1);
    shared_ptr< LocalLinearRegression > regressor =  make_shared< LocalLinearRegression >(nbMesh);

    // final value
    // grid
    ArrayXd lowValues = ArrayXd::Constant(2, 0.);
    ArrayXd step = ArrayXd::Constant(2, 0.1);
    ArrayXi nbstep = ArrayXi::Constant(2, 20);
    ArrayXi npoly = ArrayXi::Constant(2, 1);
    shared_ptr<reflow::FullGrid>  grid = make_shared<RegularLegendreGrid>(lowValues, step, nbstep, npoly);

    // max investment intensity
    double lMax = 2.;
    double lStep = 0.05 ;
    // Optimizer
    shared_ptr<OptimizeDPEmissive> optimizer = make_shared<OptimizeDPEmissive>(alpha, funcPi, funcCost, s, lambda, maturity / nstep, maturity, lMax, lStep, grid->getExtremeValues());
    // optimizer and simulator are linked
    optimizer->setSimulator(backSimulator);

    // optimization with Full grids
    // ****************************
    string fileToDump = "NonEmissivDPDump";
    DpTimeNonEmissive(grid, optimizer, regressor, finalFunctionValue,  fileToDump, world);

    // simulate with Full Grids
    //************************
    bForward = true;
    int nbSimul = 10000;
    shared_ptr< AR1Simulator> forwardSimulator = make_shared<AR1Simulator> (m, m, sig, alpha, maturity, nstep, nbSimul, bForward);
    optimizer->setSimulator(forwardSimulator);

    ArrayXd stateInit(2);
    stateInit(0) = 0; // Q
    stateInit(1) = 0.2; //L

    // nb simulations to print
    int nbPrint = 10 ;
    double average = simuDPNonEmissive(grid, optimizer, finalFunctionValue,  stateInit, fileToDump, nbPrint, world);
    if (world.rank() == 0)
        cout << " AVerage DP " << average << endl ;


    // Optimization  with sparse grids
    //********************************
    ArrayXd sizeDomain(2);
    sizeDomain << 2., 2. ;
    ArrayXd  weight = ArrayXd::Constant(2, 1.);
    int level = 4;
    size_t degree = 1;
    shared_ptr<SparseSpaceGrid> gridSparse = make_shared<SparseSpaceGridBound>(lowValues, sizeDomain, level, weight, degree);
    // simulator backward
    bForward = false;
    shared_ptr< AR1Simulator> backSimulatorFS = make_shared<AR1Simulator> (m, m, sig, alpha, maturity, nstep, nbsimulOpt, bForward);
    optimizer->setSimulator(backSimulatorFS);
    string fileToDumpSparse = "NonEmissivDPSparseDump";
    DpTimeNonEmissiveSparse(gridSparse, optimizer, regressor, finalFunctionValue,  fileToDumpSparse, world);


    // simulate the optimal policy calculated with sparse grids
    //*********************************************************
    bForward = true;
    shared_ptr< AR1Simulator> forwardSimulatorFS = make_shared<AR1Simulator> (m, m, sig, alpha, maturity, nstep, nbSimul, bForward);
    optimizer->setSimulator(forwardSimulatorFS);

    double averageSparse = simuDPNonEmissive(gridSparse, optimizer, finalFunctionValue,  stateInit, fileToDumpSparse, nbPrint, world);
    if (world.rank() == 0)
        cout << " AVerage DP sparse " << averageSparse << endl ;

    return 0. ;
}
