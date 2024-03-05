
#define BOOST_TEST_DYN_LINK
#include <memory>
#include <boost/test/unit_test.hpp>
#include <boost/mpi.hpp>
#include <boost/timer/timer.hpp>
#include <Eigen/Dense>
#include "libflow/regression/LocalLinearRegression.h"
#include "test/c++/tools/simulators/NIGSimulator.h"
#include "test/c++/tools/dp/OptimizeOptionL2.h"
#include "test/c++/tools/dp/DynamicHedgeL2Dist.h"
#include "test/c++/tools/dp/SimulateHedgeL2ControlDist.h"

/* \file mainOptionNIGL2.cpp
 * \brief Permits to valuate and hedge an asset with Normal Inverse Gaussian modelization
 *        using the minization of the variance of the hedge portfolio.
 * \author  Xavier Warin
 */

using namespace std;
using namespace Eigen;
using namespace libflow;

/// Call option on basket
class CallFunction
{
private :
    double m_strike;
    ArrayXd m_constSpread;
    ArrayXd m_linSpread;

public:
    /// \brief take into account spread
    CallFunction(const double &p_strike, const ArrayXd &p_constSpread, const ArrayXd &p_linSpread): m_strike(p_strike),
        m_constSpread(p_constSpread), m_linSpread(p_linSpread)
    {}

    /// \brief get function value including cost
    /// \param  p_delta  amount in future position
    /// \param  p_asset  asset  per simulation
    double operator()(const int &, const Eigen::ArrayXd   &p_delta, const Eigen::ArrayXd &p_asset) const
    {
        return std::max(p_asset.mean() - m_strike, 0.) + (p_delta.abs() * (m_constSpread + m_linSpread * p_asset)).sum();
    }

    /// \brief function value used in simulation
    /// \param p_asset   asset values
    double  operator()(const Eigen::ArrayXd &p_asset) const
    {
        return   std::max(p_asset.mean() - m_strike, 0.);
    }

};

double accuracyClose =  2.;


#if defined   __linux
#include <fenv.h>
#define enable_abort_on_floating_point_exception() feenableexcept(FE_DIVBYZERO | FE_INVALID)
#endif

BOOST_AUTO_TEST_CASE(telibflowionNIGL2)
{
    boost::mpi::communicator world;

    // asset
    double S0 = 1.;
    double alpha = 2. ;
    double beta = -0.5;
    double delta = 0.06;
    double mu = 0.07;

    // strike
    double strike = 1.;
    double T = 1.;

    // max amount to buy and sell in portfolio
    ArrayXd  posBuy = ArrayXd::Constant(1, 1.);
    ArrayXd  posSell = ArrayXd::Constant(1, 1.);
    // max amount to sell  or buy
    ArrayXd  posVarBuy =  ArrayXd::Constant(1, 0.3);
    ArrayXd  posVarSell =  ArrayXd::Constant(1, 0.3);
    ArrayXd  stepForHedge = ArrayXd::Constant(1, 0.1)  ;
    // no spread bid ask
    ArrayXd  spread1 = ArrayXd::Zero(1) ;
    ArrayXd  spread2 = ArrayXd::Zero(1);

    // parameters for resolution
    int nMesh = 8; // mesh number for regression
    int nbSimOptim = 40000; // number of simulation on optimization part.
    int nbTimeStep = 10 ; // number of time steps for resolution
    int nbSimSimu = 40000 ; // number of simulations for simulation part.
    // file to store control and simultations
    //***************************************
    string fileForControl("Control");
    string fileForSimu("Simulation");

    // Optimizer
    //**********
    shared_ptr<OptimizeOptionL2<NIGSimulator> > optimizer = make_shared<OptimizeOptionL2<NIGSimulator> >(posVarBuy, posVarSell, stepForHedge, spread1, spread2);


    // grid to describe the amount of future available
    //************************************************
    ArrayXd lowValues = -posSell;
    int nStepLoc = static_cast<int>((posBuy(0) + posSell(0)) / stepForHedge(0) + tiny);
    double stepLoc = (posBuy(0) + posSell(0)) / nStepLoc;
    ArrayXd step = ArrayXd::Constant(1, stepLoc);
    ArrayXi  nbStep = ArrayXi::Constant(1, nStepLoc);
    shared_ptr<FullGrid> grid = make_shared<RegularSpaceGrid>(lowValues, step, nbStep);

    // regressor
    //**********
    ArrayXi nbMesh =  ArrayXi::Constant(1, nMesh);
    shared_ptr< BaseRegression > regressor = make_shared<LocalLinearRegression>(nbMesh);

    // final value function
    //*********************
    function<double(const int &, const Eigen::ArrayXd &, const Eigen::ArrayXd &)>  vFunction = CallFunction(strike, spread1, spread2);
    function<double(const Eigen::ArrayXd &)>  vFunctionSim = CallFunction(strike, spread1, spread2);

    // Archive to store Bellman value or control
    //******************************************
    shared_ptr<gs::BinaryFileArchive> ar;
    if (world.rank() == 0)
        ar = make_shared<gs::BinaryFileArchive>(fileForControl.c_str(), "w");

    // NIG simulator
    //**************
    VectorXd initialValues = VectorXd::Constant(1, S0);
    VectorXd aAlpha = VectorXd::Constant(1, alpha);
    VectorXd bBeta = VectorXd::Constant(1, beta);
    VectorXd dDelta = VectorXd::Constant(1, delta);
    VectorXd  mMu = VectorXd::Constant(1, mu);

    bool bForward = false;
    shared_ptr<NIGSimulator> simulatorBackward = make_shared<NIGSimulator>(initialValues, aAlpha, bBeta, dDelta, mMu, T, nbTimeStep, nbSimOptim, bForward, fileForSimu, world);
    // update simulator for optimizer in backward mode
    optimizer->setSimulator(simulatorBackward);

    shared_ptr<boost::timer::auto_cpu_timer> tCpu;

    // Optimize
    //*********
    double valOptim = 0;
    {
        if (world.rank() == 0)
            tCpu = make_shared<boost::timer::auto_cpu_timer>();
        valOptim = DynamicHedgeL2Dist<NIGSimulator>(grid, optimizer, regressor, vFunction, ar, world);
    }

    // send optimization value to all processes
    broadcast(world, valOptim, 0);

    if (world.rank() == 0)
        cout << " Value in optimization : " <<  valOptim << endl ;


    // create a NIG simulator for forward simulations
    // ***********************************************
    bForward = true;
    shared_ptr<NIGSimulator> simulatorForward = make_shared<NIGSimulator>(initialValues, aAlpha, bBeta, dDelta, mMu, T, nbTimeStep, nbSimSimu, bForward, fileForSimu, world);
    // update simulator for optimizer in forward mode
    optimizer->setSimulator(simulatorForward);


    // Simulate
    // *********
    double valSimu ;
    {
        if (world.rank() == 0)
            tCpu = make_shared<boost::timer::auto_cpu_timer>();
        valSimu = SimulateHedgeL2ControlDist<NIGSimulator>(grid, optimizer, vFunctionSim, valOptim, fileForControl, world);
    }

    if (world.rank() == 0)
        cout << " Value in simulation " << valSimu << endl ;

    if (world.rank() == 0)
        BOOST_CHECK_CLOSE(valOptim, valSimu, accuracyClose);
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
